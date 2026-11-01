from contextlib import contextmanager
import random
import numpy as np
import torch
from main.common import Timer, sync
import ast
import copy
import inspect
import textwrap
_PREFIX = {}

def compile_native(model_class):
    if model_class in _PREFIX:
        return
    original = model_class.forward
    fn = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    loops = [i for (i, n) in enumerate(fn.body) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'zip(self.convs[:-1], self.bns, self.histories)']
    if len(loops) != 1:
        raise RuntimeError('Unknown frozen PNA forward')
    loop_at = loops[0]
    loop = copy.deepcopy(fn.body[loop_at])
    if ast.unparse(loop.body[0]) != 'h = conv(x, adj_t)':
        raise RuntimeError('Original BN input boundary changed')
    loop.body[1:1] = ast.parse('if bn is self.bns[-1]:\n    R75_FIT(bn, h, self.num_nodes)\n    return h').body
    pushes = [i for (i, statement) in enumerate(loop.body) if isinstance(statement, ast.Assign) and ast.unparse(statement) == 'x = self.push_and_pull(hist, h, *args)']
    if len(pushes) != 1:
        raise RuntimeError('Original complete-population history side effect changed')
    loop.body[pushes[0]] = ast.parse('x = h').body[0]
    derived = copy.deepcopy(fn)
    derived.name = '_r75_pna_calibration_prefix'
    derived.decorator_list = []
    derived.body = derived.body[:loop_at] + [loop]
    tree = ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))
    env = dict(original.__globals__, R75_FIT=fit)
    exec(compile(tree, __file__ + '::original-prefix', 'exec'), env)
    _PREFIX[model_class] = env[derived.name]

def fit(bn, hidden, nodes):
    import torch
    if bn.training or hidden.shape != (nodes, bn.num_features) or hidden.dtype != torch.float32:
        raise RuntimeError('Original full FP32 BN population required')
    (variance, mean) = torch.var_mean(hidden, dim=0, correction=0)
    bn.running_mean.copy_(mean)
    bn.running_var.copy_(variance)

def prefix(model, features, matrix):
    import torch
    if model.num_layers != 4 or len(model.bns) != 3 or (not model.batch_norm) or model.residual or (not model.drop_input) or (model.dropout != 0.5) or (features.shape[0] != model.num_nodes) or model.training or (type(model) not in _PREFIX):
        raise ValueError('Only original current PNA4 population calibration')
    seen = []
    handles = []
    for (i, bn) in enumerate(model.bns[:2]):

        def hook(module, args, index=i):
            if index != len(seen):
                raise RuntimeError('Original sequential BN order required')
            fit(module, args[0], model.num_nodes)
            seen.append(index)
        handles.append(bn.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            _PREFIX[type(model)](model, features, matrix, None, None, None, None)
        if seen != [0, 1]:
            raise RuntimeError('Missing first two native BN populations')
    finally:
        for handle in handles:
            handle.remove()

@contextmanager
def isolated_rng(device):
    (python_state, numpy_state) = (random.getstate(), np.random.get_state())
    try:
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class Delivery:
    def __init__(self, factory, data, loader, device):
        self.factory, self.data, self.loader, self.device = factory, data, loader, device
        self.model = self.features = self.matrix = None
        self.setup_seconds = 0.0

    @torch.no_grad()
    def observe(self, training_model, average):
        with isolated_rng(self.device):
            if self.model is None:
                self.model = self.factory()
                self.model.eval()
                sync(self.device)
                setup = Timer(self.device)
                with setup:
                    self.features = self.data.x.to(self.device)
                    self.matrix = self.data.adj_t.to(self.device)
                self.setup_seconds = setup.seconds
            else:
                sync(self.device)
            sync(self.device)
            self.model.load_state_dict(training_model.state_dict(), strict=True)
            sync(self.device)
            preparation = Timer(self.device)
            with preparation:
                average.install(self.model)
                prefix(self.model, self.features, self.matrix)
            sync(self.device)
            logits = self.model(loader=self.loader)
            sync(self.device)
        sync(self.device)
        return logits, self.setup_seconds + preparation.seconds
