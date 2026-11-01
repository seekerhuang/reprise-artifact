from pathlib import Path
import ast
import copy
import inspect
import textwrap
import time

def make_compute(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    fn = tree.body[0]
    old = copy.deepcopy(fn)
    loop = next((n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'self.data'))
    condition = loop.body[0]
    if not isinstance(condition, ast.If) or ast.unparse(condition.test) != 'isinstance(v, Tensor) and v.size(0) == self.data.num_nodes':
        raise RuntimeError('Original node-attribute gather changed')
    condition.test = ast.BoolOp(op=ast.And(), values=[ast.parse("k not in ('x', 'y')", mode='eval').body, condition.test])
    assignment = ast.parse('data.x, data.y = self._resident.gather(n_id, self._resident_kind)').body[0]
    fn.body.insert(-1, assignment)
    ast.fix_missing_locations(tree)
    inverse = copy.deepcopy(fn)
    reverse_loop = next((n for n in inverse.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'self.data'))
    reverse_loop.body[0].test = reverse_loop.body[0].test.values[1]
    del inverse.body[-2]
    if ast.dump(inverse) != ast.dump(old):
        raise RuntimeError('Native CPU geometry was modified')
    namespace = dict(original.__globals__)
    exec(compile(tree, '<own-GAS-resident-XY-gather-only>', 'exec'), namespace)
    return (namespace[fn.name], source, ast.unparse(fn))

class ResidentXY:

    def __init__(self, data, device, span):
        import torch
        (self.data, self.device, self.span) = (data, torch.device(device), span)
        if data.x.device.type != 'cpu' or data.y.device.type != 'cpu' or data.x.dtype != torch.float32 or (data.y.dtype != torch.float32):
            raise ValueError('Original already normalized CPU FP32 X/Y required')
        if data.x.requires_grad or data.y.requires_grad or data.x.shape[0] != data.y.shape[0]:
            raise ValueError('Only immutable nontrained attributes')
        self.incoming_versions = (data.x._version, data.y._version)
        self.shapes = (tuple(data.x.shape), tuple(data.y.shape))
        start = time.perf_counter()
        with span('resident_one_time_XY_H2D'):
            self.x = data.x.to(self.device, copy=True)
            self.y = data.y.to(self.device, copy=True)
        self.setup_seconds = time.perf_counter() - start
        self.base_bytes = sum((t.numel() * t.element_size() for t in (self.x, self.y)))
        self.calls = {'train': 0, 'eval': 0}
        self.materialized_bytes = {'train': 0, 'eval': 0}

    def gather(self, n_id, kind):
        if n_id.device.type != 'cpu':
            raise ValueError('Original CPU n_id retained for history/offset API')
        with self.span('resident_XY_GPU_gather'):
            ids = n_id.to(self.device)
            x = self.x.index_select(0, ids)
            y = self.y.index_select(0, ids)
        self.calls[kind] += 1
        self.materialized_bytes[kind] += x.numel() * x.element_size() + y.numel() * y.element_size()
        return (x, y)

    def verify(self):
        if self.incoming_versions != (self.data.x._version, self.data.y._version):
            raise RuntimeError('Original host attributes mutated')
        return dict(full_XY_bytes=self.base_bytes, one_time_copy_seconds=self.setup_seconds, full_shapes=self.shapes, gather_calls=dict(self.calls), materialized_XY_bytes=dict(self.materialized_bytes), persistent_eval_XY_cache_bytes=self.materialized_bytes['eval'], scope='Full resident XY plus original eight precomputed eval groups now holding GPU XY; report both, not just1.15GB. Training groups are ephemeral. Full neighbor Y rows retained, no loss-row compression.')

class Registry:

    def __init__(self, device, span):
        (self.device, self.span) = (device, span)
        self.rows = None
        self.loader_records = []

    def get(self, data):
        if self.rows is None:
            self.rows = ResidentXY(data, self.device, self.span)
        elif self.rows.data is not data:
            raise RuntimeError('Shared original permuted normalized data object required')
        return self.rows

    def classes(self, original_train, original_eval):
        (compute, old_source, new_source) = make_compute(original_train.compute_subgraph)
        registry = self

        def make(base, kind):

            class OwnResidentLoader(base):

                def __init__(self, data, *args, **kwargs):
                    if kwargs.get('num_workers', 0) != 0 or kwargs.get('pin_memory', False):
                        raise ValueError('Original single-process unpinned loader required')
                    self._resident = registry.get(data)
                    self._resident_kind = kind
                    super().__init__(data, *args, **kwargs)
                    registry.loader_records.append(dict(kind=kind, batches=len(self), dataset_items=len(self.dataset)))
                compute_subgraph = compute
            return OwnResidentLoader
        return (make(original_train, 'train'), make(original_eval, 'eval'), dict(original_compute=old_source, resident_compute=new_source, scope='Only X/Y CPU gather removed/replaced by GPU gather, all other compute_subgraph AST exactly preserved. Original DataLoader constructor, shuffle/maxsteps and eval precompute remain.'))
