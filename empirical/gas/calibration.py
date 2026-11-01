import ast
import copy
import inspect
import textwrap
_FUNCTIONS = {}

def _function(name, args, body):
    return ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in args], vararg=ast.arg(arg='args'), kwonlyargs=[], kw_defaults=[], defaults=[]), body=copy.deepcopy(body), decorator_list=[])

def compile_native(model_class):
    if model_class in _FUNCTIONS:
        return
    original = model_class.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)))
    body = fn.body
    last_bn = [i for (i, n) in enumerate(body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.batch_norm' and ('self.bns[-1](h)' in ast.unparse(n))]
    loops = [i for (i, n) in enumerate(body) if isinstance(n, ast.For) and 'zip(self.convs[:-1], self.bns, self.histories)' in ast.unparse(n.iter)]
    if len(last_bn) != 1 or len(loops) != 1:
        raise RuntimeError('Unknown original stateless GCN calibration boundary')
    (end, loop_at) = (last_bn[0], loops[0])
    ppi_prefix = body[:end] + [ast.parse('return h, x').body[0]]
    ppi_suffix = body[end:]
    loop = body[loop_at]
    if ast.unparse(loop.body[0]) != 'h = conv(x, adj_t)':
        raise RuntimeError('Original first convolution must precede the first BN')
    shortened = copy.deepcopy(loop)
    shortened.body = [copy.deepcopy(loop.body[0]), ast.parse('return h').body[0]]
    flickr_prefix = body[:loop_at] + [shortened]
    flickr_suffix = ast.parse('bn = self.bns[0]\nhist = self.histories[0]').body + loop.body[1:] + body[loop_at + 1:]
    module = ast.fix_missing_locations(ast.Module(body=[_function('ppi_prefix', ['self', 'x', 'adj_t'], ppi_prefix), _function('ppi_suffix', ['self', 'h', 'x'], ppi_suffix), _function('flickr_prefix', ['self', 'x', 'adj_t'], flickr_prefix), _function('flickr_suffix', ['self', 'h', 'adj_t'], flickr_suffix)], type_ignores=[]))
    env = dict(original.__globals__)
    exec(compile(module, __file__ + '::native-prefix-tail', 'exec'), env)
    _FUNCTIONS[model_class] = {k: env[k] for k in ('ppi_prefix', 'ppi_suffix', 'flickr_prefix', 'flickr_suffix')}

def _guard(model, data, *, linear):
    import torch
    nodes = data.x.size(0)
    if model.num_nodes != 0 or model.pool is not None or any((h.emb.numel() for h in model.histories)) or (model.num_layers != 2) or (model.linear != linear) or (not model.batch_norm) or (len(model.bns) != 2) or model.training or any((bn.training for bn in model.bns)) or (tuple(data.adj_t.sizes()) != (nodes, nodes)) or (data.x.dtype != torch.float32) or (linear and (not model.residual or model.dropout != 0.0 or model.drop_input)) or (not linear and (model.residual or model.dropout != 0.3 or (not model.drop_input))):
        raise RuntimeError('Only the fixed stateless PPI/Flickr delivery clone is covered')
    if type(model) not in _FUNCTIONS:
        raise RuntimeError('Calibration AST must be compiled before the paid main lifecycle')
    return (nodes, _FUNCTIONS[type(model)])

def _fit(bn, h, nodes):
    import torch
    if bn.training or h.dtype != torch.float32 or tuple(h.shape) != (nodes, bn.num_features):
        raise RuntimeError('Original complete FP32 population at the original BN input required')
    (variance, mean) = torch.var_mean(h, dim=0, correction=0)
    bn.running_mean.copy_(mean)
    bn.running_var.copy_(variance)

def ppi_prepare(model, data):
    import torch
    (nodes, functions) = _guard(model, data, linear=True)
    seen = []

    def first_fit(bn, args):
        if bn is not model.bns[0] or seen:
            raise RuntimeError('Exactly one first-BN population required')
        _fit(bn, args[0], nodes)
        seen.append(0)
    handle = model.bns[0].register_forward_pre_hook(first_fit)
    try:
        with torch.no_grad():
            (h, x) = functions['ppi_prefix'](model, data.x.to(model.device), data.adj_t.to(model.device))
            if seen != [0]:
                raise RuntimeError('Missing native first-BN population')
            _fit(model.bns[1], h, nodes)
    finally:
        handle.remove()
    return (h, x)

def ppi_finish(model, payload):
    import torch
    with torch.no_grad():
        return _FUNCTIONS[type(model)]['ppi_suffix'](model, *payload).cpu()

def flickr_prepare(model, data):
    import torch
    (nodes, functions) = _guard(model, data, linear=False)
    with torch.no_grad():
        matrix = data.adj_t.to(model.device)
        h = functions['flickr_prefix'](model, data.x.to(model.device), matrix)
        _fit(model.bns[0], h, nodes)
    return (h, matrix)

def flickr_finish(model, payload):
    import torch
    with torch.no_grad():
        return _FUNCTIONS[type(model)]['flickr_suffix'](model, *payload).cpu()
