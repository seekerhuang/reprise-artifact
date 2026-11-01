import copy
import math
from numbers import Real
TERMINAL_NAMES = ('conv_layers.3.params.2', 'conv_layers.3.params.1')

def _host_tree(value):
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all((_host_tree(item) for item in value))
    if isinstance(value, dict):
        return all((isinstance(key, str) and _host_tree(item) for (key, item) in value.items()))
    return False

def _options(group):
    options = {key: value for (key, value) in group.items() if key != 'params'}
    if not _host_tree(options):
        raise TypeError('Original Reddit group options must be finite host scalars/containers')
    return copy.deepcopy(options)

def _model_parameters(model):
    named = dict(model.named_parameters())
    if any((name not in named for name in TERMINAL_NAMES)):
        raise ValueError('Missing original selected terminal affine parameters')
    if any((tuple(named[name].shape) != (128,) for name in TERMINAL_NAMES)):
        raise ValueError('Original Reddit128 selected affine shapes required')
    if len({id(p) for p in named.values()}) != len(named):
        raise ValueError('Named parameter identities must be unique')
    return named
