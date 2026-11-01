import copy
import math
from numbers import Real
VERSION = 'PPI-terminal-BN-beta1-07-v1'
TERMINAL_NAMES = ('bns.1.weight', 'bns.1.bias')
ATTRIBUTE = '_r69_terminal_beta_policy'

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
        raise TypeError('Original PPI group options must be finite host scalars/containers')
    return copy.deepcopy(options)

def _model_parameters(model):
    named = dict(model.named_parameters())
    if any((name not in named for name in TERMINAL_NAMES)):
        raise ValueError('Missing original terminal BN affine parameters')
    if len({id(p) for p in named.values()}) != len(named):
        raise ValueError('Named parameter identities must be unique')
    return named

def install(model, optimizer, *, beta1=0.7):
    import torch
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError('Require the original native torch.optim.Adam instance')
    if hasattr(optimizer, ATTRIBUTE):
        raise RuntimeError('Terminal beta policy may be installed only once')
    if optimizer.state:
        raise RuntimeError('Install before the first optimizer state/update')
    if beta1 not in (0.7, 0.9) or isinstance(beta1, bool):
        raise ValueError('Only fixed beta1=.7 and CPU identity beta1=.9 are supported')
    if len(optimizer.param_groups) != 2:
        raise ValueError('Require the original two parameter groups')
    options = [_options(group) for group in optimizer.param_groups]
    if options[0] != options[1]:
        raise ValueError('Original groups differ in non-parameter options')
    original_options = options[0]
    if original_options.get('weight_decay') != 0.0:
        raise ValueError('Original PPI weight decay must be zero in both groups')
    if tuple(original_options.get('betas', ())) != (0.9, 0.999):
        raise ValueError('Require original beta1=.9 and beta2=.999')
    if not _host_tree(optimizer.defaults):
        raise TypeError('Original optimizer defaults must be host options')
    if original_options['lr'] != optimizer.defaults['lr']:
        raise ValueError('Original global group/default LR must agree')
    if original_options.get('capturable', False):
        raise ValueError('This PPI policy retains the original noncapturable Adam')
    named = _model_parameters(model)
    original = [p for group in optimizer.param_groups for p in group['params']]
    original_ids = [id(p) for p in original]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError('An original Parameter occurs more than once')
    if set(original_ids) != {id(p) for p in named.values()}:
        raise ValueError('Original groups must cover every model Parameter exactly once')
    terminal = [named[name] for name in TERMINAL_NAMES]
    terminal_ids = {id(p) for p in terminal}
    nonterminal = [p for p in original if id(p) not in terminal_ids]
    if len(terminal_ids) != 2 or not nonterminal:
        raise ValueError('Require two distinct terminal parameters and a nonterminal group')
    new_groups = []
    for (parameters, new_beta1) in ((nonterminal, 0.9), (terminal, float(beta1))):
        group = copy.deepcopy(original_options)
        group['params'] = parameters
        group['betas'] = (new_beta1, original_options['betas'][1])
        new_groups.append(group)
    policy = {'version': VERSION, 'terminal_beta1': float(beta1), 'original_options': original_options, 'original_defaults': copy.deepcopy(optimizer.defaults), 'parameter_ids_by_name': {name: id(p) for (name, p) in named.items()}, 'nonterminal_ids_in_order': [id(p) for p in nonterminal], 'terminal_ids_in_order': [id(p) for p in terminal], 'original_group_sizes': [len(group['params']) for group in optimizer.param_groups]}
    optimizer.param_groups[:] = new_groups
    setattr(optimizer, ATTRIBUTE, policy)
    return record(model, optimizer)

def record(model, optimizer):
    import torch
    if not hasattr(optimizer, ATTRIBUTE):
        raise RuntimeError('Terminal beta policy is not installed')
    policy = getattr(optimizer, ATTRIBUTE)
    named = _model_parameters(model)
    if {name: id(p) for (name, p) in named.items()} != policy['parameter_ids_by_name']:
        raise RuntimeError('Model Parameter identities changed after installation')
    if len(optimizer.param_groups) != 2:
        raise RuntimeError('Terminal beta policy must retain exactly two native groups')
    for (index, expected) in enumerate((policy['nonterminal_ids_in_order'], policy['terminal_ids_in_order'])):
        if [id(p) for p in optimizer.param_groups[index]['params']] != expected:
            raise RuntimeError('Parameter coverage/order changed after installation')
    defaults = optimizer.defaults
    if not _host_tree(defaults) or set(defaults) != set(policy['original_defaults']) or any((defaults[key] != value for (key, value) in policy['original_defaults'].items() if key != 'lr')):
        raise RuntimeError('A non-LR optimizer default changed')
    if not isinstance(defaults['lr'], Real) or defaults['lr'] <= 0:
        raise RuntimeError('Global LR must remain a positive host scalar')
    by_id = {id(p): name for (name, p) in named.items()}
    groups = []
    for (index, group) in enumerate(optimizer.param_groups):
        actual = _options(group)
        expected = copy.deepcopy(policy['original_options'])
        expected['lr'] = defaults['lr']
        expected['betas'] = (0.9 if index == 0 else policy['terminal_beta1'], policy['original_options']['betas'][1])
        if actual != expected:
            raise RuntimeError('Actual group options depart from fixed terminal-beta/global-LR policy')
        groups.append({'index': index, 'role': 'nonterminal' if index == 0 else 'terminal_BN', 'parameter_names': [by_id[id(p)] for p in group['params']], 'options': actual})
    clocks = {}
    for (name, p) in named.items():
        state = optimizer.state.get(p, {})
        if not state:
            clocks[name] = 0
            continue
        step = state['step']
        if isinstance(step, torch.Tensor):
            if step.device.type != 'cpu' or step.numel() != 1:
                raise RuntimeError('Record requires the original CPU Adam step scalar')
            step = step.item()
        if not isinstance(step, Real) or not math.isfinite(float(step)) or int(step) != step:
            raise RuntimeError('Invalid native Adam parameter clock')
        clocks[name] = int(step)
    return {'version': policy['version'], 'terminal_parameter_names': list(TERMINAL_NAMES), 'terminal_beta1': policy['terminal_beta1'], 'nonterminal_beta1': 0.9, 'beta2': policy['original_options']['betas'][1], 'original_group_sizes': list(policy['original_group_sizes']), 'actual_group_sizes': [len(group['params']) for group in optimizer.param_groups], 'parameter_count': len(named), 'parameters_exactly_once': True, 'groups': groups, 'defaults': copy.deepcopy(defaults), 'parameter_Adam_steps': clocks, 'scope': 'Ordinary fixed terminal-BN beta1; original native Adam, clocks and core; no policy feedback or extra optimizer calls'}
