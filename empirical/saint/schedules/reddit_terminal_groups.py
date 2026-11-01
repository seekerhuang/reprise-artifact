import copy
from numbers import Real
from empirical.saint.schedules import reddit_terminal_helpers as parent
ATTRIBUTE = '_r70_Reddit_three_group_Head2_policy'
VERSION = 'R74-Reddit-init0-Head1quarter-beta09-three-native-groups-v1'
TERMINAL_NAMES = parent.TERMINAL_NAMES
HEAD_NAMES = ('classifier.f_lin.0.weight', 'classifier.params.0', 'classifier.params.1', 'classifier.params.2')
UNUSED_NAMES = ('classifier.params.1', 'classifier.params.2')

def install(model, optimizer):
    import torch
    if type(optimizer) is not torch.optim.Adam or model.optimizer is not optimizer:
        raise TypeError('Original model-owned native Adam required')
    if optimizer.state or hasattr(optimizer, ATTRIBUTE):
        raise RuntimeError('Three groups install once on original empty Adam')
    if len(optimizer.param_groups) != 1:
        raise ValueError('Original single group required before this split')
    options = parent._options(optimizer.param_groups[0])
    if tuple(options['betas']) != (0.9, 0.999) or options['eps'] != 1e-06 or options['weight_decay'] != 0.0 or options.get('capturable', False) or (options['lr'] != optimizer.defaults['lr']):
        raise ValueError('Fixed ordinary global LR/native eps6/zero WD policy required')
    named = parent._model_parameters(model)
    if len(named) != 28 or tuple((n for n in named if n.startswith('classifier.'))) != HEAD_NAMES:
        raise ValueError('Actual original 28-parameter Reddit support required')
    original = [p for group in optimizer.param_groups for p in group['params']]
    if len(original) != 28 or len({id(p) for p in original}) != 28 or {id(p) for p in original} != {id(p) for p in named.values()}:
        raise ValueError('Original parameter coverage must be exact')
    selected = [named[n] for n in TERMINAL_NAMES]
    head = [named[n] for n in HEAD_NAMES]
    excluded = {id(p) for p in selected + head}
    body = [p for p in original if id(p) not in excluded]
    if len(body) != 22 or len(excluded) != 6:
        raise ValueError('Three disjoint native groups must be22/2/4')
    defaults = copy.deepcopy(optimizer.defaults)
    (state_id, optimizer_id) = (id(optimizer.state), id(optimizer))
    groups = []
    for (parameters, beta1) in ((body, 0.9), (selected, 0.9), (head, 0.9)):
        group = copy.deepcopy(options)
        group['params'] = parameters
        group['betas'] = (beta1, options['betas'][1])
        groups.append(group)
    policy = dict(version=VERSION, original_options=options, original_defaults=defaults, original_group_sizes=[28], optimizer_id=optimizer_id, state_id=state_id, parameter_ids={n: id(p) for (n, p) in named.items()}, parameter_storage={n: p.data_ptr() for (n, p) in named.items()}, group_ids=[[id(p) for p in group['params']] for group in groups])
    optimizer.param_groups[:] = groups
    setattr(optimizer, ATTRIBUTE, policy)
    if optimizer.state or id(optimizer.state) != state_id or optimizer.defaults != defaults:
        raise RuntimeError('Native empty state/defaults changed during grouping')
    return record(model, optimizer)

def record(model, optimizer):
    import torch
    policy = getattr(optimizer, ATTRIBUTE)
    named = parent._model_parameters(model)
    if type(optimizer) is not torch.optim.Adam or id(optimizer) != policy['optimizer_id'] or id(optimizer.state) != policy['state_id']:
        raise RuntimeError('Original native optimizer/state objects changed')
    if {n: id(p) for (n, p) in named.items()} != policy['parameter_ids'] or {n: p.data_ptr() for (n, p) in named.items()} != policy['parameter_storage']:
        raise RuntimeError('Original model parameter/storage identity changed')
    if len(optimizer.param_groups) != 3 or [[id(p) for p in g['params']] for g in optimizer.param_groups] != policy['group_ids']:
        raise RuntimeError('Actual three-group parameter coverage/order changed')
    defaults = optimizer.defaults
    if not parent._host_tree(defaults) or set(defaults) != set(policy['original_defaults']) or any((defaults[k] != v for (k, v) in policy['original_defaults'].items() if k != 'lr')) or (not isinstance(defaults['lr'], Real)) or (defaults['lr'] <= 0):
        raise RuntimeError('Only the original global LR event may change defaults')
    ids = {id(p): n for (n, p) in named.items()}
    groups = []
    for (i, group) in enumerate(optimizer.param_groups):
        actual = parent._options(group)
        expected = copy.deepcopy(policy['original_options'])
        expected['lr'] = defaults['lr']
        expected['betas'] = (0.9, 0.999)
        if actual != expected:
            raise RuntimeError('Temporary Head rate was not restored or ordinary options changed')
        groups.append(dict(index=i, role=('nonselected_Body', 'selected_affine', 'classifier_including_unused')[i], parameter_names=[ids[id(p)] for p in group['params']], options=actual))
    clocks = {}
    for (name, p) in named.items():
        step = optimizer.state.get(p, {}).get('step', 0)
        if isinstance(step, torch.Tensor):
            expected_device = p.device if optimizer.defaults.get('fused', False) else torch.device('cpu')
            if step.device != expected_device or step.numel() != 1:
                raise RuntimeError('Native Adam step must use the declared backend device and scalar shape')
            step = step.item()
        if not isinstance(step, Real) or int(step) != step:
            raise RuntimeError('Invalid native clock')
        clocks[name] = int(step)
    if any((clocks[n] != 0 for n in UNUSED_NAMES)):
        raise RuntimeError('Unused classifier parameters must keep zero clocks')
    return dict(version=policy['version'], terminal_parameter_names=list(TERMINAL_NAMES), terminal_beta1=0.9, nonterminal_beta1=0.9, beta2=0.999, original_group_sizes=[28], actual_group_sizes=[22, 2, 4], parameter_count=28, parameters_exactly_once=True, classifier_group_index=2, classifier_parameter_names=list(HEAD_NAMES), unused_parameter_names=list(UNUSED_NAMES), groups=groups, defaults=copy.deepcopy(defaults), parameter_Adam_steps=clocks, clipping_groups_unchanged=dict(P=20, L=4, C=2), scope='Three real native groups, not the parent two-group validator. Head1 temporary LR is restored before this metadata boundary; group dispatch is paid.')
