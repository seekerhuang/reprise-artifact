import copy
import time
VERSION = 'R74_Yelp_terminal_selected_native_Adam_beta1_07_v1'
SELECTED = ('conv_layers.2.params.2', 'conv_layers.2.params.1')
UNUSED = ('classifier.params.1', 'classifier.params.2')
P_NAMES = tuple((name for layer in (0, 1) for name in [f'conv_layers.{layer}.f_lin.{i}.weight' for i in (0, 1)] + [f'conv_layers.{layer}.params.{i}' for i in range(6)]))
L_NAMES = ('conv_layers.2.f_lin.0.weight', 'conv_layers.2.params.0', 'conv_layers.2.params.1', 'conv_layers.2.params.2')
C_NAMES = ('classifier.f_lin.0.weight', 'classifier.params.0')
ALL_NAMES = P_NAMES + L_NAMES + C_NAMES + UNUSED
BASE_OPTIONS = dict(lr=0.015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None, decoupled_weight_decay=False)
_ATTRIBUTE = '_r74_terminal_selected_beta1_contract'

def policy():
    return dict(version=VERSION, optimizer='Unreplaced native torch.optim.Adam', selected_parameter_names=list(SELECTED), selected_parameter_elements=1024, selected_beta1=0.7, selected_epsilon=0.0001, nonselected_beta1=0.9, beta2=0.999, epsilon=1e-08, installation='From the loaded actual native initial tensor and empty Adam, before engine construction/first update', group_order=['all_other_native_parameters_including_two_unused', 'terminal_selected_affine'], group_parameter_counts=[22, 2], unused_parameter_names=list(UNUSED), declared_full_run_parameter_clocks='Every original active parameter is required to get16200 native updates; the two unused parameters must keep empty state/clock0', unchanged='All original model/core/query/objective, native P/L/C clipping supports, drop.05, .015/.01/.0075 LR events, late uniform positive1.1, raw bank, TRAIN RNG, full evaluation and strict native.5 VALID selector', cost='Installation is nested in required actual_initial_load_and_engine. Native two-group dispatch is inside original TRAIN outer time. Nested setup telemetry must not be added again.', interpretation='The fixed-gradient two-submission parameter-response coefficient beta1*(1-beta1) changes .09 to .21 under a fixed denominator; this is conditional, not a claim of9x total action or a quality guarantee')

def _options(group):
    return {k: copy.deepcopy(v) for (k, v) in group.items() if k != 'params'}

def _named(model):
    pairs = list(model.named_parameters())
    names = [k for (k, _p) in pairs]
    ids = [id(p) for (_k, p) in pairs]
    if len(pairs) != 24 or len(set(names)) != 24 or len(set(ids)) != 24 or (set(names) != set(ALL_NAMES)):
        raise RuntimeError('Exact original Yelp24 unique native parameters required')
    return (pairs, dict(pairs))

def _group_record(optimizer, pairs):
    by_id = {id(p): name for (name, p) in pairs}
    return [dict(index=i, parameter_names=[by_id[id(p)] for p in group['params']], options=_options(group)) for (i, group) in enumerate(optimizer.param_groups)]

def install(model):
    started = time.perf_counter()
    if hasattr(model, _ATTRIBUTE):
        raise RuntimeError('Ordinary terminal beta1 must be installed exactly once')
    optimizer = model.optimizer
    if type(optimizer).__name__ != 'Adam' or type(optimizer).__module__ != 'torch.optim.adam':
        raise RuntimeError('The original native Adam object is required')
    (pairs, named) = _named(model)
    if optimizer.state or len(optimizer.param_groups) != 1:
        raise RuntimeError('The loaded one-group native Adam must still be empty')
    if any((p.grad is not None for (_name, p) in pairs)):
        raise RuntimeError('Install before the first native gradient submission')
    group = optimizer.param_groups[0]
    original = list(group['params'])
    if len(original) != 24 or len({id(p) for p in original}) != 24 or {id(p) for p in original} != {id(p) for (_n, p) in pairs}:
        raise RuntimeError('Original native optimizer/model parameter support differs')
    if _options(group) != BASE_OPTIONS or optimizer.defaults != BASE_OPTIONS:
        raise RuntimeError('Original native Adam options differ from the pinned FP32 Yelp recipe')
    if any((named[name].numel() != 512 for name in SELECTED)):
        raise RuntimeError('The two selected terminal affine vectors must each have512 elements')
    defaults = copy.deepcopy(optimizer.defaults)
    selected_ids = {id(named[name]) for name in SELECTED}
    other_parameters = [p for p in original if id(p) not in selected_ids]
    selected_parameters = [p for p in original if id(p) in selected_ids]
    groups = [dict(_options(group), params=other_parameters), dict(_options(group), params=selected_parameters)]
    groups[1]['betas'] = (0.7, 0.999)
    groups[1]['eps'] = 0.0001
    optimizer.param_groups[:] = groups
    if optimizer.state or optimizer.defaults != defaults:
        raise RuntimeError('Group installation changed empty state/defaults')
    record = dict(version=VERSION, policy=policy(), install_count=1, native_optimizer_object_preserved=True, native_state_owner_preserved=True, param_groups_list_owner_preserved=True, parameters_preserved=True, initial_optimizer_state_empty=True, original_parameter_names=[next((n for (n, q) in pairs if q is p)) for p in original], original_options=BASE_OPTIONS.copy(), groups=_group_record(optimizer, pairs), selected_elements=sum((named[name].numel() for name in SELECTED)), dispatch='Unmodified native Adam.step over these two groups; no per-step helper hook', setup_seconds_nested_in_required_initial_engine=time.perf_counter() - started)
    setattr(model, _ATTRIBUTE, dict(optimizer=optimizer, state=optimizer.state, groups_owner=optimizer.param_groups, groups=groups, defaults=defaults, pairs=pairs, initial_record=copy.deepcopy(record)))
    return record
