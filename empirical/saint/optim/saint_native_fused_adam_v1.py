import copy
import inspect

def install(model):
    import torch
    old=model.optimizer
    if type(old) is not torch.optim.Adam or old.state:
        raise ValueError('Actual fresh native Adam with empty state required')
    if hasattr(old,'_phase2_declared_rate'):
        raise ValueError('Install execution choice before ordinary rate policy')
    accepted=set(inspect.signature(torch.optim.Adam).parameters)-{'params'}
    defaults={k:copy.deepcopy(v) for k,v in old.defaults.items() if k in accepted}
    if defaults.get('differentiable') or defaults.get('capturable') or defaults.get('decoupled_weight_decay'):
        raise ValueError('Only the original ordinary Adam execution is supported')
    groups=[]
    before=[]
    for group in old.param_groups:
        before.extend(id(p) for p in group['params'])
        out={k:(list(v) if k=='params' else copy.deepcopy(v)) for k,v in group.items()
             if k=='params' or k in accepted}
        out.update(fused=True,foreach=False)
        groups.append(out)
    defaults.update(fused=True,foreach=False)
    new=torch.optim.Adam(groups,**defaults)
    if new.state or before!=[id(p) for g in new.param_groups for p in g['params']]:
        raise RuntimeError('Fused option changed parameter objects/order or empty state')
    model.optimizer=new
    return dict(optimizer_class='Adam',fused=True,foreach=False,
                original_fused=[g.get('fused') for g in old.param_groups],
                original_foreach=[g.get('foreach') for g in old.param_groups],
                same_parameter_objects_and_order=True,initial_state_empty=True,
                arithmetic_scope='Native PyTorch Adam in FP32; floating point kernel rounding may differ; no A core or ordinary hyperparameter change')
