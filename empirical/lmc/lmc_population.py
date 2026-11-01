import torch
BUFFER_NAMES = tuple(('bns.0.' + branch + '.' + field for branch in ('bn_ib', 'bn_ob') for field in ('running_mean', 'running_var')))

def off_statistics(model, geometry, ids):
    import torch
    import torch.nn.functional as F
    if len(model.state_dict()) != 24 or not model.drop_input or model.linear or model.residual:
        raise ValueError('Only frozen Flickr DoubleBN24 geometry is supported')
    with torch.no_grad():
        h = F.linear(geometry.eval_first_image, model.convs[0].lin.weight, model.convs[0].bias)
        result = {}
        for branch in ('bn_ib', 'bn_ob'):
            rows = h.index_select(0, ids[branch].to(h.device))
            (variance, mean) = torch.var_mean(rows, dim=0, correction=0)
            if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
                raise RuntimeError('Invalid population moments')
            result['bns.0.' + branch + '.running_mean'] = mean
            result['bns.0.' + branch + '.running_var'] = variance
    return result

def install_statistics(model, values):
    import torch
    if set(values) != set(BUFFER_NAMES):
        raise ValueError('Only four BN0 float buffers may change')
    actual = dict(model.named_buffers())
    with torch.no_grad():
        for name in BUFFER_NAMES:
            if values[name].shape != actual[name].shape or values[name].dtype != actual[name].dtype:
                raise ValueError('Moment schema differs: ' + name)
            actual[name].copy_(values[name])
