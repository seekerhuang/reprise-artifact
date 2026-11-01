from types import MethodType

VERSION = 'Yelp-original-full-COO-eval-native-order-low-temporaries-v3'

def _check_eval(layer):
    import torch
    if layer.training or torch.is_grad_enabled():
        raise RuntimeError('Low-temporary methods are confined to native no-grad EVAL')

def _spmm(self, adj_norm, feat):
    import torch
    _check_eval(self)
    if adj_norm.layout != torch.sparse_coo or feat.dtype != torch.float32 or adj_norm.dtype != feat.dtype:
        raise RuntimeError('Fixed original float32 COO EVAL required')
    
    
    
    
    zero = feat.new_zeros(())
    return torch.sparse.addmm(zero.expand(adj_norm.shape[0], feat.shape[1]),
                             adj_norm, feat, beta=0, alpha=1)

def _feat_trans(self, _feat, _id):
    import torch
    _check_eval(self)
    linear = self.f_lin[_id](_feat)
    del _feat
    biased = linear + self.f_bias[_id]
    del linear
    feat = self.act(biased)
    del biased
    
    mean = feat.mean(dim=1).view(feat.shape[0], 1)
    var = feat.var(dim=1, unbiased=False).view(feat.shape[0], 1) + 1e-09
    out = feat - mean
    del feat
    
    out.mul_(self.scale[_id])
    inverse = torch.rsqrt(var)
    out.mul_(inverse)
    out.add_(self.offset[_id])
    return out

def _forward(self, inputs):
    import torch
    _check_eval(self)
    adj_norm, feat_in = inputs
    feat_in = self.f_dropout(feat_in)
    feat_hop = [feat_in]
    for o in range(self.order):
        feat_hop.append(self._spmm(adj_norm, feat_hop[-1]))
    feat_partial = []
    for idf in range(len(feat_hop)):
        
        
        feat_partial.append(self._f_feat_trans(feat_hop.pop(0), idf))
    del feat_hop
    
    if self.aggr == 'mean':
        feat_out = feat_partial[0]
        for o in range(len(feat_partial) - 1):
            feat_out += feat_partial[o + 1]
    elif self.aggr == 'concat':
        feat_out = torch.cat(feat_partial, 1)
    else:
        raise NotImplementedError
    del feat_partial
    return (adj_norm, feat_out)

def evaluate(original_evaluate, model, minibatch, thresholds=(0.5,)):
    layers = list(model.aggregators)
    if (len(layers) != 3 or [layer.order for layer in layers] != [1, 1, 0]
            or any(layer.bias != 'norm' or layer.aggr != 'concat'
                   or type(layer).__name__ != 'HighOrderAggregator' for layer in layers)):
        raise RuntimeError('Exactly original Yelp HighOrderAggregator norm/concat 1-1-0 required')
    saved = []
    try:
        for layer in layers:
            for name, function in [('forward', _forward), ('_f_feat_trans', _feat_trans), ('_spmm', _spmm)]:
                existed = name in layer.__dict__
                saved.append((layer, name, existed, layer.__dict__.get(name)))
                setattr(layer, name, MethodType(function, layer))
        return original_evaluate(model, minibatch, thresholds)
    finally:
        for layer, name, existed, value in reversed(saved):
            if existed:
                setattr(layer, name, value)
            else:
                delattr(layer, name)
