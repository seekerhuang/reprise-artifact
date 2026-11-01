import torch
import torch.nn.functional as F
from empirical.lmc.flickr import lmc_flickr_core as core
BODY_NAMES = ('convs.0.bias', 'convs.0.lin.weight', 'bns.0.bn_ib.weight', 'bns.0.bn_ib.bias', 'bns.0.bn_ob.weight', 'bns.0.bn_ob.bias')
HEAD_NAMES = ('convs.1.bias', 'convs.1.lin.weight')
UNUSED_NAMES = ('bns.1.bn_ib.weight', 'bns.1.bn_ib.bias', 'bns.1.bn_ob.weight', 'bns.1.bn_ob.bias')

def prefix(model, condition):
    dropped = F.dropout(condition.features, p=model.dropout, training=model.training)
    h = model.convs[0](dropped, condition.matrix, None)
    h = model.bns[0](h, condition.core_size).relu()
    h = F.dropout(h, p=model.dropout, training=model.training)
    return model.convs[-1].propagate(condition.matrix, x=h, edge_weight=None, size=None)

def head(model, boundary):
    conv = model.convs[-1]
    return F.linear(boundary, conv.lin.weight, conv.bias)

class FrozenMeanP2:

    def __init__(self, model, optimizer):
        (self.model, self.optimizer) = (model, optimizer)
        self.parameters = dict(model.named_parameters())
        if set(self.parameters) != set(BODY_NAMES + HEAD_NAMES + UNUSED_NAMES) or len(model.state_dict()) != 24 or len(model.convs) != 2 or (model.num_nodes != 0) or (not model.drop_input) or (model.dropout != 0.3):
            raise ValueError('Require original Flickr current-conditional L2/DoubleBN/drop_input.3/actual24')
        if any((g['weight_decay'] != 0 or g['lr'] != 0.01 for g in optimizer.param_groups)):
            raise ValueError('Original .01 zero-decay Adam is frozen')
        self.blocks = self.updates = 0
        self.receiver_labels = self.core_labels = 0
        self.cache_alive = False
        self.cache_peak_payload_bytes = 0

    def clocks(self):
        result = {name: int(self.optimizer.state.get(p, {}).get('step', 0)) for (name, p) in self.parameters.items()}
        for (name, steps) in result.items():
            expected = 0 if name in UNUSED_NAMES else self.updates
            if steps != expected or (name in UNUSED_NAMES and self.optimizer.state.get(self.parameters[name], {})):
                raise RuntimeError('Actual Adam support/clock differs: ' + name)
        for branch in ('bn_ib', 'bn_ob'):
            if int(getattr(self.model.bns[0], branch).num_batches_tracked) != self.blocks:
                raise RuntimeError('First DoubleBN population clock differs')
            if int(getattr(self.model.bns[1], branch).num_batches_tracked):
                raise RuntimeError('Unused final DoubleBN changed')
        if self.cache_alive or any((p.grad is not None for p in self.parameters.values())):
            raise RuntimeError('Completed block retained cache or submitted gradients')
        return result
