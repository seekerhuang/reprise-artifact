import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

def map_fresh_state(common):
    return {k.replace('.lin_l.weight', '.fc_neigh.weight').replace('.lin_l.bias', '.fc_self.bias').replace('.lin_r.weight', '.fc_self.weight'): v for (k, v) in common.items()}

def fresh_state(native):
    return {k.replace('.fc_neigh.weight', '.lin_l.weight').replace('.fc_self.bias', '.lin_l.bias').replace('.fc_self.weight', '.lin_r.weight'): v for (k, v) in native.items()}

class LayerView:

    def __init__(self, model, index):
        (self.model, self.index) = (model, index)

    def __call__(self, block, x, edge_weight=None):
        h = self.model.convs[self.index](block, x, edge_weight=edge_weight)
        if self.index < self.model.n_layers - 1:
            return self.model.bns[self.index](h)
        return F.log_softmax(h, dim=-1)

def model_class(native_sage, context):

    class CommonSAGE(nn.Module):

        def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
            super().__init__()
            if n_layers != 3:
                raise ValueError('Common Fresh model has3 layers')
            (self.n_layers, self.n_hidden, self.n_classes) = (n_layers, n_hidden, n_classes)
            widths = [in_feats, n_hidden, n_hidden, n_classes]
            self.convs = nn.ModuleList([SAGEConv(widths[i], widths[i + 1], 'mean') for i in range(n_layers)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers - 1)])
            (self.activation, self.dropout) = (activation, nn.Dropout(dropout))
            self._layer_views = tuple((LayerView(self, i) for i in range(n_layers)))

        @property
        def layers(self):
            return self._layer_views
        forward = native_sage.forward
        inference = native_sage.inference
    return CommonSAGE
