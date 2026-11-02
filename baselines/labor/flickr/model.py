import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class CommonConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, block, x, degree, edge_weight=None):
        with block.local_scope():
            src = block.srcdata[dgl.NID].long()
            dst = block.dstdata[dgl.NID].long()
            src_degree = degree[src]
            dst_degree = degree[dst]
            transformed = self.lin(x)
            block.srcdata['h'] = transformed * torch.rsqrt(src_degree + 1).unsqueeze(1)
            message = fn.copy_u('h', 'm')
            if edge_weight is not None:
                if edge_weight.shape[0] != block.num_edges():
                    raise RuntimeError('Original LABOR edge-weight length differs')
                block.edata['_edge_weight'] = edge_weight
                message = fn.u_mul_e('h', '_edge_weight', 'm')
            block.update_all(message, fn.mean('m', 'neigh'))
            neighbors = block.dstdata['neigh'] * (dst_degree * torch.rsqrt(dst_degree + 1)).unsqueeze(1)
            own = transformed[:block.num_dst_nodes()] / (dst_degree + 1).unsqueeze(1)
            return neighbors + own + self.bias

class LayerView:

    def __init__(self, model, index):
        self.model = model
        self.index = index

    def __call__(self, block, x, edge_weight=None):
        m = self.model
        if self.index == 0:
            x = F.dropout(x, p=m.dropout.p, training=m.training)
        value = m.convs[self.index](block, x, m._full_degree, edge_weight=edge_weight)
        return m.bns[0](value) if self.index == 0 else value

def model_class(native_sage, context):

    class CommonFlickr(nn.Module):

        def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
            super().__init__()
            if n_layers != 2:
                raise ValueError('Shared Flickr model has two graph layers')
            if context.get('degree') is None:
                raise RuntimeError('Bind verified full-graph degree before model construction')
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.n_classes = n_classes
            self.convs = nn.ModuleList([CommonConv(in_feats, n_hidden), CommonConv(n_hidden, n_classes)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(2)])
            self.dropout = nn.Dropout(dropout)
            self.activation = activation
            self.register_buffer('_full_degree', context['degree'].float(), persistent=False)
            self._layer_views = tuple((LayerView(self, i) for i in range(n_layers)))

        @property
        def layers(self):
            return self._layer_views
        forward = native_sage.forward
        inference = native_sage.inference
    return CommonFlickr
