import torch
from torch import nn
import dgl
import dgl.function as fn

class CommonConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.linear_before_mp = in_channels > out_channels

    def forward(self, block, x, degree, edge_weight=None):
        with block.local_scope():
            src = block.srcdata[dgl.NID].long()
            dst = block.dstdata[dgl.NID].long()
            (src_degree, dst_degree) = (degree[src], degree[dst])
            value = self.lin(x) if self.linear_before_mp else x
            block.srcdata['h'] = value * torch.rsqrt(src_degree).unsqueeze(1)
            message = fn.copy_u('h', 'm')
            if edge_weight is not None:
                if edge_weight.shape[0] != block.num_edges():
                    raise RuntimeError('Original LABOR edge-weight length differs')
                block.edata['_edge_weight'] = edge_weight
                message = fn.u_mul_e('h', '_edge_weight', 'm')
            block.update_all(message, fn.mean('m', 'neigh'))
            neighbors = block.dstdata['neigh'] * (dst_degree * torch.rsqrt(dst_degree)).unsqueeze(1)
            total = neighbors
            if not self.linear_before_mp:
                total = self.lin(total)
            return total + self.bias

class LayerView:

    def __init__(self, model, index):
        (self.model, self.index) = (model, index)

    def __call__(self, block, x, edge_weight=None):
        return self.model.convs[self.index](block, x, self.model._full_degree, edge_weight)

def model_class(native_sage, context):

    class CommonYelp(nn.Module):

        def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
            super().__init__()
            if n_layers != 2:
                raise ValueError('Common Yelp model requires two graph layers')
            if context.get('degree') is None:
                raise RuntimeError('Bind full-graph degree before construction')
            (self.n_layers, self.n_hidden, self.n_classes) = (n_layers, n_hidden, n_classes)
            self.convs = nn.ModuleList([CommonConv(in_feats, n_hidden), CommonConv(n_hidden, n_classes)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(2)])
            (self.dropout, self.activation) = (nn.Dropout(dropout), activation)
            self.register_buffer('_full_degree', context['degree'].float(), persistent=False)
            self._layer_views = tuple((LayerView(self, i) for i in range(n_layers)))

        @property
        def layers(self):
            return self._layer_views
        forward = native_sage.forward
        inference = native_sage.inference
    return CommonYelp
