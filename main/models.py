import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, inputs, hidden, outputs, layers, dropout, *, residual=False,
                 batch_norm=True, drop_input=False, linear=False):
        super().__init__()
        self.num_layers = layers
        self.num_nodes = 0
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.drop_input = drop_input
        self.linear = linear
        self.lins = nn.ModuleList()
        if linear:
            self.lins.extend([nn.Linear(inputs, hidden), nn.Linear(hidden, outputs)])
        self.convs = nn.ModuleList([
            GCNConv(inputs if i == 0 and not linear else hidden,
                    outputs if i == layers - 1 and not linear else hidden,
                    normalize=False)
            for i in range(layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

    @property
    def reg_modules(self):
        return nn.ModuleList(list(self.convs if self.linear else self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins if self.linear else self.convs[-1:]

    def reset_parameters(self):
        for module in [*self.lins, *self.convs, *self.bns]:
            module.reset_parameters()


class TOPGCN(nn.Module):
    def __init__(self, inputs, hidden, outputs, dropout):
        super().__init__()
        from dgl.nn import GraphConv
        self.dropout = dropout
        self.batch_norm = True
        self.residual = True
        self.linear = self.drop_input = False
        self.convs = nn.ModuleList([
            GraphConv(inputs, hidden, norm='none', allow_zero_in_degree=True),
            GraphConv(hidden, hidden, norm='none', allow_zero_in_degree=True),
            GraphConv(hidden, outputs, norm='none', allow_zero_in_degree=True),
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(3)])

    def reset_parameters(self):
        for module in [*self.convs, *self.bns]:
            module.reset_parameters()


class DoubleBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn_ib = nn.BatchNorm1d(channels)
        self.bn_ob = nn.BatchNorm1d(channels)

    def forward(self, x, batch_size=None):
        if batch_size is None:
            return self.bn_ib(x)
        return torch.cat((self.bn_ib(x[:batch_size]), self.bn_ob(x[batch_size:])), dim=0)

    def reset_parameters(self):
        self.bn_ib.reset_parameters()
        self.bn_ob.reset_parameters()


class DoubleBNGCN(GCN):
    def __init__(self, inputs, outputs, layers, dropout, drop_input):
        super().__init__(inputs, 256, outputs, layers, dropout, drop_input=drop_input)
        self.bns = nn.ModuleList([DoubleBN(256) for _ in range(layers)])

