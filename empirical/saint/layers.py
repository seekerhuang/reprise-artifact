import torch
from torch import nn
import scipy.sparse as sp
F_ACT = {'relu': nn.ReLU(), 'I': lambda x: x}

class HighOrderAggregator(nn.Module):

    def __init__(self, dim_in, dim_out, dropout=0.0, act='relu', order=1, aggr='mean', bias='norm-nn', **kwargs):
        super(HighOrderAggregator, self).__init__()
        assert bias in ['bias', 'norm', 'norm-nn']
        (self.order, self.aggr) = (order, aggr)
        (self.act, self.bias) = (F_ACT[act], bias)
        self.dropout = dropout
        (self.f_lin, self.f_bias) = ([], [])
        (self.offset, self.scale) = ([], [])
        self.num_param = 0
        for o in range(self.order + 1):
            self.f_lin.append(nn.Linear(dim_in, dim_out, bias=False))
            nn.init.xavier_uniform_(self.f_lin[-1].weight)
            self.f_bias.append(nn.Parameter(torch.zeros(dim_out)))
            self.num_param += dim_in * dim_out
            self.num_param += dim_out
            self.offset.append(nn.Parameter(torch.zeros(dim_out)))
            self.scale.append(nn.Parameter(torch.ones(dim_out)))
            if self.bias == 'norm' or self.bias == 'norm-nn':
                self.num_param += 2 * dim_out
        self.f_lin = nn.ModuleList(self.f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params = nn.ParameterList(self.f_bias + self.offset + self.scale)
        self.f_bias = self.params[:self.order + 1]
        if self.bias == 'norm':
            self.offset = self.params[self.order + 1:2 * self.order + 2]
            self.scale = self.params[2 * self.order + 2:]
        elif self.bias == 'norm-nn':
            final_dim_out = dim_out * ((aggr == 'concat') * (order + 1) + (aggr == 'mean'))
            self.f_norm = nn.BatchNorm1d(final_dim_out, eps=1e-09, track_running_stats=True)
        self.num_param = int(self.num_param)

    def _spmm(self, adj_norm, _feat):
        return torch.sparse.mm(adj_norm, _feat)

    def _f_feat_trans(self, _feat, _id):
        feat = self.act(self.f_lin[_id](_feat) + self.f_bias[_id])
        if self.bias == 'norm':
            mean = feat.mean(dim=1).view(feat.shape[0], 1)
            var = feat.var(dim=1, unbiased=False).view(feat.shape[0], 1) + 1e-09
            feat_out = (feat - mean) * self.scale[_id] * torch.rsqrt(var) + self.offset[_id]
        else:
            feat_out = feat
        return feat_out

    def forward(self, inputs):
        (adj_norm, feat_in) = inputs
        feat_in = self.f_dropout(feat_in)
        feat_hop = [feat_in]
        for o in range(self.order):
            feat_hop.append(self._spmm(adj_norm, feat_hop[-1]))
        feat_partial = [self._f_feat_trans(ft, idf) for (idf, ft) in enumerate(feat_hop)]
        if self.aggr == 'mean':
            feat_out = feat_partial[0]
            for o in range(len(feat_partial) - 1):
                feat_out += feat_partial[o + 1]
        elif self.aggr == 'concat':
            feat_out = torch.cat(feat_partial, 1)
        else:
            raise NotImplementedError
        if self.bias == 'norm-nn':
            feat_out = self.f_norm(feat_out)
        return (adj_norm, feat_out)
