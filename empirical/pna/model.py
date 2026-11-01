from empirical.pna.base import ScalableGNN
from itertools import product
from typing import Optional, List
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from torch_scatter import segment_csr
EPS = 1e-05

class PNAConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, aggregators: List[str], scalers: List[str], deg: Tensor, **kwargs):
        super().__init__(aggr=None, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        deg = deg.to(torch.float)
        self.avg_deg = {'lin': deg.mean().item(), 'log': (deg + 1).log().mean().item()}
        self.pre_lins = torch.nn.ModuleList([Linear(in_channels, out_channels) for _ in range(len(aggregators) * len(scalers))])
        self.post_lins = torch.nn.ModuleList([Linear(out_channels, out_channels) for _ in range(len(aggregators) * len(scalers))])
        self.lin = Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.pre_lins:
            lin.reset_parameters()
        for lin in self.post_lins:
            lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, adj_t):
        out = self.propagate(adj_t, x=x)
        out += self.lin(x)[:out.size(0)]
        return out

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        deg = adj_t.storage.rowcount().to(x.dtype).view(-1, 1)
        out = 0
        for ((aggr, scaler), pre_lin, post_lin) in zip(product(self.aggregators, self.scalers), self.pre_lins, self.post_lins):
            h = pre_lin(x).relu_()
            h = segment_csr(h[adj_t.storage.col()], adj_t.storage.rowptr(), reduce=aggr)
            h = post_lin(h)
            if scaler == 'amplification':
                h *= (deg + 1).log() / self.avg_deg['log']
            elif scaler == 'attenuation':
                h *= self.avg_deg['log'] / ((deg + 1).log() + EPS)
            out += h
        return out

    def propagation(self, in_batch_x, P_tilde):
        to_outbatch_emb = self.message_and_aggregate(P_tilde, in_batch_x)
        return to_outbatch_emb

    def transformation(self, out_batch_x: Tensor, emb):
        out = emb
        out += self.lin(out_batch_x)
        return out

class PNA(ScalableGNN):

    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, aggregators: List[int], scalers: List[int], deg: Tensor, dropout: float=0.0, drop_input: bool=True, batch_norm: bool=True, residual: bool=False, pool_size: Optional[int]=None, buffer_size: Optional[int]=None, device=None, gamma: float=0.0):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size, buffer_size, device, gamma)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.residual = residual
        self.batch_norm = batch_norm
        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            conv = PNAConv(in_dim, out_dim, aggregators=aggregators, scalers=scalers, deg=deg)
            self.convs.append(conv)
        self.bns = ModuleList()
        for i in range(num_layers - 1):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.convs[-1:]

    def reset_parameters(self):
        super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if len(args) > 0:
            batch_size = args[0]
            n_id = args[1]
            P_bar = adj_t[:, batch_size:]
            P_tilde = P_bar.t()
        else:
            print('no batch_size in the forward of conv')
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for (conv, bn, hist) in zip(self.convs[:-1], self.bns, self.histories):
            h = conv(x, adj_t)
            if self.batch_norm:
                h = bn(h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()
            if n_id is not None and batch_size is not None:
                with torch.no_grad():
                    if self.batch_norm:
                        bn.eval()
                        to_outbatch_emb = conv.propagation(x[:batch_size].detach(), P_tilde)
                        outbatch_emb = conv.transformation(x[batch_size:].detach(), emb=to_outbatch_emb)
                        outbatch_emb = bn(outbatch_emb).relu()
                        hist.forward(outbatch_emb, n_id[batch_size:])
                        bn.train()
                    else:
                        to_outbatch_emb = conv.propagation(x[:batch_size].detach(), P_tilde)
                        outbatch_emb = conv.transformation(x[batch_size:].detach(), emb=to_outbatch_emb).relu()
                        hist.forward(outbatch_emb, n_id[batch_size:])
            x = self.push_and_pull(hist, h, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0 and self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[layer](x, adj_t)
        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
