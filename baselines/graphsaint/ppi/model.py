import torch
from torch import nn
import torch.nn.functional as F

def model_class(native_saint_class, args_global, *, hidden_channels=1024):
    from torch_geometric.nn import GCNConv

    class CommonPPI(nn.Module):

        def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
            super().__init__()
            if arch_gcn['loss'] != 'sigmoid':
                raise ValueError('PPI multilabel original loss required')
            self.use_cuda = args_global.gpu >= 0 and (not cpu_eval)
            self.sigmoid_loss = True
            self.dropout = float(train_params['dropout'])
            self.lr = float(train_params['lr'])
            self.weight_decay = float(train_params['weight_decay'])
            if self.weight_decay != 0.0:
                raise ValueError('Published PPI2 recipe has weight_decay0')
            self.num_classes = num_classes
            self.lins = nn.ModuleList([nn.Linear(feat_full.shape[1], hidden_channels), nn.Linear(hidden_channels, num_classes)])
            self.convs = nn.ModuleList([GCNConv(hidden_channels, hidden_channels, normalize=False) for _ in range(2)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(2)])
            self.feat_full = torch.from_numpy(feat_full.astype('float32'))
            self.label_full = torch.from_numpy(label_full.astype('float32'))
            if self.use_cuda:
                self.feat_full = self.feat_full.cuda()
                self.label_full = self.label_full.cuda()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def forward(self, node_subgraph, adj_subgraph):
            x = self.feat_full[node_subgraph]
            labels = self.label_full[node_subgraph]
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            for (conv, bn) in zip(self.convs, self.bns):
                h = bn(conv(x, adj_subgraph))
                h += x
                x = F.dropout(h.relu_(), p=self.dropout, training=self.training)
            return (self.lins[1](x), labels, labels)
        _loss = native_saint_class._loss
        predict = native_saint_class.predict
        train_step = native_saint_class.train_step
        eval_step = native_saint_class.eval_step
    return CommonPPI
