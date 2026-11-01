import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from empirical.saint.layers import HighOrderAggregator
from empirical.saint.data import parse_layer_yml
from types import SimpleNamespace
layers = SimpleNamespace(HighOrderAggregator=HighOrderAggregator)
args_global = SimpleNamespace(gpu=-1)

class GraphSAINT(nn.Module):

    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        super(GraphSAINT, self).__init__()
        self.use_cuda = args_global.gpu >= 0
        if cpu_eval:
            self.use_cuda = False
        if 'attention' in arch_gcn:
            if 'gated_attention' in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls = layers.GatedAttentionAggregator
                    self.mulhead = int(arch_gcn['attention'])
            else:
                self.aggregator_cls = layers.AttentionAggregator
                self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls = layers.HighOrderAggregator
            self.mulhead = 1
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        self.sigmoid_loss = arch_gcn['loss'] == 'sigmoid'
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
            self.label_full = self.label_full.cuda()
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(label_full.argmax(axis=1).astype(np.int64))
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()
        self.num_classes = num_classes
        (_dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer) = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        self.set_idx_conv()
        self.set_dims(_dims)
        self.loss = 0
        self.opt_op = None
        self.num_params = 0
        (self.aggregators, num_param) = self.get_aggregators()
        self.num_params += num_param
        self.conv_layers = nn.Sequential(*self.aggregators)
        self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.num_classes, act='I', order=0, dropout=self.dropout, bias='bias')
        self.num_params += self.classifier.num_param
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_dims(self, dims):
        self.dims_feat = [dims[0]] + [((self.aggr_layer[l] == 'concat') * self.order_layer[l] + 1) * dims[l + 1] for l in range(len(dims) - 1)]
        self.dims_weight = [(self.dims_feat[l], dims[l + 1]) for l in range(len(dims) - 1)]

    def set_idx_conv(self):
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])

    def forward(self, node_subgraph, adj_subgraph):
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        (_, emb_subg) = self.conv_layers((adj_subgraph, feat_subg))
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        pred_subg = self.classifier((None, emb_subg_norm))[1]
        return (pred_subg, label_subg, label_subg_converted)

    def _loss(self, preds, labels, norm_loss):
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss, reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss * _ls).sum()

    def get_aggregators(self):
        num_param = 0
        aggregators = []
        for l in range(self.num_layers):
            aggr = self.aggregator_cls(*self.dims_weight[l], dropout=self.dropout, act=self.act_layer[l], order=self.order_layer[l], aggr=self.aggr_layer[l], bias=self.bias_layer[l], mulhead=self.mulhead)
            num_param += aggr.num_param
            aggregators.append(aggr)
        return (aggregators, num_param)

    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        self.eval()
        with torch.no_grad():
            (preds, labels, labels_converted) = self(node_subgraph, adj_subgraph)
            loss = self._loss(preds, labels_converted, norm_loss_subgraph)
        return (loss, self.predict(preds), labels)
