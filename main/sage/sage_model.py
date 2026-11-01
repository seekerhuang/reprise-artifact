import torch
import torch.nn.functional as F

class MySageConvHistory(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, need_grad, root_weight: bool=True, bias: bool=True):
        super(MySageConvHistory, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.root_weight = root_weight
        self.lin_l = torch.nn.Linear(in_channels, hidden_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.reset_parameters()
        self.need_grad = need_grad

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

class HistorySAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_type, config, table, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.graph_type = graph_type
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(self.num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.table = table
        self.init_convs(**kwargs)

    def init_convs(self, **kwargs):
        self.convs = torch.nn.ModuleList()
        self.convs.append(self.init_history_conv(self.in_channels, self.hidden_channels, need_grad=self.table.method == 'grad', **kwargs))
        for _ in range(self.num_layers - 2):
            self.convs.append(self.init_history_conv(self.hidden_channels, self.hidden_channels, need_grad=self.table.method == 'grad', **kwargs))
        if self.num_layers > 1:
            self.last_layer_history = False
            self.convs.append(self.init_history_conv(self.hidden_channels, self.out_channels, need_grad=self.table.method == 'grad', **kwargs))

    def init_history_conv(self, in_channels, out_channels, **kwargs):
        assert 'need_grad' in kwargs
        return MySageConvHistory(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
