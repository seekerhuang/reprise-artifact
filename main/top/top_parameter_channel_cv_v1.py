import torch
import torch.nn.functional as F
from main.top.top_graph_operator import ExactPropagation

class FullGraphExecution:

    def __init__(self, model, matrix, features, *, method='direct', channels=32, generator=None, fixed_indices=None):
        if len(model.convs) != 3 or not model.batch_norm or (not model.residual) or model.linear or model.drop_input:
            raise ValueError('Requires the declared native TOP residual/BN/no-input-dropout backbone')
        if method not in ('direct', 'gradient_cache', 'current_tail', 'parameter_cv', 'channel_refresh'):
            raise ValueError(method)
        from main.top.top_native_graph_ops_v2 import make_graph, spmm as native_graph_spmm
        (self.model, self.features) = (model, features)
        self.matrix = make_graph(matrix)
        self.transpose = self.matrix.reverse(copy_edata=True)
        with torch.no_grad():
            self.first_image = native_graph_spmm(self.matrix, features)
        self.controller = None

    def forward(self):
        (model, c) = (self.model, self.controller)
        if not model.training:
            from main.top.top_common_eval_native_v2 import evaluate as memory_evaluate
            return memory_evaluate(self)
        (first, middle, last) = model.convs
        cheap = model.training and c is not None and (not c.paid)
        if cheap:
            with torch.no_grad():
                z = self.first_image @ first.weight + first.bias
                h = F.dropout(F.relu(model.bns[0](z)), p=model.dropout, training=True)
                if c.method in ('parameter_cv', 'channel_refresh'):
                    c.current_gate = h.index_select(1, c.indices).gt(0).to(h.dtype) / (1 - model.dropout)
        else:
            z = self.first_image @ first.weight + first.bias
            h = F.dropout(F.relu(model.bns[0](z)), p=model.dropout, training=model.training)
        if not model.training:
            del z
        aggregate = ExactPropagation.apply(h, self.matrix, self.transpose)
        z_middle = aggregate @ middle.weight + middle.bias
        if not model.training:
            del aggregate
        norm_middle = model.bns[1](z_middle)
        if not model.training:
            del z_middle
        if cheap and c.method in ('parameter_cv', 'channel_refresh'):
            z_middle.register_hook(c.capture_delta)
            norm_middle.register_hook(c.capture_residual)
        h = F.dropout(F.relu(norm_middle + h), p=model.dropout, training=model.training)
        if not model.training:
            del norm_middle
        return ExactPropagation.apply(h @ last.weight, self.matrix, self.transpose) + last.bias
