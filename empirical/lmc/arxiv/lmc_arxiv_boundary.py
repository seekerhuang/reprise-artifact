import torch
import torch.nn.functional as F
from empirical.lmc.arxiv import lmc_arxiv_core as core
from main.graph_operator_reuse import ExactPropagation
core.ExactPropagation = ExactPropagation

def prefix(model, condition):
    h = F.linear(condition.first_image, model.convs[0].lin.weight, model.convs[0].bias)
    h = model.bns[0](h, condition.core_size).relu()
    h = F.dropout(h, p=model.dropout, training=model.training)
    h = core.ExactPropagation.apply(h, condition.matrix, condition.transpose)
    h = F.linear(h, model.convs[1].lin.weight, model.convs[1].bias)
    h = model.bns[1](h, condition.core_size).relu()
    h = F.dropout(h, p=model.dropout, training=model.training)
    return core.ExactPropagation.apply(h, condition.label_matrix, condition.label_transpose)

def head(model, boundary):
    last = model.convs[-1]
    return F.linear(boundary, last.lin.weight, last.bias)

class FrozenMeanP2:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.actual_updates = 0
        self.cache_alive = False
        self.work = dict(prefix_forwards=0, prefix_VJPs=0, loss_queries=0, head_updates=0, body_updates=0, supervised_node_presentations=0)

    def verify_clocks(self):
        clocks = {name: int(self.optimizer.state[p]['step']) if p in self.optimizer.state else 0 for (name, p) in self.model.named_parameters()}
        for (name, steps) in clocks.items():
            expected = 0 if name.startswith('bns.2.') else self.actual_updates
            if steps != expected:
                raise AssertionError((name, steps, expected))
        if self.cache_alive:
            raise AssertionError('Retained a completed block')
        return clocks
