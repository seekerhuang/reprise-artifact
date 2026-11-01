import torch
import torch.nn.functional as F
from main.iglu import iglu_model as base
from main.iglu.tf_adam import TF1Adam
SEMANTICS = ('Q', 'R', 'w', 'b1', 'b2', 'b3', 'L2_O', 'L2_S', 'L1_O', 'L1_S')

def prefix(model, geometry, condition, dropout=0.0):
    hidden = model.norm_relu(geometry.first @ model.q + model.b1, 1)
    hidden = F.dropout(hidden, p=dropout, training=model.training)
    hidden = base.ExactPropagation.apply(hidden, geometry.matrix, geometry.transpose)
    hidden = model.norm_relu(hidden @ model.r + model.b2, 2)
    hidden = F.dropout(hidden, p=dropout, training=model.training)
    return base.ExactPropagation.apply(hidden, condition.matrix, condition.transpose)

class FrozenMeanP2:

    def __init__(self, model, geometry, dropout=0.0):
        (self.model, self.geometry) = (model, geometry)
        assert 0 <= dropout < 1
        self.dropout = dropout
        self.optimizers = {name: TF1Adam(params, lr=0.001) for (name, params) in model.groups().items()}
        self.actual_updates = 0
        self.work = dict(prefix_forwards=0, prefix_VJPs=0, head_queries=0, optimizer_API_calls=0, supervised_presentations=0)

    def clocks(self):
        result = {}
        for (name, opt) in self.optimizers.items():
            for group in opt.param_groups:
                assert group['tf_step'] == self.actual_updates
                assert all((opt.state[p]['step'] == self.actual_updates for p in group['params']))
            result[name] = self.actual_updates
        return result
