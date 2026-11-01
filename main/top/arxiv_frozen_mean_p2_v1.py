import torch
import torch.nn.functional as F
from main.top.top_graph_operator import ExactPropagation
from main.top.top_parameter_channel_cv_v1 import FullGraphExecution
from main.top.top_first_block_masks_v1 import first_block as memory_first_block
from main.top.top_compact_activations_v1 import compact_relu, dense_bn

class FrozenMeanP2(FullGraphExecution):

    def __init__(self, model, matrix, features, *, training_ids):
        super().__init__(model, matrix, features, method='direct')
        self.training_ids = training_ids
        self.blocks = 0
        self.work = dict(prefix_forwards=0, prefix_VJPs=0, loss_queries=0, head_Adam=0, body_Adam=0, optimizer_API_calls=0, supervised_node_presentations=0)
        self.cache_alive = False
        self.peak_named_boundary_bytes = 0

    def prefix(self):
        (first, middle, _) = self.model.convs
        h = memory_first_block(self.first_image, first, self.model.bns[0], self.model.dropout, self.work)
        aggregate = ExactPropagation.apply(h, self.matrix, self.transpose)
        z = dense_bn(aggregate, middle, self.model.bns[1], self.work)
        del aggregate
        z.add_(h)
        del h
        return F.dropout(compact_relu(z, inplace=True), p=self.model.dropout, training=self.model.training)

    def head(self, h):
        last = self.model.convs[-1]
        from main.top.top_deferred_head_v1 import deferred_mm
        return ExactPropagation.apply(deferred_mm(h, last.weight, self._head_factors), self.matrix, self.transpose) + last.bias
