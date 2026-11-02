import torch
import torch.nn.functional as F

from main.top.top_native_graph_ops_v2 import make_graph, spmm
from main.top.top_first_block_masks_v1 import first_block
from main.top.top_compact_activations_v1 import compact_relu, dense_bn


class Direct:
    def __init__(self, model, matrix, features, *, training_ids):
        self.model, self.graph = model, make_graph(matrix)
        self.training_ids = training_ids
        self.work = {}
        with torch.no_grad():
            self.first_image = spmm(self.graph, features)

    def forward(self):
        model, graph = self.model, self.graph
        first, middle, last = model.convs
        if model.training:
            x = first_block(self.first_image, first, model.bns[0], model.dropout, self.work)
            aggregate = spmm(graph, x)
            h = dense_bn(aggregate, middle, model.bns[1], self.work)
            del aggregate
        else:
            h = self.first_image @ first.weight + first.bias
            h = model.bns[0](h)
            x = F.dropout(h.relu_(), p=model.dropout, training=False)
            del h
            h = model.bns[1](spmm(graph, x) @ middle.weight + middle.bias)
        h += x
        del x
        x = F.dropout(compact_relu(h, inplace=True), p=model.dropout, training=model.training)
        del h
        return spmm(graph, x @ last.weight) + last.bias

    def update_block(self, optimizer, labels, method=None):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.forward()[self.training_ids], labels)
        loss.backward()
        optimizer.step()

