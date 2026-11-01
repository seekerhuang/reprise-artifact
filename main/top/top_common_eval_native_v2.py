import torch
from main.top.top_native_graph_ops_v2 import spmm as zero_spmm

@torch.no_grad()
def evaluate(engine):
    model = engine.model
    assert not model.training
    (first, middle, last) = model.convs
    z = engine.first_image @ first.weight + first.bias
    h = model.bns[0](z)
    del z
    h.relu_()
    aggregate = zero_spmm(engine.matrix, h)
    z = aggregate @ middle.weight + middle.bias
    del aggregate
    normal = model.bns[1](z)
    del z
    normal.add_(h)
    del h
    normal.relu_()
    image = normal @ last.weight
    del normal
    return zero_spmm(engine.matrix, image) + last.bias
