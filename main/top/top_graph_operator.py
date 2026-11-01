import torch
from main.top.top_native_graph_ops_v2 import spmm as native_graph_spmm

class ExactPropagation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, matrix, transpose):
        ctx.transpose = transpose
        return native_graph_spmm(matrix, hidden)

    @staticmethod
    def backward(ctx, cotangent):
        return (native_graph_spmm(ctx.transpose, cotangent.contiguous()), None, None)
