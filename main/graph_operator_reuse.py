import torch

class ExactPropagation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, matrix, transpose):
        ctx.transpose = transpose
        return torch.sparse.mm(matrix, hidden)

    @staticmethod
    def backward(ctx, cotangent):
        return (torch.sparse.mm(ctx.transpose, cotangent.contiguous()), None, None)
