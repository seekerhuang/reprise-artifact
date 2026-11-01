import torch

class ConsumedRoot(torch.autograd.Function):

    @staticmethod
    def forward(ctx, value, covector):
        assert not covector.requires_grad
        ctx.covector = covector
        return torch.dot(value.reshape(-1), covector.reshape(-1))

    @staticmethod
    def backward(ctx, scale):
        covector = ctx.covector
        del ctx.covector
        covector.mul_(scale)
        return (covector, None)

def root(value, covector):
    return ConsumedRoot.apply(value, covector)
