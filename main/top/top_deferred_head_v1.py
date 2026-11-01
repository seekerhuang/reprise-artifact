import torch

class DeferredLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, factors):
        ctx.save_for_backward(hidden, weight)
        ctx.factors = factors
        return hidden @ weight

    @staticmethod
    def backward(ctx, upstream):
        (hidden, weight) = ctx.saved_tensors
        weight_gradient = hidden.T @ upstream
        ctx.factors.append((upstream.detach(), weight.detach().clone()))
        return (None, weight_gradient, None)

def deferred_mm(hidden, weight, factors):
    return DeferredLinear.apply(hidden, weight, factors)

@torch.no_grad()
def consume(factors, dead_boundary):
    assert len(factors) == 2 and dead_boundary.is_contiguous()
    (first_gradient, first_weight) = factors[0]
    (second_gradient, second_weight) = factors[1]
    result = first_gradient @ first_weight.T
    second = dead_boundary.detach()
    torch.mm(second_gradient, second_weight.T, out=second)
    result.add_(second)
    return result
