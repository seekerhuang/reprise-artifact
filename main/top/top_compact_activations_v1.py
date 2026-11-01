import torch
import torch.nn.functional as F

class CompactReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, value, inplace):
        output = torch.relu_(value) if inplace else torch.relu(value)
        if inplace:
            ctx.mark_dirty(value)
        ctx.save_for_backward(~(output <= 0))
        return output

    @staticmethod
    def backward(ctx, upstream):
        (gate,) = ctx.saved_tensors
        return (upstream.masked_fill(~gate, 0), None)

def compact_relu(value, inplace=False):
    if not torch.is_grad_enabled():
        return torch.relu_(value) if inplace else torch.relu(value)
    return CompactReLU.apply(value, inplace)

class DenseBN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, image, weight, bias, gamma, beta, bn, work):
        assert bn.training and bn.track_running_stats and (bn.momentum is not None)
        ctx.save_for_backward(image, weight, bias, gamma, beta)
        ctx.running_mean = bn.running_mean.detach().clone()
        ctx.running_var = bn.running_var.detach().clone()
        (ctx.eps, ctx.momentum, ctx.work) = (bn.eps, bn.momentum, work)
        return bn(image @ weight + bias)

    @staticmethod
    def backward(ctx, upstream):
        (image, weight, bias, gamma, beta) = ctx.saved_tensors
        with torch.enable_grad():
            local = tuple((x.detach().requires_grad_(True) for x in (image, weight, bias, gamma, beta)))
            (x, w, b, g, offset) = local
            z = x @ w + b
            norm = F.batch_norm(z, ctx.running_mean, ctx.running_var, g, offset, training=True, momentum=ctx.momentum, eps=ctx.eps)
            edge = torch.autograd.graph.get_gradient_edge(norm)
            del norm, z
            gradients = torch.autograd.grad(edge, local, upstream, create_graph=False)
        ctx.work['activation_recomputed_middle_dense_blocks'] = ctx.work.get('activation_recomputed_middle_dense_blocks', 0) + 1
        return (*gradients, None, None)

def dense_bn(image, layer, bn, work):
    return DenseBN.apply(image, layer.weight, layer.bias, bn.weight, bn.bias, bn, work)
