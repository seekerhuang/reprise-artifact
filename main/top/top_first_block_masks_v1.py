import torch
import torch.nn.functional as F

class FirstBlock(torch.autograd.Function):

    @staticmethod
    def forward(ctx, image, weight, bias, gamma, beta, bn, probability, work):
        assert bn.training and bn.track_running_stats and (bn.momentum is not None)
        ctx.running_mean = bn.running_mean.detach().clone()
        ctx.running_var = bn.running_var.detach().clone()
        (ctx.eps, ctx.momentum, ctx.probability, ctx.work) = (bn.eps, bn.momentum, probability, work)
        z = image @ weight + bias
        activated = bn(z)
        del z
        relu_mask = ~(activated <= 0)
        activated.relu_()
        (output, dropout_mask) = torch.ops.aten.native_dropout(activated, probability, True)
        ctx.save_for_backward(image, weight, bias, gamma, beta, relu_mask, dropout_mask)
        return output

    @staticmethod
    def backward(ctx, upstream):
        (image, weight, bias, gamma, beta, relu_mask, dropout_mask) = ctx.saved_tensors
        gradient = upstream.contiguous()
        torch.ops.aten.native_dropout_backward.out(gradient, dropout_mask, 1 / (1 - ctx.probability), out=gradient)
        gradient.masked_fill_(~relu_mask, 0)
        with torch.enable_grad():
            local = tuple((v.detach().requires_grad_(True) for v in (weight, bias, gamma, beta)))
            (w, b, g, offset) = local
            z = image @ w + b
            norm = F.batch_norm(z, ctx.running_mean, ctx.running_var, g, offset, training=True, momentum=ctx.momentum, eps=ctx.eps)
            edge = torch.autograd.graph.get_gradient_edge(norm)
            del norm, z
            gradients = torch.autograd.grad(edge, local, gradient, create_graph=False)
        ctx.work['activation_recomputed_first_blocks'] = ctx.work.get('activation_recomputed_first_blocks', 0) + 1
        return (None, *gradients, None, None, None)

def first_block(image, layer, bn, probability, work):
    return FirstBlock.apply(image, layer.weight, layer.bias, bn.weight, bn.bias, bn, probability, work)
