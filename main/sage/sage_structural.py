import torch
import torch.nn.functional as F
from main.sage.finite_first_mean_p2 import FiniteFirstMeanP2

class SampledSAGEStructural(FiniteFirstMeanP2):

    def __init__(self, *args, observe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe = observe
        self.measurements = []
        self.work.update(structural_measurements=0, structural_installs=0, extra_prefix_forwards=0, extra_prefix_VJPs=0, extra_graph_multiplications=0, optimizer_API_calls=0)

    @torch.no_grad()
    def geometry(self):
        bn = self.model.bns[-1]
        gamma = bn.weight.detach().to(dtype=torch.float64, copy=True)
        beta = bn.bias.detach().to(dtype=torch.float64, copy=True)
        return (gamma, beta, gamma.square() + beta.square())

    @torch.no_grad()
    def head_measurement(self):
        head = self.model.convs[-1]
        left = (head.lin_l.weight.detach().double() * head.lin_l.weight.grad.detach().double()).sum(0)
        right = (head.lin_r.weight.detach().double() * head.lin_r.weight.grad.detach().double()).sum(0)
        return left + right

    def update_block(self, queries=2, scheduler=None):
        if queries not in (1, 2) or not self.model.training:
            raise ValueError('Same period-two core; one query only for an odd tail')
        opt = self.optimizer
        opt.zero_grad(set_to_none=True)
        boundary = self.prefix()
        detached = tuple((value.detach().requires_grad_(True) for value in boundary))
        cotangents = [None, None]
        (losses, learning_rates, measurements) = ([], [], [])
        (gamma, beta, norm_squared) = self.geometry()
        for _ in range(queries):
            opt.zero_grad(set_to_none=True)
            for query in detached:
                query.grad = None
            loss = F.cross_entropy(self.head(detached), self.labels, weight=self.loss_weights)
            loss.backward()
            measurements.append(self.head_measurement())
            for (i, query) in enumerate(detached):
                if cotangents[i] is None:
                    cotangents[i] = query.grad.detach().clone()
                else:
                    cotangents[i].add_(query.grad.detach())
            losses.append(loss.detach())
            learning_rates.append([group['lr'] for group in opt.param_groups])
            opt.step()
            if scheduler is not None:
                scheduler.step()
        next_learning_rates = [group['lr'] for group in opt.param_groups]
        opt.zero_grad(set_to_none=True)
        for query in detached:
            query.grad = None
        torch.autograd.backward(boundary, tuple((g.div_(queries) for g in cotangents)))
        bn = self.model.bns[-1]
        with torch.no_grad():
            means = (bn.weight.grad.detach().clone(), bn.bias.grad.detach().clone())
            half = (measurements[1] - measurements[0]) * 0.5 if queries == 2 else torch.zeros_like(measurements[0])
            valid = norm_squared != 0
            coefficient = half / torch.where(valid, norm_squared, torch.ones_like(norm_squared))
            coefficient = torch.where(valid, coefficient, torch.zeros_like(coefficient))
            corrections = ((gamma * coefficient).to(bn.weight.dtype), (beta * coefficient).to(bn.bias.dtype))
        for (index, rates) in enumerate(learning_rates):
            for (group, lr) in zip(opt.param_groups, rates):
                group['lr'] = lr
            with torch.no_grad():
                sign = -1 if index == 0 else 1
                for (parameter, mean, correction) in zip((bn.weight, bn.bias), means, corrections):
                    parameter.grad.copy_(mean + sign * correction)
                if self.observe is not None:
                    self.observe(index, self, measurements, (gamma, beta, norm_squared), means, corrections)
            opt.step()
        for (group, lr) in zip(opt.param_groups, next_learning_rates):
            group['lr'] = lr
        self.measurements.append((half, norm_squared, coefficient))
        self.actual_updates += queries
        self.work['prefix_forwards'] += 1
        self.work['prefix_VJPs'] += 1
        self.work['head_queries'] += queries
        self.work['head_updates'] += queries
        self.work['body_updates'] += queries
        self.work['supervised_presentations'] += queries * len(self.labels)
        self.work['structural_measurements'] += queries
        self.work['structural_installs'] += queries
        self.work['optimizer_API_calls'] += 2 * queries
        return losses
