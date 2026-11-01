import torch
import torch.nn.functional as F
from main.sage.sage_structural import SampledSAGEStructural as Original
from main.sage.sage_head_query import HeadQuery

class ExactHeadSAGE(Original):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exact_head = HeadQuery(self)

    def update_block(self, queries=2, scheduler=None):
        if queries not in (1, 2) or not self.model.training:
            raise ValueError('Same period-two core; one query only for an odd tail')
        opt = self.optimizer
        opt.zero_grad(set_to_none=True)
        boundary = self.prefix()
        detached = tuple((value.detach().requires_grad_(True) for value in boundary))
        self._exact_head.begin(detached, self.labels, queries)
        cotangents = [None, None]
        (losses, learning_rates, measurements) = ([], [], [])
        (gamma, beta, norm_squared) = self.geometry()
        retained_head_gradients = None
        for query_index in range(queries):
            opt.zero_grad(set_to_none=True)
            for query in detached:
                query.grad = None
            (loss, _exact_mu) = self._exact_head.query(detached, self.labels, query_index)
            measurements.append(_exact_mu)
            for (i, query) in enumerate(detached):
                if cotangents[i] is None:
                    cotangents[i] = query.grad.detach().clone()
                else:
                    cotangents[i].add_(query.grad.detach())
            losses.append(loss.detach())
            learning_rates.append([group['lr'] for group in opt.param_groups])
            if query_index + 1 < queries:
                opt.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                retained_head_gradients = tuple(((p, p.grad) for p in self.model.convs[-1].parameters()))
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
            if index + 1 == queries:
                for (parameter, gradient) in retained_head_gradients:
                    parameter.grad = gradient
            opt.step()
        if scheduler is not None:
            scheduler.step()
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
        self.work['optimizer_API_calls'] += 2 * queries - 1
        self._exact_head.finish_context()
        return losses
