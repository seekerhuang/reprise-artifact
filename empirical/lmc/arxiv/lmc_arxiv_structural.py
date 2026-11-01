import torch
from empirical.lmc.arxiv.lmc_arxiv_boundary import FrozenMeanP2, prefix, head, core

class LMCStructural(FrozenMeanP2):

    def __init__(self, *args, observe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe = observe
        self.measurements = []
        self.work.update(structural_measurements=0, structural_installs=0, extra_prefix_forwards=0, extra_prefix_VJPs=0, extra_graph_multiplications=0)

    def update_block(self, condition):
        if not self.model.training or self.cache_alive:
            raise RuntimeError('Expected one fresh training context')
        self.optimizer.zero_grad(set_to_none=True)
        boundary = prefix(self.model, condition)
        query = boundary.detach().requires_grad_(True)
        self.cache_alive = True
        bn = self.model.bns[1]
        parameters = (bn.bn_ib.weight, bn.bn_ib.bias, bn.bn_ob.weight, bn.bn_ob.bias)
        directions = tuple((p.detach().to(dtype=torch.float64, copy=True) for p in parameters))
        norm = sum((n.square() for n in directions))
        cotangent = None
        (measurements, losses) = ([], [])
        for _ in range(2):
            self.optimizer.zero_grad(set_to_none=True)
            query.grad = None
            loss = core.conditional_loss(head(self.model, query), condition)
            loss.backward()
            with torch.no_grad():
                weight = self.model.convs[-1].lin.weight
                measurements.append((weight.detach().double() * weight.grad.detach().double()).sum(0))
            if cotangent is None:
                cotangent = query.grad.detach().clone()
            else:
                cotangent.add_(query.grad.detach())
            losses.append(loss.detach())
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        query.grad = None
        boundary.backward(cotangent.mul_(0.5))
        with torch.no_grad():
            means = tuple((p.grad.detach().clone() for p in parameters))
            half = (measurements[1] - measurements[0]) * 0.5
            coefficient = half / torch.where(norm != 0, norm, torch.ones_like(norm))
            coefficient = torch.where(norm != 0, coefficient, torch.zeros_like(coefficient))
            corrections = tuple(((n * coefficient).to(p.dtype) for (p, n) in zip(parameters, directions)))
        for (index, sign) in enumerate((-1, 1)):
            with torch.no_grad():
                for (p, q, r) in zip(parameters, means, corrections):
                    p.grad.copy_(q + sign * r)
                if self.observe is not None:
                    self.observe(index, self, measurements, directions, means, corrections)
            self.optimizer.step()
        self.measurements.append((half, norm, coefficient))
        self.cache_alive = False
        self.actual_updates += 2
        self.work['prefix_forwards'] += 1
        self.work['prefix_VJPs'] += 1
        self.work['loss_queries'] += 2
        self.work['head_updates'] += 2
        self.work['body_updates'] += 2
        self.work['supervised_node_presentations'] += 2 * len(condition.labels)
        self.work['structural_measurements'] += 2
        self.work['structural_installs'] += 2
        return losses
