import torch
import torch.nn.functional as F
from main.iglu.iglu_prefix import FrozenMeanP2, prefix

class IGLUStructural(FrozenMeanP2):

    def __init__(self, *args, observe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe = observe
        self.measurements = []
        self.work.update(structural_measurements=0, structural_installs=0, extra_prefix_forwards=0, extra_prefix_VJPs=0, extra_graph_multiplications=0)

    def update_block(self, batch, lr):
        condition = self.geometry.conditions[batch]
        for opt in self.optimizers.values():
            for group in opt.param_groups:
                group['lr'] = lr
        self.model.zero_grad(set_to_none=True)
        boundary = prefix(self.model, self.geometry, condition, self.dropout)
        query = boundary.detach().requires_grad_(True)
        parameters = (self.model.scale2, self.model.offset2)
        directions = tuple((p.detach().to(dtype=torch.float64, copy=True) for p in parameters))
        norm = sum((n.square() for n in directions))
        cotangent = None
        measurements = []
        for _ in range(2):
            self.model.zero_grad(set_to_none=True)
            query.grad = None
            loss = F.cross_entropy(query @ self.model.w + self.model.b3, condition.labels, reduction='sum')
            loss.backward()
            with torch.no_grad():
                measurements.append((self.model.w.detach().double() * self.model.w.grad.detach().double()).sum(1).unsqueeze(0))
            if cotangent is None:
                cotangent = query.grad.detach().clone()
            else:
                cotangent.add_(query.grad.detach())
            self.optimizers['w'].step()
        self.model.zero_grad(set_to_none=True)
        query.grad = None
        boundary.backward(cotangent.mul_(0.5))
        with torch.no_grad():
            means = tuple((p.grad.detach().clone() for p in parameters))
            half = (measurements[1] - measurements[0]) * 0.5
            coefficient = half / torch.where(norm != 0, norm, torch.ones_like(norm))
            coefficient = torch.where(norm != 0, coefficient, torch.zeros_like(coefficient))
            corrections = tuple(((n * coefficient).to(p.dtype) for (n, p) in zip(directions, parameters)))
        for (index, sign) in enumerate((-1, 1)):
            with torch.no_grad():
                for (p, q, r) in zip(parameters, means, corrections):
                    p.grad.copy_(q + sign * r)
                if self.observe is not None:
                    self.observe(index, self, measurements, directions, means, corrections)
            self.optimizers['Q'].step()
            self.optimizers['R'].step()
        self.measurements.append((half, norm, coefficient))
        self.actual_updates += 2
        self.work['prefix_forwards'] += 1
        self.work['prefix_VJPs'] += 1
        self.work['head_queries'] += 2
        self.work['optimizer_API_calls'] += 6
        self.work['supervised_presentations'] += 2 * len(condition.labels)
        self.work['structural_measurements'] += 2
        self.work['structural_installs'] += 2
