import torch
import torch.nn.functional as F
from main.top.arxiv_frozen_mean_p2_v1 import FrozenMeanP2

class CoupledStructural(FrozenMeanP2):

    def __init__(self, model, matrix, features, *, training_ids, observe=None):
        super().__init__(model, matrix, features, training_ids=training_ids)
        self.observe = observe
        self.measurements = []
        self.work.update(structural_queries=0, structural_installs=0, structural_scope='BN0 affine + middle weight rows + BN1 affine', extra_prefix_forwards=0, extra_prefix_VJPs=0, extra_SpMM=0)

    def selected(self):
        (b0, b1) = self.model.bns[:2]
        return (b0.weight, b0.bias, self.model.convs[1].weight, b1.weight, b1.bias)

    @torch.no_grad()
    def geometry(self):
        selected = self.selected()
        directions = tuple(((-p if i == 2 else p).detach().to(torch.float64).clone() for (i, p) in enumerate(selected)))
        norm_squared = sum((n.square().sum(1) if i == 2 else n.square() for (i, n) in enumerate(directions)))
        return (directions, norm_squared)

    def update_block(self, optimizer, labels, method):
        if method != 'A' or not self.model.training or self.cache_alive:
            raise ValueError('This candidate only owns the structural A branch')
        optimizer.zero_grad(set_to_none=True)
        cut = self.prefix()
        leaf = cut.detach().requires_grad_(True)
        self.cache_alive = True
        self._head_factors = []
        measurements = []
        (directions, norm_squared) = self.geometry()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            leaf.grad = None
            logits = self.head(leaf)
            F.cross_entropy(logits[self.training_ids], labels).backward()
            with torch.no_grad():
                head_weight = self.model.convs[-1].weight
                measurement = (head_weight.detach().double() * head_weight.grad.detach().double()).sum(1)
                measurements.append(measurement)
            assert leaf.grad is None
            optimizer.step()
            del logits
        self.peak_named_boundary_bytes = max(self.peak_named_boundary_bytes, 3 * cut.numel() * cut.element_size())
        optimizer.zero_grad(set_to_none=True)
        leaf.grad = None
        from main.top.top_deferred_head_v1 import consume
        from main.top.top_consumed_vjp_v1 import root as consumed_vjp
        cotangent = consume(self._head_factors, cut)
        self._head_factors.clear()
        body_root = consumed_vjp(cut, cotangent.mul_(0.5))
        del leaf, cut, cotangent
        body_root.backward()
        del body_root
        selected = self.selected()
        with torch.no_grad():
            means = tuple((p.grad.detach().clone() for p in selected))
            half_measurement = (measurements[1] - measurements[0]) * 0.5
            valid = norm_squared != 0
            coefficient = half_measurement / torch.where(valid, norm_squared, torch.ones_like(norm_squared))
            coefficient = torch.where(valid, coefficient, torch.zeros_like(coefficient))
            corrections = tuple(((n * coefficient[:, None] if i == 2 else n * coefficient).to(p.dtype) for (i, (n, p)) in enumerate(zip(directions, selected))))
            for (p, q, correction) in zip(selected, means, corrections):
                p.grad.copy_(q - correction)
            if self.observe is not None:
                self.observe('body_first', self, measurements, directions, norm_squared, means, corrections)
        optimizer.step()
        with torch.no_grad():
            for (p, q, correction) in zip(selected, means, corrections):
                p.grad.copy_(q + correction)
            if self.observe is not None:
                self.observe('body_second', self, measurements, directions, norm_squared, means, corrections)
            self.measurements.append((measurements[0], measurements[1], norm_squared, coefficient))
        optimizer.step()
        self.cache_alive = False
        self.blocks += 1
        self.work['prefix_forwards'] += 1
        self.work['prefix_VJPs'] += 1
        self.work['loss_queries'] += 2
        self.work['head_Adam'] += 2
        self.work['body_Adam'] += 2
        self.work['optimizer_API_calls'] += 4
        self.work['supervised_node_presentations'] += 2 * len(self.training_ids)
        self.work['structural_queries'] += 2
        self.work['structural_installs'] += 2
