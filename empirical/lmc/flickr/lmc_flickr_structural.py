import torch
from empirical.lmc.flickr import lmc_flickr_boundary as old
from empirical.lmc.flickr.lmc_flickr_boundary import BODY_NAMES, HEAD_NAMES, UNUSED_NAMES, prefix, head, core
S_NAMES = ('bns.0.bn_ib.weight', 'bns.0.bn_ib.bias', 'bns.0.bn_ob.weight', 'bns.0.bn_ob.bias')
STAT_NAMES = ('measurement_difference_norm', 'correction_norm', 'mean_S_norm', 'mean_constraint_residual_norm', 'zero_direction_channels')

class FrozenMeanP2(old.FrozenMeanP2):

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.S = tuple((self.parameters[k] for k in S_NAMES))
        self.measurements = self.installs = 0
        self.stat_sum = self.stat_max = None
        self.window_contexts = 0

    def block(self, condition):
        if self.cache_alive:
            raise RuntimeError('Previous structural context retained')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        boundary = prefix(self.model, condition)
        query = boundary.detach().requires_grad_(True)
        self.cache_alive = True
        cotangent = None
        losses = []
        mu = []
        for _ in range(2):
            self.optimizer.zero_grad(set_to_none=True)
            query.grad = None
            loss = core.conditional_loss(head(self.model, query), condition)
            loss.backward()
            if cotangent is None:
                cotangent = query.grad.detach().clone()
            else:
                cotangent.add_(query.grad.detach())
            if any((self.parameters[k].grad is not None for k in BODY_NAMES + UNUSED_NAMES)):
                raise RuntimeError('Head phase acquired prefix/unused gradients')
            if any((self.parameters[k].grad is None for k in HEAD_NAMES)):
                raise RuntimeError('Missing Head derivative')
            W = self.model.convs[-1].lin.weight
            mu.append((W.detach().double() * W.grad.detach().double()).sum(0))
            self.measurements += 1
            self.optimizer.step()
            losses.append(loss.detach())
        self.optimizer.zero_grad(set_to_none=True)
        query.grad = None
        boundary.backward(cotangent.mul_(0.5))
        if any((self.parameters[k].grad is not None for k in HEAD_NAMES + UNUSED_NAMES)):
            raise RuntimeError('Body phase acquired Head/unused gradients')
        if any((self.parameters[k].grad is None for k in BODY_NAMES)):
            raise RuntimeError('Missing Body mean derivative')
        n = torch.stack([p.detach().double() for p in self.S])
        q = torch.stack([p.grad.detach().double() for p in self.S])
        denominator = n.square().sum(0)
        live = denominator != 0
        half_difference = (mu[1] - mu[0]) * 0.5
        safe = torch.where(live, denominator, torch.ones_like(denominator))
        r = n * torch.where(live, half_difference / safe, torch.zeros_like(safe)).unsqueeze(0)
        for sign in (-1.0, 1.0):
            for (j, p) in enumerate(self.S):
                p.grad = (q[j] + sign * r[j]).to(p.dtype)
            self.optimizer.step()
            self.installs += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.cache_alive = False
        self.blocks += 1
        self.updates += 2
        self.receiver_labels += len(condition.labels)
        self.core_labels += condition.core_training_labels
        return losses
