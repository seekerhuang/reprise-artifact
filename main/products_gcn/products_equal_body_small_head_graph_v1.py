import torch
import torch.nn.functional as F
from main.products_gcn.products_head_second_small_graph_v1 import GraphMixin
from main.products_gcn.sustained_head_rate_v1 import HeadRateMixin, head_rate
from main.products_gcn.products_structural_no_scalar_sync_v1 import BNStructuralMeanP2

class EqualBodyContract:

    def head_rate_contract(self):
        contract = super().head_rate_contract()
        contract['Body_slot_multipliers'] = [1.0, 1.0]
        contract['scope'] = 'Original two Head queries and two equal-rate native Body submissions; only second Head rate reduced from80.'
        return contract

class EqualBodySmallSecondStructural(EqualBodyContract, GraphMixin, HeadRateMixin, BNStructuralMeanP2):

    def update_block(self):
        self.model.train()
        opt = self.optimizer
        opt.zero_grad(set_to_none=True)
        boundary = self.prefix()
        query = boundary.detach().requires_grad_(True)
        average = None
        measurements = []
        head = self.model.convs[-1]
        for head_position in range(2):
            opt.zero_grad(set_to_none=True)
            query.grad = None
            loss = F.cross_entropy(F.linear(query, head.lin.weight, head.bias), self.labels)
            loss.backward()
            with torch.no_grad():
                measurements.append((head.lin.weight.detach().double() * head.lin.weight.grad.detach().double()).sum(0))
            if average is None:
                average = query.grad.detach().clone()
            else:
                average.add_(query.grad.detach())
            with head_rate(self, head_position):
                opt.step()
        opt.zero_grad(set_to_none=True)
        query.grad = None
        boundary.backward(average.mul_(0.5))
        bn = self.model.bns[1]
        with torch.no_grad():
            decoded = self.recovery_graph.decode(measurements)
            self.stat.add_(decoded.values)
            self.last_measurement = decoded.last_measurement
            (bn.weight.grad, bn.bias.grad) = decoded.pair[0]
        opt.step()
        with torch.no_grad():
            (bn.weight.grad, bn.bias.grad) = decoded.pair[1]
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.blocks += 1
        self.updates += 2
        self.work['prefix_forwards'] += 1
        self.work['prefix_VJPs'] += 1
        for name in ('head_queries', 'head_updates', 'body_updates'):
            self.work[name] += 2
        self.work['supervised_presentations'] += 2 * len(self.labels)
