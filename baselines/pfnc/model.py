import dgl
import dgl.nn as dglnn
import torch
from torch import nn
import torch.nn.functional as F
from baselines.vendor.pfnc.pfn_pkg import PFN

class PaperEmbeddingCache:

    def __init__(self, capacity, width, device, dtype=torch.float32, beta=0.95):
        if capacity <= 0 or not 0 <= beta <= 1:
            raise ValueError('Positive capacity and beta in [0,1] required')
        (self.capacity, self.beta) = (int(capacity), float(beta))
        self.ids = torch.empty(0, device=device, dtype=torch.long)
        self.values = torch.empty((0, width), device=device, dtype=dtype)
        self.scores = torch.empty(0, device=device, dtype=dtype)
        self.pending = None
        self.hit_nodes = torch.zeros((), device=device, dtype=torch.long)
        self.visited_nodes = 0

    def mix(self, ids, fresh):
        if self.pending is not None:
            raise RuntimeError('Prior forward must be committed after backward')
        ids = ids.long()
        if self.ids.numel():
            pos = torch.searchsorted(self.ids, ids)
            safe = pos.clamp(max=self.ids.numel() - 1)
            hit = (pos < self.ids.numel()) & (self.ids[safe] == ids)
            mixed = torch.where(hit[:, None], self.beta * fresh + (1 - self.beta) * self.values[safe], fresh)
            self.hit_nodes += hit.sum()
        else:
            mixed = fresh
        mixed.retain_grad()
        self.visited_nodes += ids.numel()
        self.pending = (ids.detach(), mixed)
        return mixed

    @torch.no_grad()
    def commit(self):
        if self.pending is None or self.pending[1].grad is None:
            raise RuntimeError('Cache importance requires post-mix cotangent')
        (current, mixed) = self.pending
        current_sorted = current.sort().values
        if current_sorted.numel() and self.ids.numel():
            pos = torch.searchsorted(current_sorted, self.ids)
            safe = pos.clamp(max=current_sorted.numel() - 1)
            keep_old = (pos == current_sorted.numel()) | (current_sorted[safe] != self.ids)
        else:
            keep_old = torch.ones_like(self.ids, dtype=torch.bool)
        ids = torch.cat((self.ids[keep_old], current))
        values = torch.cat((self.values[keep_old], mixed.detach()))
        scores = torch.cat((self.scores[keep_old], mixed.grad.detach().norm(dim=1)))
        by_id = ids.argsort(stable=True)
        ranked = scores[by_id].argsort(descending=True, stable=True)[:self.capacity]
        selected = by_id[ranked]
        selected = selected[ids[selected].argsort()]
        self.ids = ids[selected].clone()
        self.values = values[selected].clone()
        self.scores = scores[selected].clone()
        self.pending = None

class CommonSAGE(nn.Module):

    def __init__(self, inputs, hidden, outputs, dropout=0.5):
        super().__init__()
        dims = [inputs, hidden, hidden, outputs]
        self.convs = nn.ModuleList((dglnn.SAGEConv(dims[i], dims[i + 1], 'mean') for i in range(3)))
        self.bns = nn.ModuleList((nn.BatchNorm1d(hidden) for _ in range(2)))
        self.dropout = float(dropout)

    def layer(self, index, graph, x):
        x = self.convs[index](graph, x)
        if index < 2:
            x = F.dropout(F.relu(self.bns[index](x)), p=self.dropout, training=self.training)
        return x

    def forward(self, blocks, x, caches=None):
        if len(blocks) != 3 or (caches is not None and (not self.training)):
            raise ValueError('Three blocks required; caches are train-only')
        for (index, block) in enumerate(blocks):
            x = self.layer(index, block, x)
            if caches is not None:
                x = caches[index].mix(block.dstdata[dgl.NID], x)
        return F.log_softmax(x, dim=-1)

    def full_forward(self, graph, x):
        for index in range(3):
            x = self.layer(index, graph, x)
        return F.log_softmax(x, dim=-1)

    def load_fresh_state(self, source):
        mapped = {}
        for (name, value) in source.items():
            dest = name.replace('.lin_l.weight', '.fc_neigh.weight')
            dest = dest.replace('.lin_l.bias', '.fc_self.bias')
            dest = dest.replace('.lin_r.weight', '.fc_self.weight')
            mapped[dest] = value
        self.load_state_dict(mapped, strict=True)
        if len(mapped) != 19:
            raise ValueError('Expected 13 parameters and 6 BN buffers')

def released_gradient_module(model, num_nodes, queue_size=16, alpha=0.9):
    device = next(model.parameters()).device
    pfn = PFN(num_nodes=num_nodes, num_layers=3, queue_size=queue_size, queue_device=device, grad_normlized_add=True, grad_normlized_add_alpha=alpha, grad_update_again=False, add_padding_queues=False, add_param_value=False, use_grad_distances_check=False, not_use_nid_access=False, grad_add=False, grad_inflation=False)
    from collections import OrderedDict
    layers = [nn.Sequential(OrderedDict([('conv', model.convs[i])] + ([('bn', model.bns[i])] if i < 2 else []))) for i in range(3)]
    pfn.generate_meta(layers, [f'layer_{i}' for i in range(3)], [0, 1, 2])
    pfn.init_all_history_queues()
    actual = [meta[2] for group in pfn.meta for meta in group]
    if len(actual) != 13 or {id(p) for p in actual} != {id(p) for p in model.parameters()}:
        raise AssertionError('Every active parameter must be tracked exactly once')
    return pfn

class PFNCStep:

    def __init__(self, model, optimizer, nodes, capacity, beta=0.95, alpha=0.9, queue_size=16):
        (self.model, self.optimizer) = (model, optimizer)
        p = next(model.parameters())
        widths = [model.convs[0]._out_feats, model.convs[1]._out_feats, model.convs[2]._out_feats]
        self.caches = [PaperEmbeddingCache(capacity, d, p.device, p.dtype, beta) for d in widths]
        self.pfn = released_gradient_module(model, nodes, queue_size, alpha)
        self.updates = 0
        self.root_hit_nodes = torch.zeros((), device=p.device, dtype=torch.long)
        self.root_hit_slots = torch.zeros((), device=p.device, dtype=torch.long)
        self.root_presentations = 0

    def step(self, blocks, x, roots, labels):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(blocks, x, self.caches)
        loss = F.nll_loss(output, labels)
        loss.backward()
        key = self.pfn.meta[0][0][0]
        old = self.pfn.queue_pos_for_nodes[0][key][roots]
        hit = old >= max(0, self.updates - self.pfn.queue_size)
        self.root_hit_nodes += hit.sum()
        self.root_hit_slots += old[hit].unique().numel()
        self.pfn.detached_grad_correction([roots] * 3)
        self.optimizer.step()
        for cache in self.caches:
            cache.commit()
        self.updates += 1
        self.root_presentations += roots.numel()
        return loss.detach()

    def stats(self):
        return dict(updates=self.updates, root_presentations=self.root_presentations, root_hit_nodes=int(self.root_hit_nodes), root_hit_slots=int(self.root_hit_slots), embedding_hit_nodes=[int(c.hit_nodes) for c in self.caches], embedding_visited_nodes=[c.visited_nodes for c in self.caches], cache_entries=[c.ids.numel() for c in self.caches])
