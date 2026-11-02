import torch
from empirical.gas.gas_boundary import Training as BaseTraining
from empirical.gas.flickr import flickr_rates
S_NAMES = ('bns.0.weight', 'bns.0.bias')
rule = 'narrow_radial'

def complete_radial(n, q, a0, a1):
    import torch
    den = n.square().sum(0)
    nonzero = den != 0
    coeff = torch.where(nonzero, (a1 - a0) * 0.5 / torch.where(nonzero, den, torch.ones_like(den)), torch.zeros_like(den))
    r = n * coeff.unsqueeze(0)
    return (r, None, None)

class FlickrTraining(BaseTraining):

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        if not model.batch_norm:
            raise ValueError('The locked Flickr backbone has active BN0')
        named = dict(model.named_parameters())
        self.S = tuple((named[k] for k in S_NAMES))
        if any((a is not b for (a, b) in zip(self.S, (model.bns[0].weight, model.bns[0].bias)))):
            raise ValueError('Require the actual active BN affine parameters')
        if any((h.emb.requires_grad for h in model.histories)):
            raise ValueError('Outside history is a constant for this native training call')
        self.wide_forwards = self.narrow_transposes = self.live_expansions = 0
        self.logit_queries = self.weight_snapshots = self.nonlinear_vjps = 0
        self.stat_sum = self.stat_max = None
        self.recorded_blocks = 0
        self.last_statistics = None
        self.geometry = []

    def _forward_capture(self, x, adj, core, args):
        from torch_sparse import matmul
        captured = {}
        old = self.readout.forward

        def readout(hidden, matrix):
            if captured:
                raise RuntimeError('Exactly one last readout forward is required')
            if matrix.sparse_sizes() != (core, hidden.shape[0]):
                raise ValueError('Require original core-first bipartite GAS geometry')
            value = matrix.storage.value()
            if value is not None and value.requires_grad:
                raise ValueError('This adapter requires parameter-independent graph values')
            captured.update(hidden=hidden, live=hidden[:core], matrix=matrix)
            return matmul(matrix, hidden.detach(), reduce='sum')
        try:
            self.readout.forward = readout
            cut = self.model(x, adj, core, *args)
        finally:
            self.readout.forward = old
        if len(captured) != 3 or cut.requires_grad or cut.shape != (core, self.readout.lin.weight.shape[1]):
            raise RuntimeError('Wide forward/cut ownership changed')
        self.wide_forwards += 1
        return (cut, captured)

    def block(self, x, adj, core, args, labels, mask, criterion, *, label_count):
        import torch
        from torch_sparse import matmul
        if self.cache_alive:
            raise RuntimeError('Prior prefix tape was not released')
        rate = flickr_rates.rate_at(self.blocks)
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        self.optimizer.defaults['lr'] = rate
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        self.cache_alive = True
        losses = []
        deltas = []
        weights = []
        try:
            (cut, captured) = self._forward_capture(x, adj, core, args)
            query = cut.detach()
            if query.requires_grad:
                raise RuntimeError('Wide input-gradient path must stay disabled')
            pending_head2 = None
            for head_slot in range(2):
                logits = self.tail(query)
                logits.retain_grad()
                loss = criterion(logits[mask], labels[mask])
                loss.backward()
                if logits.grad is None:
                    raise RuntimeError('Actual masked/reduced logits derivative missing')
                deltas.append(logits.grad.detach())
                weights.append(self.readout.lin.weight.detach().clone())
                self.logit_queries += 1
                self.weight_snapshots += 1
                if head_slot == 0:
                    self._step()
                else:
                    pending_head2 = tuple((p.grad for p in self.head.values()))
                    if any((g is None for g in pending_head2)):
                        raise RuntimeError('Original complete Head2 gradient support missing')
                    self.optimizer.zero_grad(set_to_none=True)
                del loss, logits
            packed = torch.cat(deltas, dim=1)
            transported = matmul(captured['matrix'].t(), packed, reduce='sum')
            self.narrow_transposes += 1
            classes = weights[0].shape[0]
            if packed.shape[1] != 2 * classes:
                raise RuntimeError('True two-query class width changed')
            live = captured['live']
            response0 = transported[:core, :classes].matmul(weights[0])
            response1 = transported[:core, classes:].matmul(weights[1])
            self.live_expansions += 2
            live64 = live.detach().double()
            a0 = (live64 * response0.detach().double()).sum(0)
            a1 = (live64 * response1.detach().double()).sum(0)
            mean = response0.add(response1).mul_(0.5)
            del response0, response1, live64, packed, transported, deltas, weights
            live.backward(mean)
            self.nonlinear_vjps += 1
            gradient = {k: p.grad for (k, p) in self.body.items()}
            if any((g is None for g in gradient.values())):
                raise RuntimeError('An active Body gradient is missing')
            n = torch.stack([p.detach().double() for p in self.S])
            q = torch.stack([p.grad.detach().double() for p in self.S])
            (r, residual, zero_channels) = complete_radial(n, q, a0, a1)
            for position in range(2):
                if position:
                    for (k, p) in self.body.items():
                        p.grad = gradient[k]
                if rule == 'narrow_radial':
                    sign = -1.0 if position == 0 else 1.0
                    for (index, p) in enumerate(self.S):
                        p.grad = (q[index] + sign * r[index]).to(p.dtype)
                if position:
                    for (p, g) in zip(self.head.values(), pending_head2):
                        p.grad = g
                self._step()
            pending_head2 = None
            self.blocks += 1
            self.visits += 2
            self.labels += label_count
            return losses
        finally:
            self.cache_alive = False
