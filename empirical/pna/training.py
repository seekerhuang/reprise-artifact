from itertools import product
import torch
METHODS = ('A',)

class _UnitSeedMeanVJP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *ports_and_cotangents):
        if len(ports_and_cotangents) != 10:
            raise RuntimeError('Exactly five original native ports/cotangents required')
        ports = ports_and_cotangents[:5]
        cotangents = ports_and_cotangents[5:]
        if any((p.shape != g.shape or p.dtype != g.dtype or p.device != g.device or g.requires_grad for (p, g) in zip(ports, cotangents))):
            raise RuntimeError('Original detached mean-cotangent contract changed')
        ctx.cotangents = cotangents
        ctx.set_materialize_grads(False)
        return ports[0].new_zeros(())

    @staticmethod
    def backward(ctx, unit_seed):
        if unit_seed is None or unit_seed.ndim != 0 or torch.is_grad_enabled():
            raise RuntimeError('Private unit-seed, first-order mean VJP required')
        gradients = ctx.cotangents
        ctx.cotangents = None
        return (*gradients, None, None, None, None, None)

class BaseTraining:

    def __init__(self, model, optimizer, method):
        if method not in METHODS:
            raise ValueError(method)
        (self.model, self.optimizer, self.method) = (model, optimizer, method)
        self.readout = model.convs[-1]
        if model.num_layers != 4 or model.residual or (not model.batch_norm) or (not model.drop_input) or (model.dropout != 0.5) or (self.readout.aggregators != ['mean', 'max']) or (self.readout.scalers != ['identity', 'amplification']):
            raise ValueError('Locked original PNA4/BN/inputdrop.5/mean-max recipe required')
        named = dict(model.named_parameters())
        head_ids = {id(p) for lin in list(self.readout.post_lins) + [self.readout.lin] for p in lin.parameters()}
        self.head = {k: p for (k, p) in named.items() if id(p) in head_ids}
        self.body = {k: p for (k, p) in named.items() if id(p) not in head_ids}
        if len(self.head) != 10 or len(self.body) != 68 or len(model.state_dict()) != 87:
            raise ValueError('PNA actual87 / body68 / head10 schema differs')
        if {id(p) for p in named.values()} != {id(p) for group in optimizer.param_groups for p in group['params']}:
            raise ValueError('Original optimizer parameter identities differ')
        if any((g['weight_decay'] or g['lr'] != 0.005 for g in optimizer.param_groups)):
            raise ValueError('Original no-decay/.005 Adam required')
        self.blocks = self.visits = self.optimizer_calls = 0
        self.prefix_forwards = self.prefix_VJPs = self.head_queries = 0
        self.supervised_presentations = self.independent_presentations = 0
        self.cache_peak_payload_bytes = self.rng_peak_CPU_bytes = 0
        self.cache_alive = False

    def _ports(self, x, adj):
        from torch_scatter import segment_csr
        deg = adj.storage.rowcount().to(x.dtype).view(-1, 1)
        amplification = (deg + 1).log() / self.readout.avg_deg['log']
        ports = []
        for ((aggr, _scaler), pre) in zip(product(self.readout.aggregators, self.readout.scalers), self.readout.pre_lins):
            h = pre(x).relu_()
            ports.append(segment_csr(h[adj.storage.col()], adj.storage.rowptr(), reduce=aggr))
        ports.append(x)
        return (tuple(ports), amplification)

    def prefix(self, x, adj, core, ids):
        old = self.readout.forward
        try:
            self.readout.forward = self._ports
            return self.model(x, adj, core, ids)
        finally:
            self.readout.forward = old

    def tail(self, ports, amplification):
        out = 0
        for ((_aggr, scaler), post, h) in zip(product(self.readout.aggregators, self.readout.scalers), self.readout.post_lins, ports):
            h = post(h)
            if scaler == 'amplification':
                h = h * amplification
            out = out + h
        out = out + self.readout.lin(ports[-1])[:out.size(0)]
        return out

    def _step(self):
        self.optimizer.step()
        self.optimizer_calls += 1
        self.optimizer.zero_grad(set_to_none=True)

    def verify(self, steps):
        clocks = {k: int(self.optimizer.state.get(p, {}).get('step', 0)) for (k, p) in self.model.named_parameters()}
        if self.visits != steps or self.cache_alive or set(clocks.values()) != {steps}:
            raise RuntimeError('All78 parameter clocks or cache boundary violated')
        if any((p.grad is not None for p in self.model.parameters())):
            raise RuntimeError('Submitted gradients remained alive')
        if [int(b.num_batches_tracked) for b in self.model.bns] != [self.prefix_forwards] * 3:
            raise RuntimeError('Actual prefix BN clocks differ')
        return clocks

class Training(BaseTraining):

    def __init__(self, model, optimizer, method):
        super().__init__(model, optimizer, method)
        self.S = [(pre.weight, pre.bias) for pre in self.readout.pre_lins]
        if len(self.S) != 4 or any((b is None for (w, b) in self.S)):
            raise ValueError('Four native affine pre branches required')
        self.measurements = self.installs = 0
        self.stat_sum = self.stat_max = None
        self.window_contexts = 0

    def block(self, x, adj, core, ids, labels, mask, criterion, *, label_count):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.cache_alive:
            raise RuntimeError('Previous PNA context retained')
        self.cache_alive = True
        try:
            (ports, scale) = self.prefix(x, adj, core, ids)
            queries = tuple((h.detach().requires_grad_(True) for h in ports))
            cotangents = None
            losses = []
            mu = []
            pending_head2 = None
            for head_slot in range(2):
                for q in queries:
                    q.grad = None
                loss = criterion(self.tail(queries, scale)[mask], labels[mask])
                loss.backward()
                if cotangents is None:
                    cotangents = [q.grad.detach().clone() for q in queries]
                else:
                    for (g, q) in zip(cotangents, queries):
                        g.add_(q.grad)
                mu.append(torch.stack([(post.weight.detach().double() * post.weight.grad.detach().double()).sum(0) for post in self.readout.post_lins]))
                self.measurements += 1
                if head_slot == 0:
                    self._step()
                else:
                    pending_head2 = tuple((p.grad for p in self.head.values()))
                    if any((g is None for g in pending_head2)):
                        raise RuntimeError('Original PNA Head2 explicit support missing')
                    self.optimizer.zero_grad(set_to_none=True)
            for q in queries:
                q.grad = None
            for g in cotangents:
                g.mul_(0.5)
            _body_VJP_root = _UnitSeedMeanVJP.apply(*ports, *cotangents)
            del ports, queries, cotangents, loss, q, g, scale
            _body_VJP_root.backward()
            del _body_VJP_root
            mean = {k: p.grad for (k, p) in self.body.items()}
            projection = []
            for (k, (w, b)) in enumerate(self.S):
                n = torch.cat([w.detach().double(), b.detach().double().unsqueeze(1)], dim=1)
                q = torch.cat([w.grad.detach().double(), b.grad.detach().double().unsqueeze(1)], dim=1)
                den = n.square().sum(1)
                live = den != 0
                safe = torch.where(live, den, torch.ones_like(den))
                half = (mu[1][k] - mu[0][k]) * 0.5
                r = n * torch.where(live, half / safe, torch.zeros_like(safe)).unsqueeze(1)
                projection.append((q, r))
            self.window_contexts += 1
            for sign in (-1.0, 1.0):
                for (k, p) in self.body.items():
                    p.grad = mean[k]
                for ((w, b), (q, r)) in zip(self.S, projection):
                    value = q + sign * r
                    w.grad = value[:, :-1].to(w.dtype)
                    b.grad = value[:, -1].to(b.dtype)
                if sign == 1.0:
                    for (p, g) in zip(self.head.values(), pending_head2):
                        p.grad = g
                self._step()
                self.installs += 1
            pending_head2 = None
            self.blocks += 1
            self.visits += 2
            self.prefix_forwards += 1
            self.prefix_VJPs += 1
            self.head_queries += 2
            self.independent_presentations += label_count
            self.supervised_presentations += 2 * label_count
            return losses
        finally:
            self.cache_alive = False
