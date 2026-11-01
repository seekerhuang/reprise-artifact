from empirical.gas.yelp.yelp_boundary import Training as PreviousTraining
import time
import torch

class Training(PreviousTraining):

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        if model.batch_norm or model.dropout != 0.0 or model.drop_input:
            raise ValueError('Require original Yelp noBN/drop0/no-inputdrop')
        if any((h.emb.requires_grad for h in model.histories)):
            raise ValueError('Outside history must remain fixed')
        self.W = model.convs[0].lin.weight
        self.b = model.convs[0].bias
        if self.b is None or set(self.body) != {'convs.0.lin.weight', 'convs.0.bias'}:
            raise ValueError('Original two Body tensors required')
        self.measurements = self.installs = self.transposes = self.local_products = 0
        self.stat_sum = self.stat_max = None
        self.window_contexts = 0
        self.geometry = []

    def forward_capture(self, x, adj, core, args):
        from torch_sparse import matmul
        saved = self.readout.forward
        captured = {}

        def readout(hidden, matrix):
            if captured or matrix.sparse_sizes() != (core, hidden.shape[0]):
                raise RuntimeError('Unexpected native final readout geometry')
            value = matrix.storage.value()
            if value is not None and value.requires_grad:
                raise ValueError('Graph values must be parameter independent')
            captured.update(live=hidden[:core].detach(), matrix=matrix.narrow(1, 0, core), neighbor_rows=hidden.shape[0])
            return matmul(matrix, hidden, reduce='sum')
        try:
            self.readout.forward = readout
            cut = self.model(x, adj, core, *args)
        finally:
            self.readout.forward = saved
        if not captured or not cut.requires_grad:
            raise RuntimeError('Missing original prefix tape')
        return (cut, captured)

    def block(self, x, adj, core, args, labels, mask, criterion, *, label_count):
        from torch_sparse import matmul
        if self.cache_alive:
            raise RuntimeError('Previous prefix tape retained')
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        self.cache_alive = True
        try:
            (cut, captured) = self.forward_capture(x, adj, core, args)
            query = cut.detach().requires_grad_(True)
            cotangent = None
            losses = []
            deltas = []
            weights = []
            for _ in range(2):
                query.grad = None
                logits = self.tail(query)
                logits.retain_grad()
                loss = criterion(logits[mask], labels[mask])
                loss.backward()
                if cotangent is None:
                    cotangent = query.grad.detach().clone()
                else:
                    cotangent.add_(query.grad)
                if logits.grad is None:
                    raise RuntimeError('Missing actual masked/reduced logit derivative')
                deltas.append(logits.grad.detach())
                weights.append(self.readout.lin.weight.detach().clone())
                self.measurements += 1
                self._step()
            query.grad = None
            cotangent.mul_(0.5)
            cut.backward(cotangent)
            if self.W.grad is None or self.b.grad is None:
                raise RuntimeError('Missing actual Body mean gradient')
            with torch.no_grad():
                transported = matmul(captured['matrix'].t(), torch.cat(deltas, dim=1), reduce='sum')
                self.transposes += 1
                classes = weights[0].shape[0]
                local0 = transported[:, :classes].T.matmul(captured['live'])
                mu0 = (weights[0].double() * local0.double()).sum(0)
                local1 = transported[:, classes:].T.matmul(captured['live'])
                mu1 = (weights[1].double() * local1.double()).sum(0)
                self.local_products += 2
                n = torch.cat([self.W.detach().double(), self.b.detach().double().unsqueeze(1)], dim=1)
                q = torch.cat([self.W.grad.detach().double(), self.b.grad.detach().double().unsqueeze(1)], dim=1)
                den = n.square().sum(1)
                live = den != 0
                safe = torch.where(live, den, torch.ones_like(den))
                half = (mu1 - mu0) * 0.5
                r = n * torch.where(live, half / safe, torch.zeros_like(safe)).unsqueeze(1)
                self.window_contexts += 1
                del transported, local0, local1, deltas, weights
                for sign in (-1.0, 1.0):
                    value = q + sign * r
                    self.W.grad = value[:, :-1].to(self.W.dtype)
                    self.b.grad = value[:, -1].to(self.b.dtype)
                    self._step()
                    self.installs += 1
            self.blocks += 1
            self.visits += 2
            self.labels += label_count
            return losses
        finally:
            self.cache_alive = False
