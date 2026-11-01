from empirical.gas.reddit.reddit_static_first import Training as PreviousTraining
import time
import torch

class Training(PreviousTraining):

    def __init__(self, model, optimizer, observe=None):
        super().__init__(model, optimizer)
        self.observe = observe
        self.live_hidden = None
        self.live_matrix = None
        self.measurement_blocks = 0
        self.structural_measurements = []

    def prefix(self, x, adj, core, *args):
        import torch.nn.functional as F
        from torch_sparse import matmul
        if self.static_input is None or self.supervised_rows is None or len(args) != 3:
            raise RuntimeError('Original fixed PX and supervised native coordinates required')
        (n_id, offset, count) = args
        if n_id.device.type != 'cpu' or len(n_id) != x.shape[0]:
            raise RuntimeError('Native complete input coordinates required')
        first = self.model.convs[0]
        (old_first, old_last) = (first.forward, self.readout.forward)

        def cached_first(unused_x, matrix):
            if matrix.size(0) != core:
                raise RuntimeError('Only complete first-layer core rows')
            fixed = self.static_input.index_select(0, n_id[:core].to(self.model.device))
            self.cached_first_queries += 1
            self.cached_first_rows += core
            return F.linear(fixed, first.lin.weight, first.bias)

        def selected_last(hidden, matrix):
            selected = matrix.index_select(0, self.supervised_rows)
            self.original_last_rows += matrix.size(0)
            self.executed_last_rows += selected.size(0)
            self.live_hidden = hidden[:core].detach()
            self.live_matrix = selected.narrow(1, 0, core)
            return matmul(selected, hidden, reduce='sum')
        try:
            first.forward = cached_first
            self.readout.forward = selected_last
            return self.model(x, adj, core, *args)
        finally:
            first.forward = old_first
            self.readout.forward = old_last

    def block(self, x, adj, core, args, labels, mask, criterion, *, label_count, row_ids=None):
        import torch
        from torch_sparse import matmul
        if self.cache_alive or self.supervised_rows is not None or mask.ndim != 1 or (len(mask) != core):
            raise RuntimeError('One fresh native context required')
        if row_ids is None:
            if mask.device.type != 'cpu':
                raise RuntimeError('Original CPU supervision mask required')
            row_ids = mask.nonzero(as_tuple=False).flatten().to(x.device)
        if row_ids.ndim != 1 or len(row_ids) != label_count or row_ids.device != x.device:
            raise RuntimeError('Supervision coordinate mismatch')
        self.supervised_rows = row_ids
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        self.cache_alive = True
        losses = []
        try:
            selected_labels = labels.index_select(0, row_ids)
            cut = self.prefix(x, adj, core, *args)
            query = cut.detach().requires_grad_(True)
            first = self.model.convs[0]
            directions = (first.lin.weight.detach().to(dtype=torch.float64, copy=True), first.bias.detach().to(dtype=torch.float64, copy=True))
            norm = directions[0].square().sum(1) + directions[1].square()
            cotangent = None
            logit_gradients = []
            head_weights = []
            for _ in range(2):
                query.grad = None
                logits = self.tail(query)
                logits.retain_grad()
                loss = criterion(logits, selected_labels)
                loss.backward()
                logit_gradients.append(logits.grad.detach().clone())
                head_weights.append(self.readout.lin.weight.detach().clone())
                if cotangent is None:
                    cotangent = query.grad.detach().clone()
                else:
                    cotangent.add_(query.grad)
                self._step()
            query.grad = None
            cotangent.mul_(0.5)
            cut.backward(cotangent)
            means = {k: p.grad.detach().clone() for (k, p) in self.body.items()}
            with torch.no_grad():
                local = matmul(self.live_matrix.t(), torch.cat(logit_gradients, dim=1), reduce='sum')
                classes = logit_gradients[0].shape[1]
                mu = []
                for (i, weight) in enumerate(head_weights):
                    local_head = local[:, i * classes:(i + 1) * classes].T @ self.live_hidden
                    mu.append((weight.double() * local_head.double()).sum(0))
                half = (mu[1] - mu[0]) * 0.5
                coefficient = half / torch.where(norm != 0, norm, torch.ones_like(norm))
                coefficient = torch.where(norm != 0, coefficient, torch.zeros_like(coefficient))
                corrections = {'convs.0.lin.weight': (directions[0] * coefficient[:, None]).to(first.lin.weight.dtype), 'convs.0.bias': (directions[1] * coefficient).to(first.bias.dtype)}
            for (index, sign) in enumerate((-1, 1)):
                with torch.no_grad():
                    for (k, p) in self.body.items():
                        p.grad = means[k] + sign * corrections[k]
                    if self.observe is not None:
                        self.observe(index, self, mu, directions, means, corrections)
                self._step()
            self.measurement_blocks += 1
            self.blocks += 1
            self.visits += 2
            self.labels += label_count
            if self.post_commit_enabled:
                self.correct_core(core, args)
            return losses
        finally:
            self.supervised_rows = None
            self.live_hidden = None
            self.live_matrix = None
            self.cache_alive = False
