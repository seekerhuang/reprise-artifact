class Training:

    def _step(self):
        self.model.optimizer.step()
        self.optimizer_calls += 1
        self.model.optimizer.zero_grad(set_to_none=True)

    def close(self):
        self.randomness.close()
        self.closed = True

    def _tail(self, boundary, adj):
        import torch.nn.functional as F
        m = self.model
        if self.method == 'earlyA':
            (_, hidden) = m.aggregators[2]((adj, boundary))
            boundary = m.classifier.f_dropout(F.normalize(hidden, p=2, dim=1))
        return m.classifier.f_lin[0](boundary) + m.classifier.f_bias[0]
