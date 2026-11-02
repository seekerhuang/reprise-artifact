class Direct:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self.visits = 0

    def block(self, x, matrix, core, ids, labels, mask, criterion, *, label_count):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x, matrix, core, ids)
        loss = criterion(logits[mask], labels[mask])
        loss.backward()
        self.optimizer.step()
        self.visits += 1

