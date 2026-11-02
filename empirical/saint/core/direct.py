import torch

from empirical.saint.schedules import late_cosine, positive_phase, pulse_schedule, reddit_late_cosine


class Direct:
    def __init__(self, model):
        self.model = model
        groups = ({id(p) for layer in model.aggregators[:-1] for p in layer.parameters()},
                  {id(p) for p in model.aggregators[-1].parameters()},
                  {id(model.classifier.f_lin[0].weight), id(model.classifier.f_bias[0])})
        self.clip_groups = tuple(tuple(p for p in model.parameters() if id(p) in group) for group in groups)
        self.visits = self.optimizer_calls = 0
        self.positive_phase = positive_phase.Phase(model) if model.sigmoid_loss else None

    def block(self, nodes, matrix, weights):
        model, optimizer = self.model, self.model.optimizer
        completed = self.visits
        if self.positive_phase is not None:
            self.positive_phase.before(completed)
            rate = pulse_schedule.next_block_rate(completed)
            multiplier = late_cosine.factor(completed - completed % 2)
        else:
            rate = 0.01 if completed < 800 else 0.0025
            multiplier = reddit_late_cosine.factor(completed - completed % 2)
        model.lr = optimizer.defaults['lr'] = rate
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = rate * multiplier
            if self.positive_phase is None and completed >= 800 and completed % 2 == 0 and index == 2:
                group['lr'] *= 0.25
        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions, _, labels = model(nodes, matrix)
            loss = model._loss(predictions, labels, weights)
            loss.backward()
            for group in self.clip_groups:
                torch.nn.utils.clip_grad_norm_(group, 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        finally:
            for group in optimizer.param_groups:
                group['lr'] = rate
        self.visits += 1
        self.optimizer_calls += 1
        return [loss.detach()]

    def close(self):
        if self.positive_phase is not None:
            self.positive_phase.close()

