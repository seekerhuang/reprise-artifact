from types import MethodType
AFTER_UPDATES = 8640
NUMERATOR = 5
DENOMINATOR = 4
ALPHA = NUMERATOR / DENOMINATOR

def _weighted_loss(model, preds, labels, norm_loss):
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(preds, labels, weight=norm_loss.unsqueeze(1), pos_weight=model._r69_late_positive_bce_scalar, reduction='sum')

def policy():
    return {}

class Phase:

    def __init__(self, model):
        if not model.sigmoid_loss or model.optimizer.state:
            raise RuntimeError('Fresh original sigmoid model and empty Adam required')
        if hasattr(model, '_r69_late_positive_bce_scalar'):
            raise RuntimeError('Late positive phase installs once')
        self.model = model
        self.original_loss = model._loss
        self.original_owned = '_loss' in model.__dict__
        self.original_attribute = model.__dict__.get('_loss')
        self.parameter = next(model.parameters())
        self.scalar = self.parameter.new_tensor(ALPHA)
        model._r69_late_positive_bce_scalar = self.scalar
        self.events = []
        self.counts = {'1.0': 0, '1.25': 0}
        self.active = False
        self.closed = False
        self.verified_native_alpha = None

    def before(self, completed):
        if self.closed:
            raise RuntimeError('Closed phase cannot train')
        if completed == AFTER_UPDATES:
            if self.events:
                raise RuntimeError('Repeated positive-loss event')
            current = self.model._loss
            if current.__func__ is not self.original_loss.__func__ or current.__self__ is not self.model:
                raise RuntimeError('Original pre-event loss binding changed')
            self.model._loss = MethodType(_weighted_loss, self.model)
            self.active = True
            self.events.append(dict(after_updates=AFTER_UPDATES, first_update=AFTER_UPDATES + 1, positive_weight=ALPHA))

    def after(self, completed):
        self.counts['1.0' if completed < AFTER_UPDATES else '1.25'] += 1

    def close(self):
        if not self.closed and self.active:
            if self.original_owned:
                self.model._loss = self.original_attribute
            else:
                delattr(self.model, '_loss')
        self.closed = True

def adapt(parent):

    class LatePositive(parent):

        def __init__(self, model, method):
            super().__init__(model, method)
            self.positive_phase = Phase(model)

        def block(self, nodes, adj, weights):
            completed = self.visits
            self.positive_phase.before(completed)
            result = super().block(nodes, adj, weights)
            self.positive_phase.after(completed)
            return result

        def close(self):
            try:
                return super().close()
            finally:
                self.positive_phase.close()
    return LatePositive
