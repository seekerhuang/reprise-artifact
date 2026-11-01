SWITCH_AFTER = 5184
INITIAL_LR = 0.015
MIDDLE_LR = 0.01
MILD_AFTER = 8640
LATE_LR = 0.0075
TOTAL_UPDATES = 16200

def policy():
    return {}

def next_block_rate(completed_updates):
    return INITIAL_LR if completed_updates < SWITCH_AFTER else MIDDLE_LR if completed_updates < MILD_AFTER else LATE_LR

def adapt(parent):

    class Early15Mild40Structural(parent):

        def __init__(self, model, method):
            super().__init__(model, method)
            if method != 'lateA' or self.visits != 0:
                raise ValueError('Fresh original structural A required')
            if model.lr != INITIAL_LR or model.optimizer.defaults['lr'] != INITIAL_LR or any((group['lr'] != INITIAL_LR for group in model.optimizer.param_groups)):
                raise ValueError('Declared early pulse initial .015 required')
            self.lr_events = []
            self.lr_context_counts = {'0.015': 0, '0.01': 0, '0.0075': 0}

        def block(self, nodes, adj, weights):
            completed = self.visits
            if completed in (SWITCH_AFTER, MILD_AFTER):
                rate = next_block_rate(completed)
                for group in self.model.optimizer.param_groups:
                    group['lr'] = rate
                self.model.optimizer.defaults['lr'] = rate
                self.model.lr = rate
                self.lr_events.append(dict(after_updates=completed, first_update=completed + 1, lr=rate))
            result = super().block(nodes, adj, weights)
            self.lr_context_counts[str(next_block_rate(completed))] += 1
            return result
    return Early15Mild40Structural
