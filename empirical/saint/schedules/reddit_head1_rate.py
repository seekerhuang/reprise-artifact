import copy
from empirical.saint.schedules import reddit_terminal_groups as terminal_groups
AFTER_UPDATES = 800
FACTOR = 0.25

def adapt(Base):

    class HeadFirstRate(Base):

        def __init__(self, model, method):
            super().__init__(model, method)
            record = terminal_groups.record(model, model.optimizer)
            if record['actual_group_sizes'] != [22, 2, 4]:
                raise RuntimeError('Three native groups required')
            self._head_group = model.optimizer.param_groups[2]
            if {id(p) for p in self._head_group['params']} != {id(p) for p in self.head.values()} | {id(p) for p in self.unused.values()}:
                raise RuntimeError('Head group must be actual classifier active and unused parameters')
            self._head_rate_counts = dict(Head1_global=0, Head1_quarter=0, Head2_global=0, Body2_global=0)
            self._head_rate_first = None
            self._head_rate_last = None
            self._head_rate_override_live = False

        def _step(self):
            optimizer = self.model.optimizer
            slot = self.optimizer_calls % 3
            if len(optimizer.param_groups) != 3 or optimizer.param_groups[2] is not self._head_group:
                raise RuntimeError('Native group objects changed')
            base_lr = optimizer.defaults['lr']
            if any((g['lr'] != base_lr for g in optimizer.param_groups)):
                raise RuntimeError('Expected restored global LR before native API')
            active = slot == 0 and self.visits >= AFTER_UPDATES
            if not active:
                result = super()._step()
                self._head_rate_counts[('Head1_global', 'Head2_global', 'Body2_global')[slot]] += 1
                return result
            phase = getattr(optimizer, '_phase2_declared_rate')
            if base_lr != 0.0025 or len(phase['transitions']) != 1 or phase['transitions'][0]['after_completed_updates'] != 800:
                raise RuntimeError('Original H800 rate transition must precede Head1 override')
            old_lr = self._head_group['lr']
            self._head_group['lr'] = old_lr * FACTOR
            self._head_rate_override_live = True
            try:
                result = super()._step()
            finally:
                self._head_group['lr'] = old_lr
                self._head_rate_override_live = False
            self._head_rate_counts['Head1_quarter'] += 1
            event = dict(after_completed_updates=self.visits, native_API_slot=0, global_lr=base_lr, actual_Head1_lr=old_lr * FACTOR, restored_Head_lr=self._head_group['lr'])
            if self._head_rate_first is None:
                self._head_rate_first = event
            self._head_rate_last = event
            return result
    return HeadFirstRate
