import copy
import math
AFTER_UPDATES = 800
LATE_CONTEXTS = 200

def factor(visits):
    if type(visits) is not int or not 0 <= visits < 1200 or visits % 2:
        raise ValueError('Original even context-start H in [0,1198] required')
    return 1.0 if visits < AFTER_UPDATES else 1.0 + math.cos(math.pi * ((visits - AFTER_UPDATES) // 2 + 0.5) / LATE_CONTEXTS)

def adapt(Base):

    class LateMassRedistribution(Base):

        def __init__(self, model, method):
            super().__init__(model, method)
            self._late_mass_calls = self._late_mass_contexts = 0
            self._late_mass_first = self._late_mass_last = None
            self._late_mass_live = False
            self._late_mass_sum_by_slot = [0.0, 0.0, 0.0]
            self._late_mass_H = self._late_mass_factor = None

        def _step(self):
            if self.visits < AFTER_UPDATES:
                return super()._step()
            optimizer = self.model.optimizer
            if optimizer.defaults['lr'] != 0.0025 or len(optimizer.param_groups) != 3:
                raise RuntimeError('Original H800 global event/three groups required')
            slot = self.optimizer_calls % 3
            if slot == 0:
                self._late_mass_H = self.visits
                self._late_mass_factor = factor(self.visits)
                self._late_mass_contexts += 1
            elif self._late_mass_H != self.visits:
                raise RuntimeError('All three original APIs must share one context factor')
            multiplier = self._late_mass_factor
            before = [g['lr'] for g in optimizer.param_groups]
            expected = [0.0025, 0.0025, 0.000625 if slot == 0 else 0.0025]
            if before != expected:
                raise RuntimeError('Cosine must be inside derived Head1-quarter wrapper')
            self._late_mass_live = True
            try:
                for (group, lr) in zip(optimizer.param_groups, before):
                    group['lr'] = lr * multiplier
                result = super()._step()
            finally:
                for (group, lr) in zip(optimizer.param_groups, before):
                    group['lr'] = lr
                self._late_mass_live = False
            self._late_mass_calls += 1
            self._late_mass_sum_by_slot[slot] += multiplier
            event = dict(H_before=self.visits, slot=slot, factor=multiplier, actual_lrs=[v * multiplier for v in before])
            if self._late_mass_first is None:
                self._late_mass_first = event
            self._late_mass_last = event
            return result
    return LateMassRedistribution
