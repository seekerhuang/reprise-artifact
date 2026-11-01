import copy
import hashlib
import json
import math
from pathlib import Path
VERSION = 'R74-Yelp-primary-alpha125-latecosine-peak010-scale2over3-v1'
AFTER = 8640
TOTAL = 16200
CONTEXTS = 3780
BASE_LATE = 0.0075
HERE = Path(__file__).resolve().parent

def factor(H):
    if type(H) is not int or H % 2 or (not 0 <= H < TOTAL):
        raise ValueError('Original even context-start H0..16198 required')
    return 1.0 if H < AFTER else 2.0 / 3.0 * (1.0 + math.cos(math.pi * ((H - AFTER) // 2 + 0.5) / CONTEXTS))

def specification():
    return {}

class _Mass:

    def __init__(self, model, slots):
        if [len(g['params']) for g in model.optimizer.param_groups] != [22, 2] or slots not in (2, 3):
            raise RuntimeError('Actual native Yelp22+2 groups and fixed S/G slots required')
        (self.model, self.slots) = (model, slots)
        self.calls = self.contexts = self.failed_calls = 0
        self.sum_by_slot = [0.0] * slots
        self.context_H = self.multiplier = self.token = None
        self.first = self.last = None

    def enter(self, H, slot):
        if self.token is not None or H < AFTER or (not 0 <= slot < self.slots):
            raise RuntimeError('Late native rate scopes must not overlap')
        if slot == 0:
            (self.context_H, self.multiplier) = (H, factor(H))
            self.contexts += 1
        elif self.context_H != H:
            raise RuntimeError('All native APIs must share one context factor')
        opt = self.model.optimizer
        before = [g['lr'] for g in opt.param_groups]
        if before != [BASE_LATE] * 2 or opt.defaults['lr'] != BASE_LATE or self.model.lr != BASE_LATE:
            raise RuntimeError('Original8640 global LR event must precede cosine')
        used = [x * self.multiplier for x in before]
        self.token = dict(H_before=H, slot=slot, factor=self.multiplier, restored_group_lrs=before, restored_default_lr=opt.defaults['lr'], restored_model_lr=self.model.lr, used_group_lrs=used, used_default_lr=BASE_LATE * self.multiplier, used_model_lr=BASE_LATE * self.multiplier)
        try:
            for (group, lr) in zip(opt.param_groups, used):
                group['lr'] = lr
            opt.defaults['lr'] = self.token['used_default_lr']
            self.model.lr = self.token['used_model_lr']
        except BaseException:
            self.finish(False)
            raise

    def finish(self, success):
        if self.token is None:
            return
        record = self.token
        opt = self.model.optimizer
        try:
            if not success:
                self.failed_calls += 1
                return
            if [g['lr'] for g in opt.param_groups] != record['used_group_lrs'] or opt.defaults['lr'] != record['used_default_lr'] or self.model.lr != record['used_model_lr']:
                raise RuntimeError('Actual used native rates changed within cosine scope')
            self.calls += 1
            self.sum_by_slot[record['slot']] += record['factor']
            if self.first is None:
                self.first = copy.deepcopy(record)
            self.last = copy.deepcopy(record)
        finally:
            for (group, lr) in zip(opt.param_groups, record['restored_group_lrs']):
                group['lr'] = lr
            opt.defaults['lr'] = record['restored_default_lr']
            self.model.lr = record['restored_model_lr']
            self.token = None

def adapt_S(parent):

    class SharedLateMassS(parent):

        def __init__(self, model, method):
            super().__init__(model, method)
            if method != 'lateA':
                raise ValueError('Original S only')
            self._late_mass = _Mass(model, 3)

        def _step(self):
            if self.visits < AFTER:
                return super()._step()
            self._late_mass.enter(self.visits, self.optimizer_calls % 3)
            success = False
            try:
                result = super()._step()
                success = True
                return result
            finally:
                self._late_mass.finish(success)
    return SharedLateMassS

def verify_finished(model, engine):
    result = engine._late_mass.verify()
    if engine.visits != TOTAL or [len(g['params']) for g in model.optimizer.param_groups] != [22, 2]:
        raise RuntimeError('Original complete parameter/group budget required')
    return result
