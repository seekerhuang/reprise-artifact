from contextlib import contextmanager
import math

def install_fields(engine, pair=(1.0, 0.1), start=80, end=1200):
    pair = tuple((float(x) for x in pair))
    if pair not in ((1.0, 1.0), (1.0, 0.1)):
        raise ValueError('Head pair must be (1.0, 1.0) or (1.0, 0.1)')
    if any((not isinstance(x, int) or x < 0 or x % 2 for x in (start, end))) or start >= end:
        raise ValueError('An increasing pair-aligned interval is required')
    head = tuple(engine.model.convs[-1].parameters())
    head_ids = {id(p) for p in head}
    engine._head_slot_body_parameters = tuple((p for p in engine.model.parameters() if id(p) not in head_ids))
    engine.head_pair = pair
    engine.head_start = start
    engine.head_end = end
    for name in ('first_Head_rate_overrides', 'second_Head_rate_overrides'):
        if name in engine.work:
            raise RuntimeError('Fresh engine rate counters required')
        engine.work[name] = 0

@contextmanager
def head_rate(engine, position):
    if position not in (0, 1):
        raise ValueError('Exactly two original Head slots')
    multiplier = engine.head_pair[position]
    if not engine.head_start <= engine.updates < engine.head_end or multiplier == 1.0:
        yield
        return
    if any((p.grad is not None for p in engine._head_slot_body_parameters)):
        raise RuntimeError('Head rate must not affect an active Body gradient')
    groups = engine.optimizer.param_groups
    previous = tuple((g['lr'] for g in groups))
    if not all((isinstance(lr, float) and math.isfinite(lr) and (lr > 0) for lr in previous)):
        raise RuntimeError('Positive finite native scalar group rates required')
    try:
        for (g, lr) in zip(groups, previous):
            g['lr'] = lr * multiplier
        name = ('first_Head_rate_overrides', 'second_Head_rate_overrides')[position]
        engine.work[name] += 1
        yield
    finally:
        for (g, lr) in zip(groups, previous):
            g['lr'] = lr

class HeadRateMixin:

    def __init__(self, *args, head_pair=(1.0, 0.1), head_start=80, head_end=1200, **kwargs):
        super().__init__(*args, **kwargs)
        install_fields(self, head_pair, head_start, head_end)

    def verify(self):
        clocks = super().verify()
        active = max(0, (min(self.updates, self.head_end) - self.head_start) // 2)
        for (i, name) in enumerate(('first_Head_rate_overrides', 'second_Head_rate_overrides')):
            expected = active if self.head_pair[i] != 1.0 else 0
            if self.work[name] != expected:
                raise RuntimeError('Head slot coverage differs from the declared interval')
        return clocks

    def head_rate_contract(self):
        return dict(Head_slot_multipliers=list(self.head_pair), start_after_completed_updates=self.head_start, end_after_completed_updates=self.head_end, predicate='start <= completed_updates < end', inactive_Head_multipliers=[1.0, 1.0], Body_slot_multipliers=[0.1, 1.9], core_changed=False, ordinary_rate_policy_changed=True, optimizer_state_reset=False, queries_or_updates_added=0, scope='Original two Head queries and native updates; rate changes only while Body gradients are None. Original arithmetic mean and structural recovery unchanged.')
