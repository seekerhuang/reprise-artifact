from pathlib import Path
import importlib.util, time

ROOT = Path(__file__).resolve().parent
ROUTE_PATH = Path(__file__).resolve().parent / 'yelp_sparse_bmm_route_v1.py'

_spec = importlib.util.spec_from_file_location('T456_original_public_Yelp_bmm', ROUTE_PATH)
ROUTE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ROUTE)
configure_before_torch = ROUTE.configure_before_torch

def specification():
    return dict(version='T456-Yelp-shared-TRAIN-strict-public-bmm-v1', route=ROUTE.specification(),
                planned_epochs=75, planned_update_budget=16200,
                scope='Shared S/D/G TRAIN and close backend only. Original native evaluation/archives/readers execute after instance-method and deterministic-flag restoration. No adjacency cache, coalescing shortcut, core or metric change.',
                timing='Complete synchronized outer TRAIN scope replaces the original nested epoch seconds once. Required helper import/setup, instance installation/restoration, public sparse representation/conversion/forward/backward and close remain paid.')

def _state(torch):
    return torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()

def adapt(parent):
    class StrictYelp(parent):
        def __init__(self, model, method):
            super().__init__(model, method)
            if method not in ['lateA', 'D', 'G']:
                raise ValueError('Original three methods only')
            self._strict_method = method
            self._strict_epochs = 0
            self._strict_counts = dict.fromkeys(['0', '1', '2'], 0)
            self._strict_live = False
            self._strict_close_calls = 0
            self._strict_outer_seconds = 0.

        def work(self):
            result = super().work()
            result['TRAIN_sparse_backend'] = dict(specification(), actual_epochs=self._strict_epochs,
                actual_forward_counts=dict(self._strict_counts), complete_outer_epoch_seconds=self._strict_outer_seconds,
                original_nested_epoch_time_not_added_or_refunded=True,
                close_calls=self._strict_close_calls, scope_live=self._strict_live)
            return result

        def verify_final(self, steps):
            result = super().verify_final(steps)
            samples = 16200 if self._strict_method == 'D' else 8100
            if (steps != 16200 or self._strict_epochs != 75 or self._strict_live
                    or self._strict_counts != {'0': samples, '1': samples, '2': 0}
                    or self._strict_close_calls != 0):
                raise RuntimeError('Exact original75epochs/input counts/close and restored strict backend required')
            return result

        def close(self):
            import torch
            sync = lambda: torch.cuda.synchronize(next(self.model.parameters()).device)
            before = _state(torch)
            sync()
            self._strict_live = True
            try:
                with ROUTE.strict_determinism(torch):
                    with ROUTE.install(self.model) as counts:
                        result = super().close()
            finally:
                self._strict_live = False
                sync()
            if _state(torch) != before or any(counts.values()):
                raise RuntimeError('Close must restore flags without an extra feature SpMM')
            self._strict_close_calls += 1
            if self._strict_close_calls != 1:
                raise RuntimeError('Original engine close must execute exactly once')
            return result
    return StrictYelp

def train_epoch(original, minibatch, engine, warm_initial, counters, audit, synchronize,
                before_epoch=None, clock=time.perf_counter):
    import torch
    started = clock()
    before = _state(torch)
    visits = engine.visits
    if engine._strict_live:
        raise RuntimeError('No nested strict TRAIN scope')
    engine._strict_live = True
    try:
        synchronize()
        with ROUTE.strict_determinism(torch):
            with ROUTE.install(engine.model) as counts:
                if before_epoch is None:
                    original(minibatch, engine, warm_initial, counters, audit, synchronize)
                else:
                    original(minibatch, engine, warm_initial, counters, audit, synchronize, before_epoch=before_epoch)
    finally:
        engine._strict_live = False
        synchronize()
    samples = 216 if engine._strict_method == 'D' else 108
    if engine.visits - visits != 216 or counts != {'0': samples, '1': samples, '2': 0} or _state(torch) != before:
        raise RuntimeError('Original216updates/feature-call budget and flag restoration required')
    engine._strict_epochs += 1
    for key, value in counts.items():
        engine._strict_counts[key] += value
    outer = clock() - started
    engine._strict_outer_seconds += outer
    return outer
