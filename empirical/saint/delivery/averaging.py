from contextlib import contextmanager
import time
AFTER_UPDATES = 10800
CADENCE = 216
WINDOW = 5

class BaseAverage:

    def __init__(self, model, *, after_updates=AFTER_UPDATES, cadence=CADENCE, window=WINDOW):
        import torch
        self.after_updates, self.cadence, self.window = after_updates, cadence, window
        self.torch = torch
        self.model = model
        if list(model.named_buffers()):
            raise ValueError('Parameter averaging requires no registered buffers')
        self.named = tuple(model.named_parameters())
        if not self.named or len({id(p) for (_, p) in self.named}) != len(self.named):
            raise ValueError('Unique original Parameter support required')
        if any((p.dtype != torch.float32 for (_, p) in self.named)):
            raise ValueError('Only the declared FP32 parameter arithmetic is supported')
        self.identity = tuple(((name, id(p), p.data_ptr(), tuple(p.shape)) for (name, p) in self.named))
        self.history = []
        self.average = None
        self.last_H = 0
        self.observations = []
        self.completed = 0
        self.failures = 0
        self.installs = self.restores = self.snapshots = 0
        self.is_installed = False
        self.required_seconds = 0.0
        self.total_snapshot_payload_bytes = 0
        self.peak_owned_payload_bytes = 0

    def _verify_identity(self):
        actual = tuple(((name, id(p), p.data_ptr(), tuple(p.shape)) for (name, p) in self.model.named_parameters()))
        if actual != self.identity or list(self.model.named_buffers()):
            raise RuntimeError('Original Parameter identity/storage/support or empty buffer domain changed')

    def _prepare(self, H):
        if self.is_installed:
            raise RuntimeError('Averaged parameters were not restored')
        if not isinstance(H, int) or H != self.last_H + self.cadence:
            raise ValueError('Each original20-visit observation must be consumed once in order')
        self._verify_identity()
        self.last_H = H
        self.observations.append(H)
        if H < self.after_updates:
            return
        with self.torch.no_grad():
            snapshot = tuple((p.detach().clone() for (_, p) in self.named))
            self.history.append((H, snapshot))
            if len(self.history) > self.window:
                self.history.pop(0)
            self.snapshots += 1
            if self.average is None:
                self.average = tuple((self.torch.empty_like(p) for (_, p) in self.named))
            for (index, value) in enumerate(self.average):
                value.copy_(self.history[0][1][index])
                for (_, earlier) in self.history[1:]:
                    value.add_(earlier[index])
                if len(self.history) > 1:
                    value.div_(len(self.history))

    def _install(self):
        if self.last_H < self.after_updates:
            return
        self.is_installed = True
        with self.torch.no_grad():
            for ((_, parameter), value) in zip(self.named, self.average):
                parameter.copy_(value)
        self.installs += 1

    def _restore(self):
        if not self.is_installed:
            return
        with self.torch.no_grad():
            for ((_, parameter), value) in zip(self.named, self.history[-1][1]):
                parameter.copy_(value)
        self.is_installed = False
        self.restores += 1

class CausalAverage(BaseAverage):

    def __init__(self, model, *, after_updates=AFTER_UPDATES, cadence=CADENCE, window=WINDOW):
        super().__init__(model, after_updates=after_updates, cadence=cadence, window=window)
        self.restoration_seconds_SCI = 0.0
        self.inactive_observation_seconds_SCI = 0.0
        self._identity_prechecked = False

    def _verify_identity(self):
        if not self._identity_prechecked:
            return super()._verify_identity()

    @contextmanager
    def installed(self, H, synchronize, clock=time.perf_counter):
        self._verify_identity()
        self._identity_prechecked = True
        synchronize()
        started = clock()
        try:
            self._prepare(H)
            self._install()
            synchronize()
        except BaseException:
            self.failures += 1
            try:
                self._restore()
                synchronize()
            finally:
                self._identity_prechecked = False
            raise
        finally:
            elapsed = clock() - started
            if H >= self.after_updates:
                self.required_seconds += elapsed
            else:
                self.inactive_observation_seconds_SCI += elapsed
        self._identity_prechecked = False
        succeeded = False
        try:
            yield None
            succeeded = True
        finally:
            started = clock()
            try:
                synchronize()
                self._restore()
                synchronize()
            finally:
                self.restoration_seconds_SCI += clock() - started
                self.completed += int(succeeded)
                self.failures += int(not succeeded)
