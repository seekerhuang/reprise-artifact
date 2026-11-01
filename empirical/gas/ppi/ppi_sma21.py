import time
AFTER_UPDATES = 6000
CADENCE = 20
WINDOW = 21

class CausalSMA21:

    def __init__(self, raw, device):
        self.raw = raw
        self.device = device
        self.named = tuple(raw.named_parameters())
        if len(self.named) != 12 or len({id(p) for (_, p) in self.named}) != 12:
            raise RuntimeError('Original PPI12 unique Parameters required')
        if any((str(p.dtype) != 'torch.float32' for (_, p) in self.named)):
            raise RuntimeError('Original FP32 parameters required')
        self.parameter_ids = tuple((id(p) for (_, p) in self.named))
        self.parameter_ptrs = tuple((p.data_ptr() for (_, p) in self.named))
        self.slots = []
        self.snapshot_H = []
        self.mean = None
        self.calls = 0
        self.snapshots = 0
        self.last_H = 0
        self.required_snapshot_seconds = 0.0
        self.last_snapshot_seconds = 0.0

    def _sync(self):
        if self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(self.device)

    def capture(self, H):
        import torch
        if type(H) is not int or H != self.last_H + CADENCE or H > 9000:
            raise RuntimeError('Exact original increasing H20 observation grid required')
        self.calls += 1
        self.last_H = H
        self.last_snapshot_seconds = 0.0
        if H < AFTER_UPDATES:
            return
        self._sync()
        started = time.perf_counter()
        if tuple((id(p) for (_, p) in self.raw.named_parameters())) != self.parameter_ids or tuple((p.data_ptr() for (_, p) in self.raw.named_parameters())) != self.parameter_ptrs:
            raise RuntimeError('Live parameter/storage identity changed')
        with torch.no_grad():
            if len(self.slots) < WINDOW:
                slot = tuple((p.detach().clone() for (_, p) in self.named))
            else:
                slot = self.slots.pop(0)
                self.snapshot_H.pop(0)
                for (target, (_, p)) in zip(slot, self.named):
                    target.copy_(p.detach())
            self.slots.append(slot)
            self.snapshot_H.append(H)
            if self.mean is None:
                self.mean = tuple((torch.empty_like(p) for (_, p) in self.named))
            for (i, target) in enumerate(self.mean):
                target.zero_()
                for current in self.slots:
                    target.add_(current[i])
                target.mul_(1.0 / len(self.slots))
        self._sync()
        self.last_snapshot_seconds = time.perf_counter() - started
        self.required_snapshot_seconds += self.last_snapshot_seconds
        self.snapshots += 1

    def install_on(self, clone):
        if self.mean is None:
            return
        named = tuple(clone.named_parameters())
        if tuple((n for (n, _) in named)) != tuple((n for (n, _) in self.named)):
            raise RuntimeError('Delivery clone named Parameters differ')
        for ((_, p), value) in zip(named, self.mean):
            if p.dtype != value.dtype or p.shape != value.shape or p.data_ptr() in self.parameter_ptrs:
                raise RuntimeError('Averaged delivery must have independent FP32 parameter storage')
            p.copy_(value)
