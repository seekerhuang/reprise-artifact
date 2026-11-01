from collections import deque
START = 800
GRID = 4
TOTAL = 1600

def selected_named_parameters(model):
    named = tuple(model.named_parameters())
    readout = model.convs[-1]
    selected = tuple((p for layer in list(readout.post_lins) + [readout.lin] for p in layer.parameters()))
    ids = {id(p) for p in selected}
    if len(named) != 78 or len(model.state_dict()) != 87 or len(readout.post_lins) != 4 or (len(selected) != 10) or (len(ids) != 10):
        raise RuntimeError('Original PNA78/87 and actual structural Head10 required')
    head = tuple(((n, p) for (n, p) in named if id(p) in ids))
    expected = {f'convs.3.post_lins.{i}.{k}' for i in range(4) for k in ('weight', 'bias')} | {'convs.3.lin.weight', 'convs.3.lin.bias'}
    if len(head) != 10 or {n for (n, _) in head} != expected:
        raise RuntimeError('Only original final post-linears plus self-linear Head10 may be averaged')
    return head

class Window:

    def __init__(self, model, *, expected_parameter_count=10, expected_elements=679):
        named = selected_named_parameters(model)
        if len(named) != expected_parameter_count or sum((p.numel() for (_, p) in named)) != expected_elements or len({id(p) for (_, p) in named}) != len(named) or any((str(p.dtype) != 'torch.float32' for (_, p) in named)):
            raise RuntimeError('Declared independent FP32 parameter support required')
        self.names = tuple((n for (n, _) in named))
        self.raw_parameters = tuple((p for (_, p) in named))
        self.shapes = tuple((tuple(p.shape) for (_, p) in named))
        self.window = deque(maxlen=3)
        self.last_H = 0
        self.capture_calls = self.snapshots = self.delivery_calls = self.averaged_deliveries = 0
        self.last_delivery_H = None
        self.last_snapshot_Hs = []

    def capture(self, model, H):
        if type(H) is not int or H != self.last_H + GRID or H > TOTAL:
            raise RuntimeError('Every original H4 interval must capture exactly once')
        named = selected_named_parameters(model)
        if tuple((n for (n, _) in named)) != self.names or any((p is not old for ((_, p), old) in zip(named, self.raw_parameters))):
            raise RuntimeError('The raw parameter objects must remain unchanged')
        if H >= START:
            snap = tuple((p.detach().clone() for (_, p) in named))
            self.window.append((H, snap))
            self.snapshots += 1
        self.last_H = H
        self.capture_calls += 1

    def install(self, clone):
        import torch
        if self.last_H < GRID or self.last_delivery_H == self.last_H:
            raise RuntimeError('One current delivery per completed native interval')
        named = selected_named_parameters(clone)
        if tuple((n for (n, _) in named)) != self.names or tuple((tuple(p.shape) for (_, p) in named)) != self.shapes or any((str(p.dtype) != 'torch.float32' for (_, p) in named)) or any((p is raw or p.data_ptr() == raw.data_ptr() for ((_, p), raw) in zip(named, self.raw_parameters))):
            raise RuntimeError('Independent same-schema FP32 eval clone required')
        if self.window:
            with torch.no_grad():
                for (index, (_, target)) in enumerate(named):
                    target.zero_()
                    for (_, snapshot) in self.window:
                        source = snapshot[index]
                        if source.data_ptr() == target.data_ptr():
                            raise RuntimeError('Snapshot storage must not alias eval clone')
                        target.add_(source)
                    target.div_(len(self.window))
            self.averaged_deliveries += 1
        self.last_delivery_H = self.last_H
        self.delivery_calls += 1
        self.last_snapshot_Hs = [h for (h, _) in self.window]
