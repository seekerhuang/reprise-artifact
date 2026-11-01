from collections import deque
VERSION = 'R44-causal-Head-only-SMA5-after800-native-H4-v1'
HEAD_NAMES = ('convs.1.bias', 'convs.1.lin.weight')
AFTER_UPDATES = 800
CADENCE = 4
WINDOW = 5
TOTAL_UPDATES = 1600

class HeadAverage:

    def __init__(self, model, engine, optimizer):
        import torch
        if model.batch_norm or model.residual or model.linear or model.drop_input or (model.num_layers != 2) or (model.in_channels != 602) or (model.out_channels != 41) or (len(model.histories) != 1) or (model.dropout != 0.5):
            raise RuntimeError('Only the actual original R44 L2H256/noBN/nores/drop.5 model is supported')
        named = dict(model.named_parameters())
        if set(engine.head) != set(HEAD_NAMES) or len(engine.body) != 2 or len(engine.unused) != 4:
            raise RuntimeError('Original four active conv parameters and four unused BN parameters required')
        self.parameters = tuple((named[name] for name in HEAD_NAMES))
        if any((engine.head[name] is not named[name] for name in HEAD_NAMES)):
            raise RuntimeError('Original engine/Head Parameter owners differ')
        if [tuple(param.shape) for param in self.parameters] != [(41,), (41, 256)]:
            raise RuntimeError('Original final GCN classifier dimensions required')
        if any((param.dtype is not torch.float32 for param in self.parameters)):
            raise RuntimeError('Original FP32 classifier required')
        if engine.optimizer is not optimizer:
            raise RuntimeError('Original model optimizer owner required')
        (self.model, self.engine, self.optimizer) = (model, engine, optimizer)
        self.contract = tuple(((id(param), param.data_ptr(), tuple(param.shape), tuple(param.stride()), param.dtype, param.device, param.requires_grad) for param in self.parameters))
        self.history = deque(maxlen=WINDOW)
        self.snapshot_H = []
        self.observations = 0
        self.last_H = 0
        self.active = False
        self.installed = False
        self._raw = None
        self._mean = None
        self._last_mean = None
        self._last_delivery_H = None
        self.snapshots = self.mean_initial_clones = self.mean_adds = self.mean_divides = 0
        self.installations = self.restorations = 0
        self.actual_parameter_copies = 0

    def _check(self):
        current = dict(self.model.named_parameters())
        if any((current[name] is not param for (name, param) in zip(HEAD_NAMES, self.parameters))):
            raise RuntimeError('Live classifier Parameter owners changed')
        actual = tuple(((id(param), param.data_ptr(), tuple(param.shape), tuple(param.stride()), param.dtype, param.device, param.requires_grad) for param in self.parameters))
        if actual != self.contract:
            raise RuntimeError('Classifier storage/layout/dtype/device contract changed')
        if self.engine.cache_alive or self.engine.supervised_rows is not None:
            raise RuntimeError('Delivery is allowed only after the complete Body2/context')

    def begin(self, completed_updates):
        import torch
        self._check()
        if self.active or completed_updates != self.last_H + CADENCE or completed_updates > TOTAL_UPDATES:
            raise RuntimeError('Exactly the original chronological H4 delivery grid is required')
        if self.engine.visits != completed_updates:
            raise RuntimeError('Actual completed native Head/Body clock differs')
        self.active = True
        self.last_H = completed_updates
        self.observations += 1
        if completed_updates < AFTER_UPDATES:
            return
        with torch.no_grad():
            snapshot = tuple((param.detach().clone() for param in self.parameters))
            self.history.append((completed_updates, snapshot))
            self.snapshot_H.append(completed_updates)
            self.snapshots += len(snapshot)
            mean = tuple((value.clone() for value in self.history[0][1]))
            self.mean_initial_clones += len(mean)
            for (_, values) in list(self.history)[1:]:
                for (result, value) in zip(mean, values):
                    result.add_(value)
                    self.mean_adds += 1
            if len(self.history) > 1:
                for result in mean:
                    result.div_(len(self.history))
                    self.mean_divides += 1
            for (param, value) in zip(self.parameters, mean):
                param.copy_(value)
                self.actual_parameter_copies += 1
        self._raw = snapshot
        self._mean = mean
        self._last_mean = mean
        self._last_delivery_H = completed_updates
        self.installed = True
        self.installations += 1

    def end(self):
        import torch
        if not self.active:
            raise RuntimeError('No active delivery to restore')
        self._check()
        if self.installed:
            with torch.no_grad():
                for (param, value) in zip(self.parameters, self._raw):
                    param.copy_(value)
                    self.actual_parameter_copies += 1
            self.restorations += 1
        self.active = self.installed = False
        self._raw = self._mean = None
