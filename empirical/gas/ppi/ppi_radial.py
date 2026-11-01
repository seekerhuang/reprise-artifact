from contextlib import contextmanager

def contract(metric):
    if metric != 'euclidean':
        raise ValueError(metric)
    return {'metric': metric}
NAMES = ('bns.1.weight', 'bns.1.bias')
STAT_NAMES = ('measurement_difference_norm', 'correction_norm', 'mean_S_norm', 'mean_constraint_residual_norm', 'zero_direction_channels')

class RadialFeedback:

    def __init__(self, model, metric='euclidean'):
        self.spec = contract(metric)
        if model.num_layers != 2 or not model.linear or (not model.residual) or (not model.batch_norm) or (model.dropout != 0.0):
            raise ValueError('Require frozen PPI L2/linear/residual/BN/drop0')
        self.model = model
        self.parameters = tuple((dict(model.named_parameters())[k] for k in NAMES))
        self.bn = model.bns[1]
        if not self.bn.affine or not self.bn.track_running_stats or any((a is not b for (a, b) in zip(self.parameters, (self.bn.weight, self.bn.bias)))):
            raise ValueError('S must be the actual final BN affine parameters')
        self.contexts = self.measurement_calls = self.body_submissions = 0
        self.captures = self.pending = 0
        self.u = self.feature = self.queries = self.n = self.r = self.mean = None
        self.sums = self.maxima = None
        self.window_contexts = 0
        self.capture_payload_peak_bytes = 0

    @contextmanager
    def capture(self):
        if self.u is not None or self.feature is not None or self.pending:
            raise RuntimeError('Previous context was not released')
        seen = [0]

        def hook(_module, _args, output):
            if seen[0]:
                raise RuntimeError('Exactly one final BN forward is required')
            self.u = output.detach().clone()
            seen[0] += 1
        handle = self.bn.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
        if seen != [1]:
            self.u = None
            raise RuntimeError('Missing final BN forward')
        self.captures += 1

    def prepare(self, cut):
        import torch
        if self.u is None or self.u.shape != cut.shape or cut.ndim != 2:
            raise RuntimeError('Final BN population and affine Head cut differ')
        self.feature = torch.where(cut.detach() > 0, self.u, torch.zeros((), dtype=self.u.dtype, device=self.u.device)).double()
        self.u = None
        self.queries = []
        self.pending = 1

    def measure(self, raw_lambda):
        if self.pending != 1 or len(self.queries) >= 2 or raw_lambda.shape != self.feature.shape:
            raise RuntimeError('Two pre-Head-step raw cotangents are required')
        self.queries.append((self.feature * raw_lambda.detach()).sum(0))
        self.measurement_calls += 1

    def _inverse_metric(self, optimizer):
        import torch
        if self.spec['metric'] == 'euclidean':
            return torch.ones_like(self.n)
        groups = {id(p): g for g in optimizer.param_groups for p in g['params']}
        scales = []
        for p in self.parameters:
            group = groups[id(p)]
            if group.get('amsgrad', False) or group.get('weight_decay', 0.0) != 0.0:
                raise ValueError('Incoming-scale option covers original no-WD/no-AMSGrad PPI Adam')
            state = optimizer.state.get(p, {})
            if not state:
                corrected_v = torch.zeros_like(p, dtype=torch.float64)
            else:
                step = state['step'].detach().to(device=p.device, dtype=torch.float64)
                correction = 1.0 - torch.pow(torch.as_tensor(group['betas'][1], device=p.device, dtype=torch.float64), step)
                corrected_v = state['exp_avg_sq'].detach().double() / correction
            scales.append((corrected_v.sqrt() + group['eps']).square())
        return torch.stack(scales)

    def complete_measurement(self, optimizer):
        import torch
        if self.pending != 1 or len(self.queries) != 2:
            raise RuntimeError('Missing paid Head measurement')
        self.n = torch.stack([p.detach().double().clone() for p in self.parameters])
        self.mean = torch.stack([p.grad.detach().double().clone() for p in self.parameters])
        inverse_metric = self._inverse_metric(optimizer)
        normal = inverse_metric * self.n
        denominator = (normal * self.n).sum(0)
        live = denominator != 0
        safe = torch.where(live, denominator, torch.ones_like(denominator))
        half_difference = (self.queries[1] - self.queries[0]) * 0.5
        coefficient = torch.where(live, half_difference / safe, torch.zeros_like(safe))
        self.r = normal * coefficient.unsqueeze(0)
        self.feature = None
        self.pending = 2

    def install(self, position):
        if self.pending != 2 or position not in (0, 1):
            raise RuntimeError('Mean VJP and complete measurement must precede both Body steps')
        sign = -1.0 if position == 0 else 1.0
        for (index, p) in enumerate(self.parameters):
            p.grad = (self.mean[index] + sign * self.r[index]).to(p.dtype)
        self.body_submissions += 1

    def finish(self):
        if self.pending != 2 or self.body_submissions != 2 * (self.contexts + 1):
            raise RuntimeError('Require exactly Head2/Body2')
        self.contexts += 1
        self.window_contexts += 1
        self.pending = 0
        self.feature = self.u = self.queries = self.n = self.mean = self.r = None
