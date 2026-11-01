from dataclasses import dataclass
CLIP_NORM = 5.0
STAT_NAMES = ('tau', 'tau_below_one', 'raw_r_norm', 'applied_r_norm', 'native_L_clip_coefficient', 'native_mean_L_norm', 'postcast_pair_max_L_norm', 'postcast_selected_mean_error_norm', 'native_mean_squared_cap_excess', 'native_mean_outside_cap', 'clamped_normalization_rows', 'raw_measurement_difference_norm', 'zero_direction_channels')
UNCOMPUTED_STATISTIC = 'scaled_mean_constraint_residual_norm'

def euclidean(n, difference_half):
    import torch
    square = n.square().sum(0)
    nonzero = square != 0
    factor = torch.where(nonzero, difference_half / torch.where(nonzero, square, torch.ones_like(square)), torch.zeros_like(square))
    return (n * factor.unsqueeze(0), None)

def maximal_tau(full_q_square, q_selected, r, cap=CLIP_NORM):
    import torch
    a = r.square().sum()
    b = (q_selected * r).sum().abs()
    raw_slack = cap * cap - full_q_square
    slack = raw_slack.clamp_min(0.0)
    denominator = (b.square() + a * slack).sqrt() + b
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    candidate = (slack / safe).clamp(max=1.0)
    tau = torch.where(a == 0, torch.ones_like(a), torch.where(raw_slack > 0, candidate, torch.zeros_like(a)))
    return (tau, None, None)

@dataclass(frozen=True)
class DecodeResult:
    pair: tuple
    values: object
    tau: object
    payload: tuple

def _tensor_decode_half(selected, late_gradients, selected_positions, difference_half, coefficient, clamped_count):
    import torch
    n = torch.stack([p.detach().double() for p in selected])
    q = torch.stack([late_gradients[j].detach().double() for j in selected_positions])
    (raw_r, zero) = euclidean(n, difference_half)
    r = raw_r * coefficient.double()
    full_q_square = torch.stack([g.detach().double().square().sum() for g in late_gradients]).sum()
    (tau, excess, outside) = maximal_tau(full_q_square, q, r)
    pair = tuple((tuple(((q[j] + sign * tau * r[j]).to(p.dtype) for (j, p) in enumerate(selected))) for sign in (-1.0, 1.0)))
    return DecodeResult(pair, None, tau, (n, q, raw_r, r))

class GraphDecode:

    def __init__(self, *, selected, late_parameters, warmup_iterations=3):
        import torch
        self.selected = tuple(selected)
        self.late_parameters = tuple(late_parameters)
        self.warmup_iterations = warmup_iterations
        if type(warmup_iterations) is not int or warmup_iterations < 1:
            raise ValueError('Positive integer warmup_iterations required')
        if len(self.selected) != 2 or len(self.late_parameters) != 4:
            raise ValueError('Locked Reddit gamma/beta and all four late parameters required')
        if len({id(p) for p in self.late_parameters}) != 4 or len({id(p) for p in self.selected}) != 2:
            raise ValueError('Unique original parameter objects required')
        self.device = self.selected[0].device
        self.dtype = self.selected[0].dtype
        if self.device.type != 'cuda' or self.dtype not in (torch.float32, torch.float64):
            raise ValueError('CUDA native FP32/FP64 inputs required; no CPU fallback')
        if any((p.device != self.device or p.dtype != self.dtype for p in self.late_parameters)):
            raise ValueError('One original device and dtype required')
        if any((tuple(p.shape) != (128,) for p in self.selected)):
            raise ValueError('Only the original 128-channel selected affine pair')
        if sorted((tuple(p.shape) for p in self.late_parameters)) != [(128,), (128,), (128,), (128, 256)]:
            raise ValueError('Require original W[128,256], bias/offset/scale[128]')
        ids = [id(p) for p in self.late_parameters]
        if any((id(p) not in ids for p in self.selected)):
            raise ValueError('Selected parameters must be original late parameter identities')
        self.selected_positions = tuple((ids.index(id(p)) for p in self.selected))
        self.parameter_storage = tuple(((p.data_ptr(), tuple(p.shape), tuple(p.stride())) for p in self.late_parameters))
        self._gradients = self._half = self._coefficient = self._clamped_count = None
        self._capture_stream = self._graph = self._output = None
        self._stream_id = None
        self._closed = self._failed = False
        self._calls = self._capture_count = self._warmups = self._staging_copies = 0

    def _validate_inputs(self, difference_half, coefficient, clamped_count):
        import torch
        if self._closed or self._failed:
            raise RuntimeError('GraphDecode is closed or a prior CUDA call failed')
        if difference_half.device != self.device or difference_half.dtype != torch.float64 or tuple(difference_half.shape) != (128,):
            raise ValueError('Actual E591 single-contraction FP64[128] half difference required')
        if coefficient.device != self.device or coefficient.dtype != self.dtype or coefficient.numel() != 1 or (coefficient.ndim != 0):
            raise ValueError('Original scalar native L clip coefficient required')
        now = tuple(((p.data_ptr(), tuple(p.shape), tuple(p.stride())) for p in self.late_parameters))
        if now != self.parameter_storage:
            raise RuntimeError('Original parameter storage/shape/stride was replaced')
        for p in self.late_parameters:
            g = p.grad
            if g is None or g.device != self.device or g.dtype != self.dtype or (g.shape != p.shape) or (not g.is_contiguous()):
                raise ValueError('Complete original contiguous current mean L gradients required')
        current = torch.cuda.current_stream(self.device)
        if self._stream_id is not None and current.cuda_stream != self._stream_id:
            raise RuntimeError('Replay must keep the original single current stream')
        return current

    def _allocate(self):
        import torch
        self._gradients = tuple((torch.empty_like(p, memory_format=torch.contiguous_format, requires_grad=False) for p in self.late_parameters))
        self._half = torch.empty((128,), dtype=torch.float64, device=self.device)
        self._coefficient = torch.empty((), dtype=self.dtype, device=self.device)
        self._clamped_count = None

    def _stage(self, difference_half, coefficient, clamped_count):
        sources = tuple((p.grad.detach() for p in self.late_parameters)) + (difference_half, coefficient)
        targets = self._gradients + (self._half, self._coefficient)
        for (target, source) in zip(targets, sources):
            target.copy_(source)
        self._staging_copies += len(targets)

    def _calculate(self):
        return _tensor_decode_half(self.selected, self._gradients, self.selected_positions, self._half, self._coefficient, self._clamped_count)

    def _capture(self, current):
        import torch
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._capture_stream.wait_stream(current)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(self.warmup_iterations):
                warm = self._calculate()
                self._warmups += 1
            del warm
        self._capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._capture_stream):
            graph.capture_begin(capture_error_mode='global')
            try:
                output = self._calculate()
            finally:
                graph.capture_end()
        current.wait_stream(self._capture_stream)
        (self._graph, self._output) = (graph, output)
        self._capture_count += 1

    def decode_half(self, difference_half, actual_clip_coefficient, clamped_count_i64=None):
        import torch
        current = self._validate_inputs(difference_half, actual_clip_coefficient, clamped_count_i64)
        try:
            with torch.no_grad():
                if self._gradients is None:
                    self._allocate()
                self._stage(difference_half, actual_clip_coefficient, clamped_count_i64)
                if self._graph is None:
                    self._stream_id = current.cuda_stream
                    self._capture(current)
                self._graph.replay()
                self._calls += 1
            return self._output
        except BaseException:
            self._failed = True
            raise

    def close(self):
        self._closed = True
        return None
PRODUCTION_PAIR_ONLY = True
