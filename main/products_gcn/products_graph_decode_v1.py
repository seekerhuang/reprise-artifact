from dataclasses import dataclass
import math
from time import perf_counter
CHANNELS = 256
STAT_NAMES = ('measurement_difference_squared_sum', 'correction_squared_sum', 'mean_BN_gradient_squared_sum', 'mean_identity_defect_squared_sum', 'mean_measurement_squared_sum', 'zero_norm_channels', 'blocks')

@dataclass(frozen=True)
class DecodeResult:
    pair: tuple
    values: object
    last_measurement: dict

def radial_pair(gamma, beta, difference, strength=1.0):
    import torch
    ng = gamma.detach().double()
    nb = beta.detach().double()
    denominator = ng.square() + nb.square()
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    scale = torch.where(denominator > 0, difference.double() / safe, torch.zeros_like(denominator)) * strength
    return ((ng * scale).to(gamma.dtype), (nb * scale).to(beta.dtype))

def tensor_decode(selected, mean_gradients, measurements, one, strength=1.0):
    import torch
    with torch.no_grad():
        (gamma, beta) = selected
        (qg, qb) = mean_gradients
        difference = (measurements[1] - measurements[0]) * 0.5
        (rg, rb) = radial_pair(gamma, beta, difference, strength)
        (ng, nb) = (gamma.detach().double(), beta.detach().double())
        average_measurement = (measurements[0] + measurements[1]) * 0.5
        defect = ng * qg.double() + nb * qb.double() - average_measurement
        values = torch.stack((difference.square().sum(), rg.double().square().sum() + rb.double().square().sum(), qg.double().square().sum() + qb.double().square().sum(), defect.square().sum(), average_measurement.square().sum(), (ng.square() + nb.square() == 0).double().sum(), one))
        last = dict(difference=difference, rg=rg, rb=rb, mean_defect=defect)
        pair = ((qg - rg, qb - rb), (qg + rg, qb + rb))
        return DecodeResult(pair, values, last)

def _bytes(tensors):
    return sum((t.numel() * t.element_size() for t in tensors))

class GraphDecode:

    def __init__(self, *, selected, strength=1.0, warmup_iterations=3):
        import torch
        started = perf_counter()
        self.selected = tuple(selected)
        if len(self.selected) != 2 or len({id(p) for p in self.selected}) != 2:
            raise ValueError('Two distinct original BN1 gamma/beta Parameters required')
        if not all((isinstance(p, torch.nn.Parameter) for p in self.selected)):
            raise TypeError('Bind original Parameter objects, not copied values')
        (self.device, self.dtype) = (self.selected[0].device, self.selected[0].dtype)
        if self.device.type != 'cuda' or self.dtype not in (torch.float32, torch.float64):
            raise ValueError('CUDA FP32/64 required; no CPU graph fallback')
        if any((p.device != self.device or p.dtype != self.dtype or tuple(p.shape) != (CHANNELS,) or (not p.is_contiguous()) for p in self.selected)):
            raise ValueError('Locked original contiguous BN1 pair [256] required')
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('Finite nonnegative original structural strength required')
        if type(warmup_iterations) is not int or warmup_iterations < 1:
            raise ValueError('Positive integer warmup count required')
        (self.strength, self.warmup_iterations) = (float(strength), warmup_iterations)
        self._storage = tuple(((p.data_ptr(), tuple(p.shape), tuple(p.stride())) for p in self.selected))
        self._q = self._mu = self._one = None
        self._graph = self._output = self._capture_stream = None
        self._stream_id = None
        self._closed = self._failed = False
        self._calls = self._captures = self._warmups = self._copies = self._staged_bytes = 0
        self._allocate_seconds = self._stage_seconds = self._capture_seconds = self._replay_seconds = 0.0
        self._memory_before = self._memory_after = None
        self._constructor_seconds = perf_counter() - started

    def _validate(self, measurements):
        import torch
        if self._closed or self._failed:
            raise RuntimeError('Graph decoder is closed or failed')
        if len(measurements) != 2:
            raise ValueError('Two current original pre-Head-step measurements required')
        for m in measurements:
            if m.device != self.device or m.dtype != torch.float64 or tuple(m.shape) != (CHANNELS,) or (not m.is_contiguous()):
                raise ValueError('Original contiguous FP64[256] measurements required')
        storage = tuple(((p.data_ptr(), tuple(p.shape), tuple(p.stride())) for p in self.selected))
        if storage != self._storage:
            raise RuntimeError('Original parameter storage/shape/stride was replaced')
        for p in self.selected:
            g = p.grad
            if g is None or g.device != self.device or g.dtype != self.dtype or (tuple(g.shape) != (CHANNELS,)) or (not g.is_contiguous()):
                raise ValueError('Current original contiguous BN1 mean gradients required')
        stream = torch.cuda.current_stream(self.device)
        if self._stream_id is not None and stream.cuda_stream != self._stream_id:
            raise RuntimeError('Original single current stream must remain unchanged')
        return stream

    def _allocate(self):
        import torch
        started = perf_counter()
        self._q = tuple((torch.empty_like(p, memory_format=torch.contiguous_format, requires_grad=False) for p in self.selected))
        self._mu = tuple((torch.empty((CHANNELS,), dtype=torch.float64, device=self.device) for _ in range(2)))
        self._one = torch.ones((), dtype=torch.float64, device=self.device)
        self._allocate_seconds += perf_counter() - started

    def _stage(self, measurements):
        started = perf_counter()
        targets = self._q + self._mu
        sources = tuple((p.grad.detach() for p in self.selected)) + tuple(measurements)
        for (target, source) in zip(targets, sources):
            target.copy_(source)
        self._copies += 4
        self._staged_bytes += _bytes(targets)
        self._stage_seconds += perf_counter() - started

    def _calculate(self):
        return tensor_decode(self.selected, self._q, self._mu, self._one, self.strength)

    def _capture(self, current):
        import torch
        started = perf_counter()
        self._memory_before = dict(allocated=torch.cuda.memory_allocated(self.device), reserved=torch.cuda.memory_reserved(self.device))
        stream = self._capture_stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_iterations):
                warm = self._calculate()
                self._warmups += 1
            del warm
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            graph.capture_begin(capture_error_mode='global')
            try:
                output = self._calculate()
            finally:
                graph.capture_end()
        current.wait_stream(stream)
        (self._graph, self._output) = (graph, output)
        self._captures += 1
        self._memory_after = dict(allocated=torch.cuda.memory_allocated(self.device), reserved=torch.cuda.memory_reserved(self.device))
        self._capture_seconds += perf_counter() - started

    def decode(self, measurements):
        import torch
        current = self._validate(measurements)
        try:
            with torch.no_grad():
                if self._q is None:
                    self._allocate()
                self._stage(measurements)
                if self._graph is None:
                    self._stream_id = current.cuda_stream
                    self._capture(current)
                started = perf_counter()
                self._graph.replay()
                self._replay_seconds += perf_counter() - started
                self._calls += 1
            return self._output
        except BaseException:
            self._failed = True
            raise

    def close(self):
        self._closed = True
        return self.work()

    def work(self):
        inputs = () if self._q is None else self._q + self._mu + (self._one,)
        outputs = () if self._output is None else (self._output.values, *self._output.last_measurement.values(), *self._output.pair[0], *self._output.pair[1])
        return dict(version='Products-BN1-post-mu-CUDA-Graph-v1', channels=CHANNELS, calls=self._calls, capture_count=self._captures, warmup_calculations=self._warmups, staging_copies=self._copies, staged_logical_bytes=self._staged_bytes, persistent_input_bytes_including_one=_bytes(inputs), named_output_bytes=_bytes(outputs), graph_private_pool_bytes=None, graph_private_pool_independently_measured=False, memory_before_capture=self._memory_before, memory_after_capture=self._memory_after, memory_scope='Whole allocator snapshots around capture, not isolated graph-pool or training peak.', host_perf_counter_seconds=dict(constructor=self._constructor_seconds, allocation=self._allocate_seconds, staging=self._stage_seconds, warmup_and_capture=self._capture_seconds, replay_dispatch=self._replay_seconds), timing_scope='Nested host dispatch intervals, not CUDA elapsed; first capture/copies and caller stats remain paid.', parameter_gradient_optimizer_writes=False, optimizer_or_rng_calls=0, external_stat_accumulation=True, external_stat_CPU_readback=True, current_stream_id=self._stream_id, closed=self._closed, failed=self._failed, storage_retained_after_close=True, lifetime='Four pair outputs and last measurement overwritten at next same-stream decode only after both native Body calls; close seals and retains storage through final synchronization.')
