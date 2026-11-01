from dataclasses import dataclass

CLIP_NORM = 5.

def euclidean(n,difference_half):
    import torch
    square=n.square().sum(0);nonzero=square!=0
    factor=torch.where(nonzero,difference_half/torch.where(nonzero,square,torch.ones_like(square)),torch.zeros_like(square))
    return n*factor.unsqueeze(0),None

def maximal_tau(full_q_square,q_selected,r,cap=CLIP_NORM):
    import torch
    a=r.square().sum();b=(q_selected*r).sum().abs();raw_slack=cap*cap-full_q_square
    slack=raw_slack.clamp_min(0.)
    denominator=(b.square()+a*slack).sqrt()+b
    safe=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    candidate=(slack/safe).clamp(max=1.)
    
    
    
    tau=torch.where(a==0,torch.ones_like(a),torch.where(raw_slack>0,candidate,torch.zeros_like(a)))
    return tau,None,None

@dataclass(frozen=True)
class DecodeResult:
    pair: tuple
    values: object
    tau: object
    payload: tuple

def _tensor_decode(selected, late_gradients, selected_positions, measurements,
                   coefficient, clamped_count):

    import torch
    n=torch.stack([p.detach().double() for p in selected])
    q=torch.stack([late_gradients[j].detach().double() for j in selected_positions])
    raw_r,zero=euclidean(n,(measurements[1]-measurements[0])*.5)
    r=raw_r*coefficient.double()
    full_q_square=torch.stack([g.detach().double().square().sum() for g in late_gradients]).sum()
    tau,excess,outside=maximal_tau(full_q_square,q,r)
    pair=tuple(tuple((q[j]+sign*tau*r[j]).to(p.dtype) for j,p in enumerate(selected)) for sign in (-1.,1.))
    return DecodeResult(pair, None, tau, (n,q,raw_r,r))

class GraphDecode:

    def __init__(self, *, selected, late_parameters, warmup_iterations=3):
        import torch
        self.selected = tuple(selected)
        self.late_parameters = tuple(late_parameters)
        self.warmup_iterations = warmup_iterations
        if type(warmup_iterations) is not int or warmup_iterations < 1:
            raise ValueError('Positive integer warmup_iterations required')
        if len(self.selected) != 2 or len(self.late_parameters) != 4:
            raise ValueError('Locked Yelp gamma/beta and all four late parameters required')
        if len({id(p) for p in self.late_parameters}) != 4 or len({id(p) for p in self.selected}) != 2:
            raise ValueError('Unique original parameter objects required')
        self.device = self.selected[0].device
        self.dtype = self.selected[0].dtype
        if self.device.type != 'cuda' or self.dtype not in (torch.float32, torch.float64):
            raise ValueError('CUDA native FP32/FP64 inputs required; no CPU fallback')
        if any(p.device != self.device or p.dtype != self.dtype for p in self.late_parameters):
            raise ValueError('One original device and dtype required')
        if any(tuple(p.shape) != (512,) for p in self.selected):
            raise ValueError('Only the original 512-channel selected affine pair')
        if sorted(tuple(p.shape) for p in self.late_parameters) != [(512,),(512,),(512,),(512,1024)]:
            raise ValueError('Require original W[512,1024], bias/offset/scale[512]')
        ids = [id(p) for p in self.late_parameters]
        if any(id(p) not in ids for p in self.selected):
            raise ValueError('Selected parameters must be original late parameter identities')
        self.selected_positions = tuple(ids.index(id(p)) for p in self.selected)
        self.parameter_storage = tuple((p.data_ptr(),tuple(p.shape),tuple(p.stride()))
                                       for p in self.late_parameters)
        self._gradients = self._measurements = self._coefficient = self._clamped_count = None
        self._capture_stream = self._graph = self._output = None
        self._stream_id = None
        self._closed = self._failed = False
        self._calls = self._capture_count = self._warmups = self._staging_copies = 0

    def _validate_inputs(self, measurements, coefficient, clamped_count):
        import torch
        if self._closed or self._failed:
            raise RuntimeError('GraphDecode is closed or a prior CUDA call failed')
        if len(measurements) != 2:
            raise ValueError('Two actual current pre-Head-step channel measurements required')
        for value in measurements:
            if value.device != self.device or value.dtype != torch.float64 or tuple(value.shape) != (512,):
                raise ValueError('Two native FP64[512] measurement vectors required')
        if coefficient.device != self.device or coefficient.dtype != self.dtype or coefficient.numel() != 1 or coefficient.ndim != 0:
            raise ValueError('Original scalar native L clip coefficient required')
        
        now = tuple((p.data_ptr(),tuple(p.shape),tuple(p.stride())) for p in self.late_parameters)
        if now != self.parameter_storage:
            raise RuntimeError('Original parameter storage/shape/stride was replaced')
        for p in self.late_parameters:
            g = p.grad
            if g is None or g.device != self.device or g.dtype != self.dtype or g.shape != p.shape or not g.is_contiguous():
                raise ValueError('Complete original contiguous current mean L gradients required')
        current = torch.cuda.current_stream(self.device)
        if self._stream_id is not None and current.cuda_stream != self._stream_id:
            raise RuntimeError('Replay must keep the original single current stream')
        return current

    def _allocate(self):
        import torch
        self._gradients = tuple(torch.empty_like(p, memory_format=torch.contiguous_format,
                                                requires_grad=False) for p in self.late_parameters)
        self._measurements = tuple(torch.empty((512,), dtype=torch.float64, device=self.device) for _ in range(2))
        self._coefficient = torch.empty((), dtype=self.dtype, device=self.device)
        self._clamped_count = None  

    def _stage(self, measurements, coefficient, clamped_count):
        sources = tuple(p.grad.detach() for p in self.late_parameters) + tuple(measurements) + (coefficient,)
        targets = self._gradients + self._measurements + (self._coefficient,)
        for target,source in zip(targets,sources):
            target.copy_(source)
        self._staging_copies += len(targets)

    def _calculate(self):
        return _tensor_decode(self.selected, self._gradients, self.selected_positions,
                              self._measurements, self._coefficient, self._clamped_count)

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
        self._graph, self._output = graph, output
        self._capture_count += 1

    def decode(self, measurements, actual_clip_coefficient, clamped_count_i64=None):
        import torch
        current = self._validate_inputs(measurements, actual_clip_coefficient, clamped_count_i64)
        try:
            with torch.no_grad():
                if self._gradients is None:
                    self._allocate()
                self._stage(measurements, actual_clip_coefficient, clamped_count_i64)
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

    def work(self):
        
        return dict(version='R75-production-no-observer-v3', calls=self._calls,
            capture_count=self._capture_count, warmup_calculations=self._warmups,
            staging_copies=self._staging_copies,
            current_stream_id=self._stream_id, closed=self._closed, failed=self._failed,
            staging_copies_per_call=7,
            diagnostic_outputs_computed=False, external_statistics_accumulation=False,
            external_statistics_flush=False, host_profiler_computed=False,
            logical_payload_bytes_computed=False, allocator_probe_computed=False,
            close_builds_report=False, storage_retained_after_close=True,
            lifetime='Original same-stream staging/capture/replay and both Body submissions; close only seals and retains ownership.')

PRODUCTION_PAIR_ONLY = True
