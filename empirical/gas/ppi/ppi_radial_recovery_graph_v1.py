import copy
VERSION = 'PPI-original-euclidean-radial-recovery-fixed1024-graph-v1'

class Recovery:

    def __init__(self, owner, complete, install, *, CPU_fixture=False):
        import torch
        self.owner = owner
        self.parameters = tuple(owner.parameters)
        if owner.spec['metric'] != 'euclidean' or len(self.parameters) != 2 or any((tuple(p.shape) != (1024,) or p.dtype != torch.float32 for p in self.parameters)):
            raise ValueError('Original PPI terminal BN2x1024 FP32/euclidean required')
        self.device = self.parameters[0].device
        if self.device.type != 'cuda' and (not CPU_fixture):
            raise ValueError('CUDA production required')
        if CPU_fixture and self.device.type != 'cpu':
            raise ValueError('CPU fixture cannot access CUDA')
        self.CPU_fixture = CPU_fixture
        self.original_complete = complete
        self.original_install = install
        self.signature = tuple(((id(p), p.data_ptr(), tuple(p.stride()), p.dtype, p.device) for p in self.parameters))
        self.static_gradients = tuple((torch.empty_like(p) for p in self.parameters))
        self.static_queries = tuple((torch.empty((1024,), dtype=torch.float64, device=self.device) for _ in range(2)))
        self.probe = copy.copy(owner)
        self.graph = self.stream = None
        self.outputs = None
        self.current_id = None
        self.calls = self.captures = self.warmups = self.staging_copies = self.installs = 0
        self.active = False
        self.next_install = 0
        self.closed = self.failed = False

    def _check(self):
        if self.closed or self.failed:
            raise RuntimeError('Recovery graph is sealed')
        if self.signature != tuple(((id(p), p.data_ptr(), tuple(p.stride()), p.dtype, p.device) for p in self.parameters)):
            raise RuntimeError('Original terminal BN Parameter/storage owner changed')

    def _compute(self, optimizer):
        probe = self.probe
        probe.pending = 1
        probe.queries = list(self.static_queries)
        probe.body_submissions = 0
        for (p, g) in zip(self.parameters, self.static_gradients):
            p.grad = g
        self.original_complete(probe, optimizer)
        self.original_install(probe, 0)
        pair0 = tuple((p.grad for p in self.parameters))
        self.original_install(probe, 1)
        pair1 = tuple((p.grad for p in self.parameters))
        return dict(n=probe.n, mean=probe.mean, r=probe.r, pair=(pair0, pair1))

    def _capture(self, optimizer):
        import torch
        saved = tuple((p.grad for p in self.parameters))
        try:
            if self.CPU_fixture:
                self.outputs = self._compute(optimizer)
                return
            current = torch.cuda.current_stream(self.device)
            self.current_id = current.cuda_stream
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                dummy = self._compute(optimizer)
                self.warmups += 1
                del dummy
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self.stream):
                graph.capture_begin(capture_error_mode='global')
                try:
                    self.outputs = self._compute(optimizer)
                finally:
                    graph.capture_end()
            current.wait_stream(self.stream)
            self.graph = graph
            self.captures += 1
        finally:
            for (p, g) in zip(self.parameters, saved):
                p.grad = g

    def complete(self, optimizer):
        import torch
        self._check()
        owner = self.owner
        if self.active or owner.pending != 1 or len(owner.queries) != 2:
            raise RuntimeError('Original two real measurements and mean VJP must precede recovery')
        if any((p.grad is None for p in self.parameters)):
            raise RuntimeError('Actual mean Body gradient support missing')
        if any((q.shape != (1024,) or q.dtype != torch.float64 or q.device != self.device for q in owner.queries)):
            raise RuntimeError('Original two FP64 row-sum outputs required')
        with torch.no_grad():
            torch._foreach_copy_(self.static_gradients, tuple((p.grad for p in self.parameters)))
            torch._foreach_copy_(self.static_queries, tuple(owner.queries))
        self.staging_copies += 4
        if self.outputs is None:
            self._capture(optimizer)
        if self.CPU_fixture:
            saved = tuple((p.grad for p in self.parameters))
            try:
                self.outputs = self._compute(optimizer)
            finally:
                for (p, g) in zip(self.parameters, saved):
                    p.grad = g
        else:
            if torch.cuda.current_stream(self.device).cuda_stream != self.current_id:
                raise RuntimeError('Original single current stream must own all recovery consumers')
            self.graph.replay()
        owner.n = self.outputs['n']
        owner.mean = self.outputs['mean']
        owner.r = self.outputs['r']
        owner.feature = None
        owner.pending = 2
        self.active = True
        self.next_install = 0
        self.calls += 1

    def install(self, position):
        self._check()
        if not self.active or self.owner.pending != 2 or position != self.next_install or (position not in (0, 1)):
            raise RuntimeError('Original Body1 then Body2 must consume each owned pair exactly once')
        for (p, g) in zip(self.parameters, self.outputs['pair'][position]):
            p.grad = g
        self.owner.body_submissions += 1
        self.installs += 1
        self.next_install += 1

    def finish(self):
        self._check()
        if not self.active or self.next_install != 2:
            raise RuntimeError('Both original native Body submissions must finish before next recovery')
        self.active = False

    def close(self):
        if self.active:
            raise RuntimeError('Recovery outputs still have an original Body consumer')
        self.closed = True

    def work(self):
        return dict(version=VERSION, real_complete_calls=self.calls, real_pair_installs=self.installs, cold_graphs=self.captures, complete_dummy_recoveries=self.warmups, exact_current_input_copies=self.staging_copies, original_two_FP64_row_sums_inherited_at_original_query_positions=True, original_complete_and_both_install_methods_invoked_during_capture=True, native_Adam_inside_graph=False, CPU_step_or_hook_protocol_changed=False, fixed_shape=[2, 1024], no_padding_or_dtype_or_expression_change=True, current_pair_has_pending_consumer=self.active, next_pair_install=self.next_install, all_actual_import_allocation_cold_warm_capture_stage_and_close_paid=True, CPU_fixture=self.CPU_fixture, failed=self.failed, closed=self.closed)

def adapt(original, *, CPU_fixture=False, total_contexts=None):

    class RecoveryGraph(original):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._recovery_graph = Recovery(self, original.complete_measurement, original.install, CPU_fixture=CPU_fixture)

        def complete_measurement(self, optimizer):
            try:
                self._recovery_graph.complete(optimizer)
            except BaseException:
                self._recovery_graph.failed = True
                raise

        def install(self, position):
            self._recovery_graph.install(position)

        def finish(self):
            super().finish()
            self._recovery_graph.finish()
            if total_contexts is not None and self.contexts == total_contexts:
                self._recovery_graph.close()

        def verify(self, contexts):
            result = super().verify(contexts)
            work = self._recovery_graph.work()
            if work['failed'] or work['real_complete_calls'] != contexts or work['real_pair_installs'] != 2 * contexts or work['current_pair_has_pending_consumer']:
                raise RuntimeError('Actual recovery graph/Body pair budget changed')
            return dict(result, recovery_graph=work)
    return RecoveryGraph
