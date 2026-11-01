from __future__ import annotations
VERSION = 'reinit2-SAGE-originalN-twoport-Head-Graph-v1'

class HeadQuery:

    def __init__(self, engine, *, rows=(2000, 615), warmups=1, CPU_fixture=False):
        import torch
        self.engine = engine
        self.model = engine.model
        self.layer = self.model.convs[-1]
        self.parameters = (self.layer.lin_l.weight, self.layer.lin_l.bias, self.layer.lin_r.weight)
        if self.layer.lin_r.bias is not None or len({id(p) for p in self.parameters}) != 3:
            raise ValueError('Original three-parameter SAGE affine Head required')
        self.device = self.parameters[0].device
        self.dtype = self.parameters[0].dtype
        if self.device.type != 'cuda' and (not CPU_fixture) or self.dtype != torch.float32:
            raise ValueError('Production requires original CUDA FP32 Head')
        if not all((p.device == self.device and p.dtype == self.dtype and p.requires_grad for p in self.parameters)):
            raise ValueError('Original same-device live Parameters required')
        for module in (self.layer.lin_l, self.layer.lin_r):
            if any((getattr(module, name, {}) for name in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'))):
                raise RuntimeError('Qualified native affine modules have no action hooks')
        self.CPU_fixture = bool(CPU_fixture)
        if self.CPU_fixture and self.device.type != 'cpu':
            raise ValueError('CPU fixture cannot initialize CUDA')
        if warmups != 1:
            raise ValueError('Exactly one complete paid cold warmup declared')
        self.warmups = warmups
        self.rows = frozenset(rows)
        self.weights = None
        self.signature = self._signature()
        self.weight_signature = None
        self.workspaces = {}
        self.graphs = {}
        self.active = None
        self.contexts = self.queries = self.cold = self.warmup_calls = self.copies = 0
        self.replays = self.completed = self.seals = 0
        self.stream_id = None
        self.closed = False
        self.graph_modules = (self.layer.lin_l, self.layer.lin_r)

    def _signature(self):
        return tuple(((id(p), p.data_ptr(), tuple(p.shape), tuple(p.stride()), p.dtype, p.device) for p in self.parameters))

    def _validate_live(self):
        if self.closed or self._signature() != self.signature:
            raise RuntimeError('Original live Head parameter ownership changed')
        if self.layer is not self.model.convs[-1]:
            raise RuntimeError('Original Head module replaced')
        if self.graph_modules != (self.layer.lin_l, self.layer.lin_r):
            raise RuntimeError('Original two affine modules replaced')
        if self.weights is None:
            self.weights = self.engine.loss_weights
            if self.weights is None or self.weights.device != self.device or self.weights.dtype != self.dtype:
                raise ValueError('Original nonuniform TRAIN class weights required')
            self.weight_signature = (id(self.weights), self.weights.data_ptr(), self.weights._version)
        if self.engine.loss_weights is not self.weights or (id(self.weights), self.weights.data_ptr(), self.weights._version) != self.weight_signature:
            raise RuntimeError('Frozen training class weights changed')

    def _workspace(self, rows):
        import torch
        if rows not in self.workspaces:
            width = self.layer.lin_l.weight.shape[1]
            self.workspaces[rows] = dict(ports=tuple((torch.empty((rows, width), device=self.device, dtype=self.dtype, requires_grad=True) for _ in range(2))), labels=torch.empty((rows,), device=self.device, dtype=torch.long))
        return self.workspaces[rows]

    def begin(self, ports, labels, queries):
        import torch
        self._validate_live()
        if self.active is not None or queries not in (1, 2) or len(ports) != 2:
            raise RuntimeError('One original complete context/query pair required')
        rows = len(labels)
        if rows not in self.rows or labels.shape != (rows,) or labels.device != self.device or (labels.dtype != torch.long):
            raise ValueError('Unknown original N/labels signature; no padding/fallback')
        for value in ports:
            if value.shape != (rows, self.layer.lin_l.weight.shape[1]) or value.dtype != self.dtype or value.device != self.device or (not value.is_contiguous()):
                raise ValueError('Original contiguous affine ports required')
        workspace = self._workspace(rows)
        with torch.no_grad():
            for (dest, source) in zip(workspace['ports'], ports):
                dest.copy_(source)
            workspace['labels'].copy_(labels)
        self.copies += 3
        self.contexts += 1
        self.active = dict(rows=rows, queries=queries, next_slot=0, original_ports=ports)

    def _compute(self, workspace):
        import torch.nn.functional as F
        for parameter in self.parameters:
            parameter.grad = None
        for leaf in workspace['ports']:
            leaf.grad = None
        loss = F.cross_entropy(self.engine.head(workspace['ports']), workspace['labels'], weight=self.weights)
        loss.backward()
        measurement = self.engine.head_measurement()
        return dict(loss=loss, measurement=measurement, parameter_grads=tuple((p.grad for p in self.parameters)), port_grads=tuple((p.grad for p in workspace['ports'])))

    def _new(self, rows, slot):
        import torch
        workspace = self.workspaces[rows]
        before = tuple((p.grad for p in self.parameters))
        try:
            if self.CPU_fixture:
                output = self._compute(workspace)
                graph = None
            else:
                current = torch.cuda.current_stream(self.device)
                if self.stream_id is not None and current.cuda_stream != self.stream_id:
                    raise RuntimeError('Original single current stream changed')
                self.stream_id = current.cuda_stream
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    warm = self._compute(workspace)
                    self.warmup_calls += 1
                    del warm
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    graph.capture_begin(capture_error_mode='global')
                    try:
                        output = self._compute(workspace)
                    finally:
                        graph.capture_end()
                current.wait_stream(stream)
                output['capture_stream'] = stream
        finally:
            for (parameter, gradient) in zip(self.parameters, before):
                parameter.grad = gradient
        output['graph'] = graph
        self.cold += 1
        return output

    def query(self, ports, labels, slot):
        import torch
        self._validate_live()
        if self.active is None or slot != self.active['next_slot'] or slot >= self.active['queries']:
            raise RuntimeError('Original two current queries must execute in order')
        if any((a is not b for (a, b) in zip(ports, self.active['original_ports']))):
            raise RuntimeError('Original detached query owners changed')
        key = (self.active['rows'], slot)
        fresh = key not in self.graphs
        if fresh:
            self.graphs[key] = self._new(*key)
        output = self.graphs[key]
        if self.CPU_fixture:
            if not fresh:
                before = tuple((p.grad for p in self.parameters))
                try:
                    output = self._compute(self.workspaces[key[0]])
                    output['graph'] = None
                    self.graphs[key] = output
                finally:
                    for (parameter, gradient) in zip(self.parameters, before):
                        parameter.grad = gradient
        else:
            current = torch.cuda.current_stream(self.device)
            if current.cuda_stream != self.stream_id:
                raise RuntimeError('Single-stream graph replay required')
            output['graph'].replay()
        for (parameter, gradient) in zip(self.parameters, output['parameter_grads']):
            parameter.grad = gradient
        for (port, gradient) in zip(ports, output['port_grads']):
            port.grad = gradient
        self.queries += 1
        self.replays += 1
        self.active['next_slot'] += 1
        return (output['loss'], output['measurement'])

    def finish_context(self):
        if self.active is None or self.active['next_slot'] != self.active['queries']:
            raise RuntimeError('Both current Head outputs must precede both original Body submissions')
        self.active = None
        self.completed += 1

    def close(self):
        if self.active is not None or self.completed != self.contexts:
            raise RuntimeError('Unconsumed graph context at original completion boundary')
        self.closed = True
        self.seals += 1
        return self.work()

    def work(self):
        return dict(version=VERSION, contexts=self.contexts, queries=self.queries, completed_contexts=self.completed, input_copies=self.copies, exact_N_queryslot_graphs=self.cold, warmup_queries=self.warmup_calls, graph_replays=self.replays, allowed_real_rows=sorted(self.rows), actual_graph_keys=[list(k) for k in sorted(self.graphs)], CPU_fixture=self.CPU_fixture, closed=self.closed, original_LR_optimizer_scheduler_actions_outside_graph=True, cold_allocation_warmup_capture_input_stage_paid=True, graph_private_pools_shared=False, bucket_padding=False, new_hot_CUDA_synchronizations=0, original_native_None_grad_acquisition=True)
