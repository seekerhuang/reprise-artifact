import math

class NativeAdamGraphs:
    def __init__(self,optimizer):
        import torch
        if type(optimizer) is not torch.optim.Adam:raise ValueError('Original native Adam required')
        if optimizer.state:raise ValueError('Install only before original empty-state training')
        if optimizer._optimizer_step_pre_hooks or optimizer._optimizer_step_post_hooks:
            raise ValueError('This finite prototype does not change or replay live hooks')
        self.optimizer=optimizer
        self.original_step=optimizer.step
        self.graphs={};self.calls=self.captures=self.staged_tensors=0
        self._stream=None;self._current_stream_id=None
        self._options=[]
        for g in optimizer.param_groups:
            if (g['fused'] is not True or g['foreach'] is not False or g['amsgrad']
                    or g['differentiable'] or g['weight_decay'] or g['maximize']):
                raise ValueError('Fixed fused FP32 native Adam/zero WD/no AMSGrad required')
            if any(p.dtype!=torch.float32 or not p.is_cuda for p in g['params']):
                raise ValueError('This prototype binds CUDA FP32 only')
            self._options.append({k:v for k,v in g.items() if k not in ['params','lr']})

    def _support(self):
        return tuple(tuple(p for p in g['params'] if p.grad is not None) for g in self.optimizer.param_groups)

    def _new(self,support):
        import torch
        real=self.optimizer
        params=[p for group in support for p in group]
        saved=[p.grad for p in params]
        
        
        for group in real.param_groups:
            real._init_group(group,[],[],[],[],[],[])
        static=[torch.empty_like(gradient) for gradient in saved]
        rates=[torch.empty((),device=params[0].device,dtype=torch.float32) for _ in support]
        for rate,source in zip(rates,real.param_groups):rate.fill_(float(source['lr']))
        groups=[]
        for source,active,rate in zip(real.param_groups,support,rates):
            g=dict(source);g.update(params=list(active),lr=rate,capturable=True)
            groups.append(g)
        shadow=torch.optim.Adam(groups)
        shadow.state=real.state
        current=torch.cuda.current_stream(params[0].device)
        if self._stream is None:self._stream=torch.cuda.Stream(device=params[0].device)
        self._stream.wait_stream(current)
        
        
        torch._foreach_copy_(static,saved)
        for rate,source in zip(rates,real.param_groups):rate.fill_(float(source['lr']))
        self._stream.wait_stream(current)
        graph=torch.cuda.CUDAGraph()
        state_tensors=[value for p in params for value in (p,real.state[p]['exp_avg'],real.state[p]['exp_avg_sq'],real.state[p]['step'])]
        versions=[value._version for value in state_tensors]
        gradient_versions=[value._version for value in static]
        try:
            for p,g in zip(params,static):p.grad=g
            with torch.cuda.stream(self._stream):
                graph.capture_begin(capture_error_mode='global')
                try:shadow.step()
                finally:graph.capture_end()
            current.wait_stream(self._stream)
        finally:
            for p,g in zip(params,saved):p.grad=g
        self.captures+=1
        deltas=[value._version-before for value,before in zip(state_tensors,versions)]
        grad_deltas=[value._version-before for value,before in zip(static,gradient_versions)]
        if any(value not in [0,1] for value in deltas+grad_deltas):
            raise RuntimeError('Unqualified native mutation-version behavior')
        return dict(graph=graph,parameters=params,static_gradients=static,rates=rates,shadow=shadow,
            version_tensors=[value for value,delta in zip(state_tensors,deltas) if delta],
            gradient_version_deltas=grad_deltas)

    def step(self,closure=None):
        import torch
        if closure is not None:raise ValueError('No closure is used by the fixed real training')
        support=self._support()
        if not any(support):raise ValueError('No empty native API support in this prototype')
        key=tuple(tuple(id(p) for p in group) for group in support)
        for group,options in zip(self.optimizer.param_groups,self._options):
            if {k:v for k,v in group.items() if k not in ['params','lr']}!=options:
                raise ValueError('Ordinary Adam options changed after capture')
        current=torch.cuda.current_stream()
        if self._current_stream_id is None:self._current_stream_id=current.cuda_stream
        if self._current_stream_id!=current.cuda_stream:raise ValueError('Original one-stream execution required')
        for group in self.optimizer.param_groups:
            value=group['lr']
            if not isinstance(value,float) or not math.isfinite(value) or value<0.:
                raise ValueError('Actual original finite Python float LR required')
        cold=key not in self.graphs
        if cold:self.graphs[key]=self._new(support)
        entry=self.graphs[key]
        gradients=[p.grad for p in entry['parameters']]
        torch._foreach_copy_(entry['static_gradients'],gradients)
        for rate,group in zip(entry['rates'],self.optimizer.param_groups):rate.fill_(group['lr'])
        entry['graph'].replay()
        
        
        
        if not cold and entry['version_tensors']:
            torch.autograd.graph.increment_version(entry['version_tensors'])
        grad_versions=[value for value,delta in zip(gradients,entry['gradient_version_deltas']) if delta]
        if grad_versions:torch.autograd.graph.increment_version(grad_versions)
        self.calls+=1;self.staged_tensors+=len(gradients)
        return None

    def work(self):
        return dict(actual_requested_API_calls=self.calls,captured_supports=self.captures,
            original_native_optimizer_state_owner=True,original_parameter_objects=True,
            native_zero_initialization_only=True,no_dummy_Adam_update=True,
            no_moment_or_step_reset=True,original_native_fused_Adam_kernel_recorded=True,
            original_native_host_mutation_versions_preserved=True,
            Python_LR_cast_to_original_kernel_FP32_opmath=True,
            gradient_tensors_staged=self.staged_tensors,
            real_driver_installed=False)
