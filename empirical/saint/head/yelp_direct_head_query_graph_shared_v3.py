from pathlib import Path
from types import FunctionType, SimpleNamespace
from empirical.saint.head import yelp_direct_head_query_graph as base
from empirical.saint.head import yelp_direct_head_query_graph_exact_v2 as native_none

VERSION='T456-Yelp-originalN-nativeNone-shared-Head-Graph-v3'

class HeadQuery(native_none.HeadQuery):
    def __init__(self,*args,workspace_rows=4096,**kwargs):
        if type(workspace_rows) is not int or workspace_rows<1:
            raise ValueError('Positive workspace capacity required')
        super().__init__(*args,**kwargs)
        self.workspace_rows=workspace_rows
        self._workspace=self._pool=self._cold_stream=None
        self._current_stream_id=None
        self.contexts_started=self.contexts_completed=self.failure_close_drains=0
        self._context_open=False

    def _inputs(self,rows):
        import torch
        if not 0<rows<=self.workspace_rows:
            raise ValueError('Original N exceeds the explicitly bounded shared workspace')
        if self._workspace is None:
            dtype,device=self.parameters[0].dtype,self.parameters[0].device
            classes=self.model.classifier.f_bias[0].numel()
            self._workspace=dict(
                boundary=torch.zeros((self.workspace_rows,512),dtype=dtype,device=device),
                z=torch.zeros((self.workspace_rows,512),dtype=dtype,device=device),
                branch=torch.zeros((self.workspace_rows,1),dtype=torch.bool,device=device),
                labels=torch.zeros((self.workspace_rows,classes),dtype=dtype,device=device),
                weights=torch.zeros((self.workspace_rows,),dtype=dtype,device=device))
        inputs={name:value[:rows] for name,value in self._workspace.items()}
        inputs['boundary']=inputs['boundary'].detach().requires_grad_(True)
        
        
        with torch.no_grad():
            for value in inputs.values():value.zero_()
        return inputs

    def _new(self,rows,late):
        import torch
        device=self.parameters[0].device
        owned=self._inputs(rows)
        owned['late']=late
        owned['handles']=tuple(p.detach() for p in self.parameters)
        original_gradients=tuple(p.grad for p in self.parameters)
        current=torch.cuda.current_stream(device)
        if self._current_stream_id is not None and current.cuda_stream!=self._current_stream_id:
            raise RuntimeError('Shared graph inputs require the original single current stream')
        self._current_stream_id=current.cuda_stream
        if self._cold_stream is None:self._cold_stream=torch.cuda.Stream(device=device)
        if self._pool is None:self._pool=torch.cuda.graph_pool_handle()
        stream=self._cold_stream
        stream.wait_stream(current)

        def compute():
            import torch.nn.functional as functional
            boundary=owned['boundary']
            boundary.grad=None
            for parameter in self.parameters:parameter.grad=None
            logits=self.model.classifier.f_lin[0](boundary)+self.model.classifier.f_bias[0]
            loss=functional.binary_cross_entropy_with_logits(logits,owned['labels'],
                weight=owned['weights'].unsqueeze(1),
                pos_weight=self.model._r69_late_positive_bce_scalar if late else None,
                reduction='sum')
            loss.backward()
            raw_lambda=boundary.grad
            owned['grads']=tuple(p.grad for p in self.parameters)
            owned['boundary_gradient']=raw_lambda
            for handle,gradient in zip(owned['handles'],owned['grads']):handle.grad=gradient
            mu=self.geometry.channel_measurement(owned['z'],boundary,owned['branch'],raw_lambda)
            norm=torch.nn.utils.clip_grad_norm_(owned['handles'],5.)
            return loss,raw_lambda,mu,norm

        try:
            with torch.cuda.stream(stream):
                for _ in range(self.warmups):dummy=compute()
                del dummy
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                graph.capture_begin(pool=self._pool,capture_error_mode='global')
                try:output=compute()
                finally:graph.capture_end()
            current.wait_stream(stream)
            owned.update(graph=graph,outputs=output,stream=stream,
                         current_id=current.cuda_stream)
        finally:
            for parameter,gradient in zip(self.parameters,original_gradients):parameter.grad=gradient
        self.cold_builds+=1
        return owned

    def begin(self,*args,**kwargs):
        if self._context_open:
            raise RuntimeError('Prior complete Body/Head consumption must finish before shared-input reuse')
        import torch
        if self._current_stream_id is not None and torch.cuda.current_stream(self.parameters[0].device).cuda_stream!=self._current_stream_id:
            raise RuntimeError('Shared input staging must keep the original current stream')
        super().begin(*args,**kwargs)
        self._context_open=True
        self.contexts_started+=1

    def finish_context(self):
        if not self._context_open or self.current_queries!=2:
            raise RuntimeError('A complete original two-current-query context must finish exactly once')
        if any(parameter.grad is not None for parameter in self.parameters):
            raise RuntimeError('Both original native Head Adam submissions must consume their gradients first')
        import torch
        current=torch.cuda.current_stream(self.parameters[0].device)
        if current.cuda_stream!=self._current_stream_id:
            raise RuntimeError('All consumers and next staging must keep the original current stream')
        
        
        self._context_open=False
        self.contexts_completed+=1

    def poison(self):
        self.failed=True

    def work(self):
        result=super().work()
        names=('boundary','z','branch','labels','weights')
        storage_counts={name:len({entry[name].data_ptr() for entry in self.graphs.values()}) for name in names}
        result.update(version=VERSION,shared_graph_pool=True,
            cold_stream_count=int(self._cold_stream is not None),
            original_N_preserved=True,workspace_capacity_rows=self.workspace_rows,
            shared_input_storage_counts=storage_counts,
            shared_input_storage_bytes=0 if self._workspace is None else sum(t.numel()*t.element_size() for t in self._workspace.values()),
            contexts_started=self.contexts_started,contexts_completed=self.contexts_completed,
            context_open=self._context_open,failure_close_drains=self.failure_close_drains,
            cross_context_host_synchronizations=0,
            next_stage_after_original_Body2_native_API=True)
        return result

    def close(self):
        if self.failed and self._cold_stream is not None:
            self._cold_stream.synchronize()
            self.failure_close_drains+=1
        super().close()

_globals=dict(base.adapted_class.__globals__)
_globals['HeadQuery']=HeadQuery
_direct_class=FunctionType(base.adapted_class.__code__,_globals,
    name='adapted_class',argdefs=base.adapted_class.__defaults__,closure=base.adapted_class.__closure__)

def adapted_class(original):
    parent=_direct_class(original)
    class SharedWorkspaceTraining(parent):
        def block(self,*args,**kwargs):
            try:
                result=super().block(*args,**kwargs)
                self._direct_head.finish_context()
                return result
            except BaseException:
                self._direct_head.poison()
                raise
        def verify_final(self,steps):
            result=super().verify_final(steps)
            work=self._direct_head.work()
            if work['contexts_started']!=self.blocks or work['contexts_completed']!=self.blocks or work['context_open']:
                raise RuntimeError('Shared Head inputs were reused before complete original block consumption')
            if any(count!=1 for count in work['shared_input_storage_counts'].values()):
                raise RuntimeError('Exactly one input storage per field required across all original N graphs')
            return result
    return SharedWorkspaceTraining

def adapter_view(original):
    old_adapt=original.adapt
    def adapt(parent,rule):
        inherited=old_adapt(parent,rule)
        return adapted_class(inherited) if rule=='ordered' else inherited
    old_specification=original.specification
    def specification(rule):
        result=dict(old_specification(rule))
        result['shared_originalN_nativeNone_Head_query_graph']=dict(
            version=VERSION,enabled=rule=='ordered',workspace_rows=4096,
            original_N_GEMM_and_reduction_shapes=True,
            one_shared_pool_and_one_cold_stream=True,
            input_storage_reused_only_after_complete_original_Body2_API=True,
            actual_capture_and_dummy_zero_warmup_staging_all_paid=True)
        return result
    result=SimpleNamespace(**vars(original))
    result.adapt,result.specification=adapt,specification
    return result
