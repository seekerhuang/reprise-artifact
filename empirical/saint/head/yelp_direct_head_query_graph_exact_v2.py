from pathlib import Path
from types import FunctionType, SimpleNamespace
from empirical.saint.head import yelp_direct_head_query_graph as base
from empirical.saint.head import yelp_direct_head_query_graph_exact as exact

VERSION='T456-Yelp-originalN-nativeNone-Head-mu-clip-Graph-v2'

class HeadQuery(exact.HeadQuery):
    def _new(self,rows,late):
        import torch
        dtype,device=self.parameters[0].dtype,self.parameters[0].device
        labels_count=self.model.classifier.f_bias[0].numel()
        owned=dict(
            boundary=torch.zeros((rows,512),dtype=dtype,device=device,requires_grad=True),
            z=torch.zeros((rows,512),dtype=dtype,device=device),
            branch=torch.zeros((rows,1),dtype=torch.bool,device=device),
            labels=torch.zeros((rows,labels_count),dtype=dtype,device=device),
            weights=torch.zeros((rows,),dtype=dtype,device=device),late=late)
        owned['handles']=tuple(p.detach() for p in self.parameters)
        original_gradients=tuple(p.grad for p in self.parameters)
        current=torch.cuda.current_stream(device)
        stream=torch.cuda.Stream(device=device)
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
                graph.capture_begin(capture_error_mode='global')
                try:output=compute()
                finally:graph.capture_end()
            current.wait_stream(stream)
            owned.update(graph=graph,outputs=output,stream=stream,
                         current_id=current.cuda_stream)
        finally:
            for parameter,gradient in zip(self.parameters,original_gradients):parameter.grad=gradient
        self.cold_builds+=1
        return owned

    def work(self):
        result=super().work()
        result.update(version=VERSION,native_None_gradient_acquisition=True,
            extra_pre_zero_or_accumulate_gradient_buffer=False,
            captured_native_gradient_output_ownership=True)
        return result

_globals=dict(base.adapted_class.__globals__)
_globals['HeadQuery']=HeadQuery
adapted_class=FunctionType(base.adapted_class.__code__,_globals,
    name='adapted_class',argdefs=base.adapted_class.__defaults__,
    closure=base.adapted_class.__closure__)

def adapter_view(original):
    old_adapt=original.adapt
    def adapt(parent,rule):
        inherited=old_adapt(parent,rule)
        return adapted_class(inherited) if rule=='ordered' else inherited
    old_specification=original.specification
    def specification(rule):
        result=dict(old_specification(rule))
        result['direct_originalN_nativeNone_Head_query_graph']=dict(
            version=VERSION,enabled=rule=='ordered',bucket_rows=1,
            native_None_gradient_acquisition=True,
            original_two_current_Head_queries_and_native_Adam=True,
            original_FP32_GEMM_FP64_measurement_native_Head_clip=True,
            no_padding_or_RNG_draws=True,all_cold_work_paid_in_TRAIN=True)
        return result
    result=SimpleNamespace(**vars(original))
    result.adapt,result.specification=adapt,specification
    return result
