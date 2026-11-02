from pathlib import Path
from types import FunctionType, SimpleNamespace
from empirical.saint.head import yelp_direct_head_query_graph_shared_v3 as shared

VERSION='T456-Yelp-originalN-shared-Head-Graph-oneWarmup-v1'

class HeadQuery(shared.HeadQuery):
    def __init__(self,*args,warmups=1,**kwargs):
        if warmups!=1:raise ValueError('This candidate declares exactly one complete cold warmup')
        super().__init__(*args,warmups=1,**kwargs)
    def work(self):
        result=super().work()
        result.update(version=VERSION,declared_complete_cold_warmups=1,
            one_complete_native_Head_backward_mu_clip_dummy_per_capture=True,
            no_optimizer_or_RNG_calls_in_warmup=True)
        return result

_direct_globals=dict(shared._direct_class.__globals__)
_direct_globals['HeadQuery']=HeadQuery
_direct_class=FunctionType(shared._direct_class.__code__,_direct_globals,
    name='adapted_class',argdefs=shared._direct_class.__defaults__,closure=shared._direct_class.__closure__)
_shared_globals=dict(shared.adapted_class.__globals__)
_shared_globals['_direct_class']=_direct_class
adapted_class=FunctionType(shared.adapted_class.__code__,_shared_globals,
    name='adapted_class',argdefs=shared.adapted_class.__defaults__,closure=shared.adapted_class.__closure__)

def adapter_view(original):
    old_adapt=original.adapt
    def adapt(parent,rule):
        inherited=old_adapt(parent,rule)
        return adapted_class(inherited) if rule=='ordered' else inherited
    old_specification=original.specification
    def specification(rule):
        result=dict(old_specification(rule))
        result['shared_originalN_oneWarmup_Head_query_graph']=dict(
            version=VERSION,enabled=rule=='ordered',complete_dummy_warmups_per_capture=1,
            original_N_native_None_gradient_GEMM_mu_clip=True,
            one_shared_pool_cold_stream_and4096row_workspace=True,
            all_dummy_cold_work_and_stage_paid_in_TRAIN=True)
        return result
    result=SimpleNamespace(**vars(original))
    result.adapt,result.specification=adapt,specification
    return result
