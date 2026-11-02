from pathlib import Path
from types import SimpleNamespace, FunctionType
from empirical.saint.head import yelp_direct_head_query_graph as base

VERSION='T456-Yelp-originalN-current-Head-mu-clip-Graph-v1'

class HeadQuery(base.HeadQuery):
    def __init__(self, model, parameters, geometry, *, bucket_rows=1, warmups=3):
        if bucket_rows!=1:
            raise ValueError('This exact-N candidate never pads original rows')
        super().__init__(model,parameters,geometry,bucket_rows=1,warmups=warmups)

    def work(self):
        result=super().work()
        result.update(version=VERSION,bucket_rows=1,original_N_preserved=True,
            zero_padding_rows=0,shape_cache_entries=len(self.graphs),
            original_GEMM_and_FP64_reduction_shapes=True)
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
        result['direct_originalN_Head_query_graph']=dict(
            version=VERSION,enabled=rule=='ordered',bucket_rows=1,
            original_two_current_Head_queries_and_native_Adam=True,
            native_Head_clip_and_original_FP64_measurement=True,
            original_GEMM_and_FP64_reduction_shapes=True,
            no_padding_or_RNG_draws=True,cold_captures_and_all_staging_paid=True)
        return result
    result=SimpleNamespace(**vars(original))
    result.adapt,result.specification=adapt,specification
    return result
