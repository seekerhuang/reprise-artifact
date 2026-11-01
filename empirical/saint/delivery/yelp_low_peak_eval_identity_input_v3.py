from pathlib import Path
import inspect
import textwrap
from empirical.saint.delivery import yelp_low_peak_eval_v3 as base

VERSION='Yelp-full-eval-native-beta0-low-temp-identity-input-v3'
_evaluators={}

def _full_features(model,nodes):
    import torch
    features=model.feat_full
    n=features.shape[0]
    if not features.is_contiguous() or len(nodes)!=n:
        raise RuntimeError('Fixed contiguous full node population required')
    if torch.is_tensor(nodes):
        if nodes.ndim!=1 or nodes.dtype not in (torch.int32,torch.int64):
            raise RuntimeError('Full natural integer node order required')
        expected=torch.arange(n,device=nodes.device,dtype=nodes.dtype)
        valid=torch.equal(nodes,expected)
    else:
        import numpy as np
        array=np.asarray(nodes)
        valid=array.ndim==1 and np.issubdtype(array.dtype,np.integer) and np.array_equal(array,np.arange(n))
    if not valid:
        raise RuntimeError('Identity feature reuse cannot reorder or omit a node')
    return features

def _native_evaluator(original):
    if original in _evaluators:return _evaluators[original]
    source=textwrap.dedent(inspect.getsource(original))
    anchor='model.feat_full[nodes]'
    if original.__name__!='saint_evaluate' or source.count(anchor)!=1:
        raise RuntimeError('Only the exact native complete evaluator may skip its identity gather')
    source=source.replace(anchor,'_yelp_full_identity_features(model,nodes)')
    namespace=dict(original.__globals__,_yelp_full_identity_features=_full_features)
    exec(compile(source,__file__+'::native_evaluator_identity_gather_removed','exec'),namespace)
    _evaluators[original]=namespace[original.__name__]
    return _evaluators[original]

def evaluate(original_evaluate,model,minibatch,thresholds=(.5,)):
    
    
    if original_evaluate.__name__!='saint_evaluate':
        return base.evaluate(original_evaluate,model,minibatch,thresholds)
    return base.evaluate(_native_evaluator(original_evaluate),model,minibatch,thresholds)
