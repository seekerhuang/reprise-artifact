from pathlib import Path
from types import FunctionType, MethodType

VERSION='SAINT-proven-canonical-positive-CPU-COO-metadata-v1'

def converter(original,work):
    def convert(adjacency):
        import numpy as np
        import torch
        work['calls']+=1
        bounds=np.finfo(np.float32)
        if (adjacency.format!='coo' or adjacency.ndim!=2 or not adjacency.has_canonical_format
                or not np.isfinite(adjacency.data).all()
                or not (adjacency.data>=bounds.tiny).all()or not (adjacency.data<=bounds.max).all()):
            work['fallback']+=1
            return original(adjacency)
        
        
        indices=np.vstack((adjacency.row,adjacency.col))
        index_tensor=torch.LongTensor(indices)
        values=torch.FloatTensor(adjacency.data)
        work['canonical_positive']+=1
        return torch.sparse_coo_tensor(index_tensor,values,torch.Size(adjacency.shape),
                                       is_coalesced=True,check_invariants=False)
    return convert

def install(instance):
    original=instance.one_batch
    function=original.__func__
    if '_coo_scipy2torch'not in function.__globals__:
        raise RuntimeError('Actual original one_batch converter lookup required')
    if getattr(instance,'_reprise_canonical_coo',None)is not None:
        raise RuntimeError('Canonical converter installed twice')
    original_converter=function.__globals__['_coo_scipy2torch']
    work=dict(version=VERSION,calls=0,canonical_positive=0,fallback=0)
    namespace=dict(function.__globals__,_coo_scipy2torch=converter(original_converter,work))
    private=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    private.__kwdefaults__=function.__kwdefaults__
    had_instance='one_batch'in instance.__dict__
    saved=instance.__dict__.get('one_batch')
    instance.one_batch=MethodType(private,instance)
    instance._reprise_canonical_coo=dict(work=work,private=private,original=function,
                                       original_converter=original_converter,
                                       had_instance=had_instance,saved=saved)
    return work

def close(instance):
    record=instance._reprise_canonical_coo
    if instance.one_batch.__func__ is not record['private']:
        raise RuntimeError('Private canonical converter was replaced')
    if record['had_instance']:instance.one_batch=record['saved']
    else:del instance.one_batch
    del instance._reprise_canonical_coo
    return dict(record['work'])
