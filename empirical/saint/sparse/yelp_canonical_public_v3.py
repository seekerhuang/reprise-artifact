from pathlib import Path
from contextlib import contextmanager
from types import FunctionType,SimpleNamespace
from empirical.saint.sparse import yelp_canonical_public_v2 as previous

HERE=Path(__file__).resolve().parent
ROOT = HERE
BACKEND=previous.BACKEND
METADATA=previous.METADATA
metadata=previous.metadata
VERSION='Yelp-proven-canonical-batch-independent-metadata-owner-public-bmm-v3'

def canonical_batch(adjacency):
    import torch
    batched=adjacency.unsqueeze(0)
    
    
    if not adjacency.is_coalesced()or adjacency._nnz()<=1:return batched
    
    
    
    return torch.sparse_coo_tensor(batched._indices(),batched._values(),batched.shape,
        is_coalesced=True,check_invariants=False)

public_singleton_mm=FunctionType(previous.public_singleton_mm.__code__,
    dict(previous.public_singleton_mm.__globals__,LOWER_CANONICAL_BATCH=canonical_batch),
    previous.public_singleton_mm.__name__,previous.public_singleton_mm.__defaults__,previous.public_singleton_mm.__closure__)
ROUTE=SimpleNamespace(**vars(previous.ROUTE));ROUTE.public_singleton_mm=public_singleton_mm
generator=previous.ROUTE.install.__wrapped__
ROUTE.install=contextmanager(FunctionType(generator.__code__,dict(generator.__globals__,public_singleton_mm=public_singleton_mm),
    generator.__name__,generator.__defaults__,generator.__closure__))
configure_before_torch=previous.configure_before_torch
_qualification=None

def specification():
    record=previous.specification()
    record['scope']='Original S TRAIN public torch.bmm, guarded fresh canonical CPU metadata and an independent canonical3D metadata owner; original arithmetic/indices/values/dtypes/transpose/native autograd retained.'
    record['canonical_CPU_metadata'].update(version=VERSION,
        original_2D_adjacency_metadata_storage_values_indices_and_versions_unchanged=True,
        new_3D_metadata_owner_uses_original_unsqueezed_index_and_value_storage=True,
        no_inplace_coalesced_mutation_of_aliasing_view=True,
        original_batch_index_creation_and_batch_D2H_retained=True)
    return record

original=SimpleNamespace(**vars(previous.original))
for name in('adapt','train_epoch'):
    function=getattr(previous.original,name)
    setattr(original,name,FunctionType(function.__code__,dict(function.__globals__,ROUTE=ROUTE,specification=specification),
        function.__name__,function.__defaults__,function.__closure__))
adapt=FunctionType(previous.adapt.__code__,dict(previous.adapt.__globals__,original=original,specification=specification),
    previous.adapt.__name__,previous.adapt.__defaults__,previous.adapt.__closure__)
train_epoch=FunctionType(previous.train_epoch.__code__,dict(previous.train_epoch.__globals__,original=original),
    previous.train_epoch.__name__,previous.train_epoch.__defaults__,previous.train_epoch.__closure__)
