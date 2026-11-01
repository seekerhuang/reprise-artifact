from pathlib import Path
from contextlib import contextmanager
import inspect
import textwrap
from types import FunctionType,SimpleNamespace
from empirical.saint.sparse import yelp_canonical_public_v1 as previous

HERE=Path(__file__).resolve().parent
ROOT = HERE
BACKEND=previous.BACKEND
METADATA=previous.METADATA
EXPECTED_METADATA_SHA256=previous.EXPECTED_METADATA_SHA256
metadata=previous.metadata
VERSION='Yelp-proven-canonical-2D-to-3D-metadata-original-public-bmm-v2'

def canonical_batch(adjacency):
    
    
    batched=adjacency.unsqueeze(0)
    if adjacency.is_coalesced():batched._coalesced_(True)
    return batched

public_source=textwrap.dedent(inspect.getsource(previous.ROUTE.public_singleton_mm))
before='return torch.bmm(adj.unsqueeze(0), dense.unsqueeze(0)).squeeze(0).contiguous()'
after='return torch.bmm(LOWER_CANONICAL_BATCH(adj), dense.unsqueeze(0)).squeeze(0).contiguous()'
assert public_source.count(before)==1
public_source=public_source.replace(before,after)
namespace=dict(previous.ROUTE.public_singleton_mm.__globals__,LOWER_CANONICAL_BATCH=canonical_batch)
exec(compile(public_source,__file__+'::same-public-bmm-with-canonical-leading-batch-metadata','exec'),namespace)
public_singleton_mm=namespace['public_singleton_mm']
ROUTE=SimpleNamespace(**vars(previous.ROUTE));ROUTE.public_singleton_mm=public_singleton_mm
generator=previous.ROUTE.install.__wrapped__
ROUTE.install=contextmanager(FunctionType(generator.__code__,dict(generator.__globals__,public_singleton_mm=public_singleton_mm),
    generator.__name__,generator.__defaults__,generator.__closure__))
original=SimpleNamespace(**vars(previous.original))
for name in('adapt','train_epoch'):
    function=getattr(previous.original,name)
    setattr(original,name,FunctionType(function.__code__,dict(function.__globals__,ROUTE=ROUTE),function.__name__,function.__defaults__,function.__closure__))
configure_before_torch=previous.configure_before_torch
_qualification=None

def specification():
    record=previous.specification()
    record['canonical_CPU_metadata'].update(version=VERSION,
        actual_public_bmm_unchanged=True,original_public_validation_int32_batch_D2H_and_autograd_retained=True,
        proven_2D_coalesced_flag_restored_on_new_3D_unsqueeze_only=True,
        native_unsqueeze_drops_flag_v1_inactive_counterexample_retained=True,
        original_transpose_coalesce_retained=True)
    return record

adapt=FunctionType(previous.adapt.__code__,dict(previous.adapt.__globals__,original=original,specification=specification),
    previous.adapt.__name__,previous.adapt.__defaults__,previous.adapt.__closure__)
train_epoch=FunctionType(previous.train_epoch.__code__,dict(previous.train_epoch.__globals__,original=original),
    previous.train_epoch.__name__,previous.train_epoch.__defaults__,previous.train_epoch.__closure__)
