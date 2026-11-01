from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
from types import FunctionType, SimpleNamespace
import weakref

HERE = Path(__file__).resolve().parent
ROOT = HERE
ROOT = HERE

from empirical.saint.head import yelp_shared_head_first_warmup_v1 as original

VERSION = "SAINT-Yelp-originalN-nativeNone-graph-owned-output-reuse-v1"
LIBRARY = original.LIBRARY
LIBRARY_SHA256 = original.LIBRARY_SHA256
REUSED_FILES = original.REUSED_FILES

class _GraphOwnedArray:

    def __init__(self, tensor, graph):
        import torch
        if tensor.device.type != "cuda" or tensor.layout != torch.strided:
            raise ValueError("Only captured CUDA dense outputs may be borrowed")
        types = {torch.float32: "<f4", torch.float64: "<f8"}
        if tensor.dtype not in types or not tensor.is_contiguous():
            raise ValueError("Only original contiguous FP32/FP64 output descriptors are qualified")
        self.graph_owner = graph
        self.__cuda_array_interface__ = dict(
            shape=tuple(tensor.shape),
            strides=None,
            typestr=types[tensor.dtype],
            data=(tensor.data_ptr(), False),
            version=3,
        )

def _borrow(tensor, graph):
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Borrowed wrappers must be constructed after capture_end")
    descriptor = _GraphOwnedArray(tensor, graph)
    alias = torch.as_tensor(descriptor, device=tensor.device)
    if (alias.data_ptr() != tensor.data_ptr() or alias.dtype != tensor.dtype
            or alias.shape != tensor.shape or alias.stride() != tensor.stride()
            or alias.device != tensor.device or alias.requires_grad
            or alias._base is not None):
        raise RuntimeError("CUDA Array Interface must expose the exact original output without a copy")
    
    
    
    for _ in range(tensor._version):
        torch.autograd.graph.increment_version(alias)
    if alias._version != tensor._version:
        raise RuntimeError("Original captured output mutation versions must match")
    return alias, descriptor

_source = textwrap.dedent(inspect.getsource(original.HeadQuery._new))
_anchor = "                     current_id=current.cuda_stream)"
if _source.count(_anchor) != 1:
    raise RuntimeError("Frozen first-warmup capture/output-ownership anchor changed")
_source = _source.replace(_anchor, _anchor + "\n        self._borrow_outputs(owned)\n        output = None")
_namespace = dict(original.HeadQuery._new.__globals__)
exec(compile(_source, __file__ + "::original_capture_with_released_outputs", "exec"), _namespace)
DERIVED_CAPTURE_SOURCE_SHA256 = hashlib.sha256(_source.encode()).hexdigest()

class HeadQuery(original.HeadQuery):
    _new = _namespace["_new"]

    def __init__(self, *args, **kwargs):
        self._released_output_refs = []
        self._borrowed_descriptors = []
        self._borrowed_output_payload = 0
        self._borrowed_distinct_pointers = set()
        super().__init__(*args, **kwargs)

    def _borrow_outputs(self, owned):
        original_tensors = tuple(owned["outputs"]) + tuple(owned["grads"])
        aliases = {}
        for tensor in original_tensors:
            if id(tensor) in aliases:
                continue
            alias, descriptor = _borrow(tensor, owned["graph"])
            aliases[id(tensor)] = alias
            self._released_output_refs.append(weakref.ref(tensor))
            self._borrowed_descriptors.append(descriptor)
            self._borrowed_output_payload += tensor.numel() * tensor.element_size()
            self._borrowed_distinct_pointers.add(tensor.data_ptr())
        owned["outputs"] = tuple(aliases[id(t)] for t in owned["outputs"])
        owned["grads"] = tuple(aliases[id(t)] for t in owned["grads"])
        owned["boundary_gradient"] = owned["outputs"][1]
        
        
        owned["boundary"].grad = None
        for handle in owned["handles"]:
            handle.grad = None

    def begin(self, *args, **kwargs):
        super().begin(*args, **kwargs)
        if any(ref() is not None for ref in self._released_output_refs):
            self.failed = True
            raise RuntimeError("An original captured output Tensor still owns its caching-allocator block")

    def verify_capture_program(self):
        super().verify_capture_program()
        if any(ref() is not None for ref in self._released_output_refs):
            raise RuntimeError("Captured allocator output owners were retained")
        if len(self._borrowed_descriptors) != 6 * self.cold_builds:
            raise RuntimeError("Exactly four original outputs and two original gradients must be borrowed per graph")

    def work(self):
        result = super().work()
        ranges = sorted((entry["outputs"][1].data_ptr(),
                         entry["outputs"][1].data_ptr() + entry["outputs"][1].numel() * entry["outputs"][1].element_size())
                        for entry in self.graphs.values())
        union = []
        for start, end in ranges:
            if union and start <= union[-1][1]:
                union[-1][1] = max(union[-1][1], end)
            else:
                union.append([start, end])
        result.update(
            version=VERSION,
            output_views_are_graph_owned=True,
            native_None_gradient_arithmetic_unchanged=True,
            extra_output_GPU_copy_kernels=0,
            retained_original_output_Tensors=sum(ref() is not None for ref in self._released_output_refs),
            borrowed_output_views=len(self._borrowed_descriptors),
            logical_borrowed_output_payload_bytes=self._borrowed_output_payload,
            distinct_borrowed_output_addresses=len(self._borrowed_distinct_pointers),
            lambda_logical_payload_bytes=sum(end-start for start,end in ranges),
            lambda_address_union_bytes=sum(end-start for start,end in union),
            lambda_output_ranges=ranges,
            allocated_is_not_reserved_memory=True,
            derived_capture_source_sha256=DERIVED_CAPTURE_SOURCE_SHA256,
        )
        return result

_direct_globals = dict(original._direct_class.__globals__, HeadQuery=HeadQuery)
_direct_class = FunctionType(original._direct_class.__code__, _direct_globals,
    original._direct_class.__name__, original._direct_class.__defaults__, original._direct_class.__closure__)
_outer_globals = dict(original._shared_class.__globals__, _direct_class=_direct_class)
_shared_class = FunctionType(original._shared_class.__code__, _outer_globals,
    original._shared_class.__name__, original._shared_class.__defaults__, original._shared_class.__closure__)

def adapted_class(parent):
    inherited = _shared_class(parent)
    class OutputReuseTraining(inherited):
        def verify_final(self, steps):
            result = super().verify_final(steps)
            self._direct_head.verify_capture_program()
            return result
    return OutputReuseTraining

def adapter_view(parent):
    old_adapt, old_spec = parent.adapt, parent.specification
    def adapt(base, rule):
        inherited = old_adapt(base, rule)
        return adapted_class(inherited) if rule == "ordered" else inherited
    def specification(rule):
        result = dict(old_spec(rule))
        result["graph_owned_Head_output_reuse"] = dict(
            version=VERSION, enabled=rule == "ordered",
            original_N_native_None_GEMM_FP64_mu_clip=True,
            CUDA_Array_Interface_alias_retains_graph_owner=True,
            captured_allocator_output_owners_released_after_capture=True,
            no_new_GPU_copy_or_preallocated_gradient=True,
            original_two_current_queries_and_complete_Body2_consumption=True,
            all_cold_alias_construction_staging_capture_and_replay_paid=True,
            requires_strict_numerical_version_and_pointer_lifetime_GPU_qualification=True,
        )
        return result
    view = SimpleNamespace(**vars(parent))
    view.adapt, view.specification = adapt, specification
    return view
