from pathlib import Path
import ctypes
import hashlib
import os
import sys
import threading
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE
ROOT = HERE

from empirical.saint.head import yelp_head_output_reuse_v2 as previous

VERSION = 'Yelp-native-serial-main-and-Head-BLAS-workspace-owner-v1'
LIBRARY = HERE / 'libserial_blas_workspace_owner_v1.so'
_library = None
_capture_streams = {}

def library():
    global _library
    if _library is None:
        value = ctypes.CDLL(str(LIBRARY))
        value.yelp_blas_workspace_abi_v1.restype = ctypes.c_int
        if value.yelp_blas_workspace_abi_v1() != 1:
            raise RuntimeError('Fixed workspace owner ABI1 required')
        value.yelp_blas_workspace_error_v1.restype = ctypes.c_char_p
        value.yelp_bind_serial_blas_workspace_v1.argtypes = [ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        value.yelp_bind_serial_blas_workspace_v1.restype = ctypes.c_int
        _library = value
    return _library

class WorkspaceMixin:
    def __init__(self, *args, **kwargs):
        self._serial_workspace_binding = None
        super().__init__(*args, **kwargs)

    def _serial_capture_stream(self, device):
        import torch
        current = torch.cuda.current_stream(device)
        key = (threading.get_ident(), device.index, current.cuda_stream)
        
        
        if key not in _capture_streams:
            _capture_streams[key] = torch.cuda.Stream(device=device)
        return _capture_streams[key]

    def _bind_serial_workspace(self, current, target):
        if self._serial_workspace_binding is not None:
            if (self._serial_workspace_binding['source_stream'] != current.cuda_stream
                    or self._serial_workspace_binding['capture_stream'] != target.cuda_stream):
                raise RuntimeError('Only the original fixed main/capture stream pair may share')
            return
        import torch
        if (torch.cuda.is_current_stream_capturing()
                or torch.cuda.current_stream(self.parameters[0].device).cuda_stream != current.cuda_stream
                or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
            raise RuntimeError('Original noncapturing current stream and workspace configuration required')
        handle = torch.cuda.current_blas_handle()
        classic, lt = ctypes.c_int(), ctypes.c_int()
        lib = library()
        status = lib.yelp_bind_serial_blas_workspace_v1(handle, current.cuda_stream,
            target.cuda_stream, ctypes.byref(classic), ctypes.byref(lt))
        if status:
            raise RuntimeError(lib.yelp_blas_workspace_error_v1().decode())
        self._serial_workspace_binding = dict(source_stream=current.cuda_stream,
            capture_stream=target.cuda_stream, classic_status=classic.value,
            optional_existing_LT_status=lt.value, thread=threading.get_ident(),
            classic_workspace_capacity_bytes=4096 * 8 * 1024,
            original_workspace_pointer_and_owner_shared=True,
            existing_wait_stream_order_retained=True)

    def work(self):
        result = super().work()
        result['serial_Blas_workspace'] = dict(version=VERSION,
            binding=self._serial_workspace_binding, workspace_size_changed=False,
            native_blas_call_or_algorithm_or_precision_changed=False,
            target_preexisting_different_address_rejected=True)
        return result

_source = previous.previous._source
_changes = [
    ('if self._cold_stream is None:self._cold_stream=torch.cuda.Stream(device=device)',
     'if self._cold_stream is None:self._cold_stream=self._serial_capture_stream(device)'),
    ('    stream.wait_stream(current)',
     '    stream.wait_stream(current)\n    self._bind_serial_workspace(current,stream)'),
]
_derived = _source
for _before, _after in _changes:
    if _derived.count(_before) != 1:
        raise RuntimeError('Original first-warm/output-owner capture source anchor changed')
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after in reversed(_changes):
    _inverse = _inverse.replace(_after, _before)
if _inverse != _source:
    raise RuntimeError('Only cold stream storage/owner bindings are permitted')
_namespace = dict(previous.previous._namespace)
exec(compile(_derived, __file__ + '::same_Head_capture_shared_workspace', 'exec'), _namespace)
DERIVED_CAPTURE_SHA256 = hashlib.sha256(_derived.encode()).hexdigest()

class HeadQuery(WorkspaceMixin, previous.HeadQuery):
    _new = _namespace['_new']

_direct_globals = dict(previous._direct_class.__globals__, HeadQuery=HeadQuery)
_direct_class = FunctionType(previous._direct_class.__code__, _direct_globals,
    previous._direct_class.__name__, previous._direct_class.__defaults__, previous._direct_class.__closure__)
_outer_globals = dict(previous._shared_class.__globals__, _direct_class=_direct_class)
_shared_class = FunctionType(previous._shared_class.__code__, _outer_globals,
    previous._shared_class.__name__, previous._shared_class.__defaults__, previous._shared_class.__closure__)

def adapted_class(parent):
    inherited = _shared_class(parent)
    class SharedSerialWorkspaceTraining(inherited):
        def verify_final(self, steps):
            result = super().verify_final(steps)
            self._direct_head.verify_capture_program()
            return result
    return SharedSerialWorkspaceTraining

def adapter_view(parent):
    original_adapt, original_spec = parent.adapt, parent.specification
    def adapt(base, rule):
        inherited = original_adapt(base, rule)
        return adapted_class(inherited) if rule == 'ordered' else inherited
    def specification(rule):
        result = dict(original_spec(rule))
        result['graph_owned_Head_output_reuse_and_serial_workspace'] = dict(
            version=VERSION, enabled=rule == 'ordered',
            original_outputReuse_v2_native_math_and_owner_rules=True,
            original_workspace_capacity_and_native_BLAS_calls=True,
            main_and_capture_stream_waits_unchanged=True,
            new_binding_and_library_load_inside_original_paid_TRAIN=True,
            requires_new_GPU_ownership_numeric_and_full_budget_validation=True)
        return result
    value = SimpleNamespace(**vars(parent))
    value.adapt, value.specification = adapt, specification
    return value
