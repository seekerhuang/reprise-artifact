from types import FunctionType
from empirical.saint.optim import yelp_native_adam_shared_staging_v2 as candidate

previous = candidate.previous
_adapted = FunctionType(previous.adapted_class.__code__,
    dict(previous.adapted_class.__globals__, install_native_adam=candidate.install_native_adam),
    previous.adapted_class.__name__, previous.adapted_class.__defaults__, previous.adapted_class.__closure__)
_view = FunctionType(previous.adapter_view.__code__,
    dict(previous.adapter_view.__globals__, adapted_class=_adapted),
    previous.adapter_view.__name__, previous.adapter_view.__defaults__, previous.adapter_view.__closure__)

def adapter_view(original):
    value = _view(original)
    old_spec = value.specification
    def specification(rule):
        result = dict(old_spec(rule))
        result['native_Adam_shared_gradient_staging'] = dict(
            version=candidate.VERSION, enabled=rule == 'ordered',
            one_stable_staging_tensor_per_actual_active_parameter=True,
            same_complete_copy_native_fused_replay_and_public_API=True,
            original_single_stream_and_options_guards_unchanged=True,
            all_import_constructor_staging_and_capture_work_paid=True,
            full_GPU_gate_required=True)
        return result
    value.specification = specification
    return value
