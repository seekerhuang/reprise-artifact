from types import FunctionType
from empirical.saint.head import yelp_serial_blas_workspace_v2 as candidate

previous = candidate.previous
_view = FunctionType(previous.adapter_view.__code__,
    dict(previous.adapter_view.__globals__, adapted_class=candidate.adapted_class),
    previous.adapter_view.__name__, previous.adapter_view.__defaults__, previous.adapter_view.__closure__)

def adapter_view(original):
    value = _view(original)
    old_spec = value.specification
    def specification(rule):
        result = dict(old_spec(rule))
        result['serial_main_Head_Blas_workspace'] = dict(version=candidate.VERSION,
            enabled=rule == 'ordered', same_original_workspace_capacity=True,
            only_known_serialized_streams_share_original_DataPtr_owner=True,
            no_global_operator_or_math_or_hyperparameter_change=True,
            existing_different_captured_target_addresses_rejected=True,
            library_import_and_initial_binding_paid=True,
            new_strict_GPU_gate_required=True)
        return result
    value.specification = specification
    return value
