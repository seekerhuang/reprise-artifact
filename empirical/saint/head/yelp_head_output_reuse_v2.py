from types import FunctionType, SimpleNamespace
from empirical.saint.head import yelp_head_output_reuse_v1 as previous

VERSION = "SAINT-Yelp-originalN-nativeNone-graph-owned-output-reuse-v2"
LIBRARY = previous.LIBRARY
REUSED_FILES = previous.REUSED_FILES

class HeadQuery(previous.HeadQuery):
    def __init__(self, *args, **kwargs):
        self._verified_released_owners = 0
        self._new_capture_ownership_checks = 0
        super().__init__(*args, **kwargs)

    def begin(self, *args, **kwargs):
        before = len(self._released_output_refs)
        
        
        previous.original.HeadQuery.begin(self, *args, **kwargs)
        after = len(self._released_output_refs)
        if after != before:
            newly_released = self._released_output_refs[before:after]
            if len(newly_released) != 6 or any(ref() is not None for ref in newly_released):
                self.failed = True
                raise RuntimeError("Exactly six new captured output owners must be released")
            self._verified_released_owners += len(newly_released)
            self._new_capture_ownership_checks += 1

    def verify_capture_program(self):
        super().verify_capture_program()
        if (self._verified_released_owners != len(self._released_output_refs)
                or self._new_capture_ownership_checks != self.cold_builds):
            raise RuntimeError("Every capture must receive one original-owner release check")

    def work(self):
        result = super().work()
        result.update(
            version=VERSION,
            released_owner_checks_only_on_new_capture=True,
            released_owner_weakref_checks=self._verified_released_owners,
            new_capture_ownership_check_calls=self._new_capture_ownership_checks,
            inherited_full_owner_check_in_work_and_verify=True,
        )
        return result

_direct_globals = dict(previous._direct_class.__globals__, HeadQuery=HeadQuery)
_direct_class = FunctionType(previous._direct_class.__code__, _direct_globals,
    previous._direct_class.__name__, previous._direct_class.__defaults__, previous._direct_class.__closure__)
_outer_globals = dict(previous._shared_class.__globals__, _direct_class=_direct_class)
_shared_class = FunctionType(previous._shared_class.__code__, _outer_globals,
    previous._shared_class.__name__, previous._shared_class.__defaults__, previous._shared_class.__closure__)

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
            released_owner_checks_only_on_new_capture=True,
        )
        return result
    view = SimpleNamespace(**vars(parent))
    view.adapt, view.specification = adapt, specification
    return view
