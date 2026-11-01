import importlib.util
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from empirical.saint.head import yelp_direct_head_query_graph_shared_warmup1 as shared

ROOT = Path(__file__).resolve().parent
REUSED = Path(__file__).resolve().parent
LIBRARY = REUSED / 'libmu_elementwise_sm80.so'
LIBRARY_SHA256='60153dd66553479877d32ca0c1bdf053612bc420d1ab1065937154edf67659f9'
REUSED_FILES=[REUSED/'mu_execution.py',REUSED/'mu_elementwise.cu',LIBRARY]
VERSION='T456-shared-oneWarm-Head-captured-exactFP64-mu-v1'

def _load():
    name='T456_original_exact_mu_CABI_for_graph'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,REUSED/'mu_execution.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]

class CapturedGeometry:
    def __init__(self,original):
        self.original=original
        self.measurement=None

    def channel_measurement(self,z,boundary,branch,raw_lambda):
        if self.measurement is None:
            
            
            self.measurement=_load().Measurement(LIBRARY)
        return self.measurement.measure(z,boundary,branch,raw_lambda,
            self.original.channel_measurement)

    def work(self):
        if self.measurement is None:
            return dict(initialized=False,host_measurement_calls=0)
        result=dict(self.measurement.work())
        result.update(initialized=True,
            host_calls_are_cold_warmup_or_capture_recordings_only=True,
            actual_steady_replays_execute_recorded_kernels_without_CABI_Python=True,
            original_two_native_ATen_sums_and_original_rounding_order=True)
        return result

    def close(self):
        if self.measurement is not None:self.measurement.close()

class HeadQuery(shared.HeadQuery):
    def __init__(self,model,parameters,geometry,**kwargs):
        self._captured_mu=CapturedGeometry(geometry)
        super().__init__(model,parameters,self._captured_mu,**kwargs)

    def verify_capture_program(self):
        expected=(self.warmups+1)*self.cold_builds
        measurement=self._captured_mu.measurement
        if measurement is None:
            if expected:raise RuntimeError('Captured mu helper was never initialized')
            return
        if (measurement.failures or measurement.calls!=expected
                or measurement.fast_calls!=expected or measurement.fallback_calls
                or measurement.launches!=2*expected or measurement.native_sums!=2*expected
                or measurement.current_stream_id!=self._cold_stream.cuda_stream):
            raise RuntimeError('Exactly two original native sums and two captured pointwise launches per cold program required')

    def work(self):
        result=super().work()
        result.update(version=VERSION,
            captured_pointwise_mu=self._captured_mu.work(),
            logical_real_FP64_mu_evaluations=self.calls,
            dummy_FP64_mu_evaluations=self.cold_builds*self.warmups,
            capture_FP64_mu_program_recordings=self.cold_builds,
            original_native_sum_device_executions=2*(self.calls+self.cold_builds*self.warmups),
            exact_pointwise_kernel_device_executions=2*(self.calls+self.cold_builds*self.warmups),
            prior_eager_performance_result_not_reused=True)
        return result

    def close(self):
        try:super().close()
        finally:self._captured_mu.close()

_original_direct=shared.adapted_class.__globals__['_direct_class']
_direct_globals=dict(_original_direct.__globals__)
_direct_globals['HeadQuery']=HeadQuery
_direct_class=FunctionType(_original_direct.__code__,_direct_globals,
    name='adapted_class',argdefs=_original_direct.__defaults__,closure=_original_direct.__closure__)
_outer_globals=dict(shared.adapted_class.__globals__)
_outer_globals['_direct_class']=_direct_class
_shared_class=FunctionType(shared.adapted_class.__code__,_outer_globals,
    name='adapted_class',argdefs=shared.adapted_class.__defaults__,closure=shared.adapted_class.__closure__)

def adapted_class(original):
    parent=_shared_class(original)
    class CapturedMuTraining(parent):
        def verify_final(self,steps):
            result=super().verify_final(steps)
            self._direct_head.verify_capture_program()
            return result
    return CapturedMuTraining

def adapter_view(original):
    old_adapt=original.adapt
    def adapt(parent,rule):
        inherited=old_adapt(parent,rule)
        return adapted_class(inherited) if rule=='ordered' else inherited
    old_specification=original.specification
    def specification(rule):
        result=dict(old_specification(rule))
        result['shared_Head_captured_exact_FP64_mu']=dict(
            version=VERSION,enabled=rule=='ordered',binary=str(LIBRARY),
            declared_binary_sha256=LIBRARY_SHA256,
            original_two_native_ATen_reductions_and_RN_order=True,
            no_FMA_reassociation_precision_or_original_N_change=True,
            host_CABI_only_at_paid_warmup_and_capture=True,
            all_load_cold_stage_replay_work_paid=True,
            old_eager_negative_preserved_not_new_performance_evidence=True,
            requires_new_combined_strict_qualification=True)
        return result
    result=SimpleNamespace(**vars(original))
    result.adapt,result.specification=adapt,specification
    return result
