from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE
ROOT = HERE

from empirical.saint.head import yelp_direct_head_query_graph_shared_v3 as shared
from empirical.saint.head import yelp_shared_head_captured_mu as original_mu

VERSION = 'T456-Yelp-originalN-shared-capturedMu-first-signature-warmup-v1'
LIBRARY = original_mu.LIBRARY
LIBRARY_SHA256 = original_mu.LIBRARY_SHA256
REUSED_FILES = original_mu.REUSED_FILES

class HeadQuery(shared.HeadQuery):
    def __init__(self, model, parameters, geometry, *, warmups=1, **kwargs):
        if warmups != 1:
            raise ValueError('Exactly one complete first-signature warmup is predeclared')
        self._captured_mu = original_mu.CapturedGeometry(geometry)
        self.total_complete_warmups = 0
        self.cold_warmup_synchronizations = 0
        self.cold_signature_preparation = []
        super().__init__(model, parameters, self._captured_mu, warmups=warmups, **kwargs)

    def _new(self,rows,late):
        import torch
        device=self.parameters[0].device
        owned=self._inputs(rows)
        owned['late']=late
        owned['handles']=tuple(p.detach() for p in self.parameters)
        original_gradients=tuple(p.grad for p in self.parameters)
        current=torch.cuda.current_stream(device)
        if self._current_stream_id is not None and current.cuda_stream!=self._current_stream_id:
            raise RuntimeError('Shared graph inputs require the original single current stream')
        self._current_stream_id=current.cuda_stream
        if self._cold_stream is None:self._cold_stream=torch.cuda.Stream(device=device)
        if self._pool is None:self._pool=torch.cuda.graph_pool_handle()
        stream=self._cold_stream
        stream.wait_stream(current)

        def compute():
            import torch.nn.functional as functional
            boundary=owned['boundary']
            boundary.grad=None
            for parameter in self.parameters:parameter.grad=None
            logits=self.model.classifier.f_lin[0](boundary)+self.model.classifier.f_bias[0]
            loss=functional.binary_cross_entropy_with_logits(logits,owned['labels'],
                weight=owned['weights'].unsqueeze(1),
                pos_weight=self.model._r69_late_positive_bce_scalar if late else None,
                reduction='sum')
            loss.backward()
            raw_lambda=boundary.grad
            owned['grads']=tuple(p.grad for p in self.parameters)
            owned['boundary_gradient']=raw_lambda
            for handle,gradient in zip(owned['handles'],owned['grads']):handle.grad=gradient
            mu=self.geometry.channel_measurement(owned['z'],boundary,owned['branch'],raw_lambda)
            norm=torch.nn.utils.clip_grad_norm_(owned['handles'],5.)
            return loss,raw_lambda,mu,norm

        actual_warmups = 1 if self.cold_builds == 0 else 0
        try:
            if actual_warmups:
                with torch.cuda.stream(stream):
                    dummy=compute()
                    self.total_complete_warmups += 1
                    del dummy
                stream.synchronize()
                self.cold_warmup_synchronizations += 1
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                graph.capture_begin(pool=self._pool,capture_error_mode='global')
                try:output=compute()
                finally:graph.capture_end()
            current.wait_stream(stream)
            owned.update(graph=graph,outputs=output,stream=stream,
                         current_id=current.cuda_stream)
        finally:
            for parameter,gradient in zip(self.parameters,original_gradients):parameter.grad=gradient
        self.cold_builds+=1
        self.cold_signature_preparation.append(dict(rows=rows,late=bool(late),
            complete_dummy_warmups=actual_warmups,host_warmup_synchronizations=actual_warmups))
        return owned

    def verify_capture_program(self):
        expected_warmups = int(self.cold_builds > 0)
        if (self.total_complete_warmups != expected_warmups
                or self.cold_warmup_synchronizations != expected_warmups
                or len(self.cold_signature_preparation) != self.cold_builds
                or sum(x['complete_dummy_warmups'] for x in self.cold_signature_preparation) != expected_warmups):
            raise RuntimeError('Actual first-only complete warmup/cold ownership counters changed')
        expected = self.cold_builds + self.total_complete_warmups
        measurement = self._captured_mu.measurement
        if measurement is None:
            if expected:
                raise RuntimeError('Original captured FP64 measurement was never initialized')
            return
        if (measurement.failures or measurement.calls != expected
                or measurement.fast_calls != expected or measurement.fallback_calls
                or measurement.launches != 2*expected or measurement.native_sums != 2*expected
                or measurement.current_stream_id != self._cold_stream.cuda_stream):
            raise RuntimeError('Original two native sums/pointwise kernels per actual warmup or capture required')

    def work(self):
        result = super().work()
        result.update(version=VERSION,
            dummy_warmup_queries=self.total_complete_warmups,
            first_signature_complete_dummy_warmups=1,
            subsequent_signature_complete_dummy_warmups=0,
            actual_cold_warmup_host_synchronizations=self.cold_warmup_synchronizations,
            cold_signature_preparation=list(self.cold_signature_preparation),
            captured_pointwise_mu=self._captured_mu.work(),
            logical_real_FP64_mu_evaluations=self.calls,
            dummy_FP64_mu_evaluations=self.total_complete_warmups,
            capture_FP64_mu_program_recordings=self.cold_builds,
            original_native_sum_device_executions=2*(self.calls+self.total_complete_warmups),
            exact_pointwise_kernel_device_executions=2*(self.calls+self.total_complete_warmups),
            every_real_current_query_kept=True,
            later_captures_keep_original_wait_stream_ownership=True,
            zero_warmup_capture_numeric_equivalence_requires_GPU_gate=True)
        return result

    def close(self):
        try:
            super().close()
        finally:
            self._captured_mu.close()

_direct_globals = dict(shared._direct_class.__globals__, HeadQuery=HeadQuery)
_direct_class = FunctionType(shared._direct_class.__code__, _direct_globals,
    shared._direct_class.__name__, shared._direct_class.__defaults__, shared._direct_class.__closure__)
_outer_globals = dict(shared.adapted_class.__globals__, _direct_class=_direct_class)
_shared_class = FunctionType(shared.adapted_class.__code__, _outer_globals,
    shared.adapted_class.__name__, shared.adapted_class.__defaults__, shared.adapted_class.__closure__)

def adapted_class(original):
    parent = _shared_class(original)
    class FirstSignatureWarmupTraining(parent):
        def verify_final(self, steps):
            result = super().verify_final(steps)
            self._direct_head.verify_capture_program()
            return result
    return FirstSignatureWarmupTraining

def adapter_view(original):
    old_adapt, old_specification = original.adapt, original.specification
    def adapt(parent, rule):
        inherited = old_adapt(parent, rule)
        return adapted_class(inherited) if rule == 'ordered' else inherited
    def specification(rule):
        result = dict(old_specification(rule))
        result['shared_Head_first_signature_warmup'] = dict(version=VERSION,enabled=rule=='ordered',
            first_signature_complete_warmups=1,subsequent_signature_complete_warmups=0,
            original_N_and_native_None_two_queries_and_FP64_mu_and_clip_preserved=True,
            original_shared_pool_Body2_consumption_and_stream_waits_preserved=True,
            all_actual_import_first_warmup_capture_staging_and_replay_work_paid=True,
            requires_new_strict_and_whole_cold_head_GPU_diagnostic=True)
        return result
    result = SimpleNamespace(**vars(original))
    result.adapt, result.specification = adapt, specification
    return result
