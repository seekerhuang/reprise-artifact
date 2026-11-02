from pathlib import Path
import inspect
import sys
import textwrap

ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent

from empirical.saint.optim import yelp_native_adam_graph as previous

VERSION = 'Yelp-native-fusedAdam-shared-per-parameter-staging-v1'
_source = textwrap.dedent(inspect.getsource(previous._prototype.NativeAdamGraphs._new))
_changes = [
    ('    static=[torch.empty_like(gradient) for gradient in saved]',
     '    static = [self._shared_staging(parameter, gradient) for parameter, gradient in zip(params, saved)]'),
    ('try:shadow.step()', 'try:self._capture_native_step(shadow)'),
]
_derived = _source
for _before, _after in _changes:
    if _derived.count(_before) != 1:
        raise RuntimeError('Original native Adam source anchor changed')
    _derived = _derived.replace(_before, _after)
_inverse = _derived
for _before, _after in reversed(_changes):
    _inverse = _inverse.replace(_after, _before)
if _inverse != _source:
    raise RuntimeError('Only staging allocation and existing hook-safe capture derivation are permitted')
_namespace = dict(previous._prototype.NativeAdamGraphs._new.__globals__)
exec(compile(_derived, __file__ + '::shared_native_capture', 'exec'), _namespace)
_capture = _namespace['_new']

class NativeAdamGraphs(previous.NativeAdamGraphs):
    def __init__(self, *args, **kwargs):
        self._shared_parameter_owners = {}
        self._shared_gradient_staging = {}
        super().__init__(*args, **kwargs)

    def _shared_staging(self, parameter, gradient):
        import torch
        key = id(parameter)
        if key not in self._shared_gradient_staging:
            self._shared_parameter_owners[key] = parameter
            self._shared_gradient_staging[key] = torch.empty_like(gradient)
        stage = self._shared_gradient_staging[key]
        if (self._shared_parameter_owners[key] is not parameter
                or stage.shape != gradient.shape or stage.dtype != gradient.dtype
                or stage.device != gradient.device or stage.stride() != gradient.stride()):
            raise RuntimeError('Original parameter and contiguous gradient descriptor must be stable')
        return stage

    def _new(self, support):
        key = tuple(tuple(id(p) for p in group) for group in support)
        if key not in self._expected_keys:
            raise RuntimeError('Only original C, C+body and body supports are qualified')
        return _capture(self, support)

    def work(self):
        result = super().work()
        result.update(shared_staging_version=VERSION,
            static_gradient_tensors=len(self._shared_gradient_staging),
            static_gradient_payload_bytes=sum(t.numel() * t.element_size()
                for t in self._shared_gradient_staging.values()),
            original_complete_copy_before_every_replay=True,
            original_single_stream_guard_unchanged=True)
        return result

def install_native_adam(optimizer, *, head_parameters, body_parameters, inactive_parameters=()):
    return NativeAdamGraphs(optimizer, head_parameters=head_parameters,
        body_parameters=body_parameters, inactive_parameters=inactive_parameters).install()
