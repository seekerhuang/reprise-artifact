from pathlib import Path
import ctypes
import inspect
from types import SimpleNamespace
VERSION = 'Reddit-native-two-sums-exact-FP64-half-pointwise-v1'
HERE = Path(__file__).resolve().parent
LIBRARY = HERE / 'libreddit_half_pointwise_sm80.so'

class Measurement:

    def __init__(self, path=LIBRARY):
        self.path = Path(path)
        self.library = ctypes.CDLL(str(self.path))
        self.library.reddit_half_pointwise_abi_version.restype = ctypes.c_int
        if self.library.reddit_half_pointwise_abi_version() != 1:
            raise RuntimeError('Exact qualified ABI1 required')
        (ptr, count, integer) = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_int)
        self.library.reddit_half_product.argtypes = [ptr, ptr, ptr, ptr, count, ptr, integer]
        self.library.reddit_half_product.restype = integer
        self.library.reddit_half_residual.argtypes = [ptr, ptr, ptr, ptr, ptr, count, ptr, integer]
        self.library.reddit_half_residual.restype = integer
        self.calls = self.fast_calls = self.fallback_calls = self.failures = self.launches = self.native_sums = 0
        self.closed = False
        self.current_stream_id = None

    @staticmethod
    def qualified(torch, z, boundary, branch, a, b):
        if z.device.type != 'cuda' or z.ndim != 2 or z.shape[1] != 128 or (z.shape[0] <= 0):
            return False
        if branch.shape != (z.shape[0], 1) or branch.dtype != torch.bool:
            return False
        if any((t.shape != z.shape or t.dtype != torch.float32 for t in (z, boundary, a, b))):
            return False
        return all((t.device == z.device and t.is_contiguous() and (t.storage_offset() == 0) for t in (z, boundary, branch, a, b)))

    def measure(self, z, boundary, branch, a, b, original, *, intermediates=False):
        import torch
        if self.closed or self.failures:
            raise RuntimeError('Failed or closed measurement cannot continue')
        if not self.qualified(torch, z, boundary, branch, a, b):
            self.calls += 1
            self.fallback_calls += 1
            self.native_sums += 2
            result = original(z, boundary, branch, a, b)
            return (result, None) if intermediates else result
        device = z.device.index
        if torch.cuda.current_device() != device:
            raise RuntimeError('Original current CUDA device required')
        stream = torch.cuda.current_stream(z.device).cuda_stream
        if self.current_stream_id is not None and stream != self.current_stream_id:
            raise RuntimeError('Original single current stream required')
        self.current_stream_id = stream
        product = torch.empty(z.shape, dtype=torch.float64, device=z.device)
        residual = torch.empty_like(product)
        if product.stride() != z.stride() or residual.stride() != product.stride():
            raise RuntimeError('Original reduction strides required')
        if product.data_ptr() % 256 or residual.data_ptr() % 256:
            raise RuntimeError('Original native allocation alignment class required')
        try:
            code = self.library.reddit_half_product(boundary.data_ptr(), a.data_ptr(), b.data_ptr(), product.data_ptr(), z.numel(), stream, device)
            if code:
                raise RuntimeError('Half-product CUDA launch failed: ' + str(code))
            self.launches += 1
            row_inner = product.sum(1, keepdim=True)
            self.native_sums += 1
            code = self.library.reddit_half_residual(z.data_ptr(), product.data_ptr(), row_inner.data_ptr(), branch.data_ptr(), residual.data_ptr(), z.numel(), stream, device)
            if code:
                raise RuntimeError('Residual CUDA launch failed: ' + str(code))
            self.launches += 1
            result = residual.sum(0)
            self.native_sums += 1
            self.calls += 1
            self.fast_calls += 1
            return (result, (product, row_inner, residual)) if intermediates else result
        except BaseException:
            self.failures += 1
            raise

    def close(self):
        self.closed = True

def adapted_class(parent):

    class ExactHalfPointwise(parent):

        def __init__(self, model, method):
            super().__init__(model, method)
            self._half_pointwise = Measurement()
            previous = super()._measure_difference_half
            closure = inspect.getclosurevars(previous.__func__)
            self._half_causal = closure.nonlocals['causal']

        def _measure_difference_half(self, a, b):
            if a is None or b is None or a.shape != b.shape:
                raise RuntimeError('Both original current raw preclip cotangents required')
            causal = self._half_causal

            def fallback(z, boundary, branch, aa, bb):
                return causal.structural.channel_half_difference(causal.geometry, z, boundary, branch, aa, bb)
            result = self._half_pointwise.measure(self._z, self._boundary, self._unclamped, a, b, fallback)
            self.structural_queries += 2
            self.structural_applications += 1
            return result

        def close(self):
            try:
                return super().close()
            finally:
                self._half_pointwise.close()
    return ExactHalfPointwise

def adapter_view(original):
    (old_adapt, old_spec) = (original.adapt, original.specification)

    def adapt(parent, rule):
        inherited = old_adapt(parent, rule)
        return adapted_class(inherited) if rule == 'ordered' else inherited
    view = SimpleNamespace(**vars(original))
    (view.adapt, view.specification) = (adapt, old_spec)
    return view
