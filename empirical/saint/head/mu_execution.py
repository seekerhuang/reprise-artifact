import ctypes
from pathlib import Path
from types import SimpleNamespace
VERSION = 'R76-Yelp-existingoffset01-muElementwiseNativeReduce-v1'

def specification():
    return {}

class Measurement:

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.library = ctypes.CDLL(str(self.path))
        self.library.mu_elementwise_abi_version.restype = ctypes.c_int
        if self.library.mu_elementwise_abi_version() != 1:
            raise RuntimeError('Qualified CABI version1 required')
        ptr = ctypes.c_void_p
        i64 = ctypes.c_int64
        integer = ctypes.c_int
        self.library.mu_product.argtypes = [ptr, ptr, ptr, i64, ptr, integer]
        self.library.mu_product.restype = integer
        self.library.mu_residual.argtypes = [ptr, ptr, ptr, ptr, ptr, i64, ptr, integer]
        self.library.mu_residual.restype = integer
        self.calls = self.fast_calls = self.fallback_calls = self.launches = self.failures = 0
        self.native_sums = self.fast_sums = 0
        self.current_stream_id = None
        self.closed = False

    def _qualified(self, torch, z, boundary, branch, raw):
        if z.device.type != 'cuda' or z.device != boundary.device or z.device != raw.device or (z.device != branch.device):
            return False
        if z.dtype != torch.float32 or boundary.dtype != z.dtype or raw.dtype != z.dtype or (branch.dtype != torch.bool):
            return False
        if z.ndim != 2 or z.shape[1] != 512 or z.shape[0] <= 0 or (boundary.shape != z.shape) or (raw.shape != z.shape) or (branch.shape != (z.shape[0], 1)):
            return False
        return all((t.is_contiguous() and t.storage_offset() == 0 for t in (z, boundary, branch, raw)))

    def measure(self, z, boundary, branch, raw, original, *, return_intermediates=False):
        import torch
        if self.closed or self.failures:
            raise RuntimeError('A failed/closed measurement engine cannot continue')
        if not self._qualified(torch, z, boundary, branch, raw):
            result = original(z, boundary, branch, raw)
            self.calls += 1
            self.fallback_calls += 1
            self.native_sums += 2
            return (result, None) if return_intermediates else result
        if torch.cuda.current_device() != z.device.index:
            raise RuntimeError('Original current CUDA device differs from qualifiedinputs')
        stream = torch.cuda.current_stream(z.device).cuda_stream
        if self.current_stream_id is not None and self.current_stream_id != stream:
            raise RuntimeError('Original current CUDA stream changed')
        self.current_stream_id = stream
        product = torch.empty(z.shape, dtype=torch.float64, device=z.device)
        residual = torch.empty_like(product)
        if product.stride() != z.stride() or residual.stride() != product.stride() or product.storage_offset() != 0 or (residual.storage_offset() != 0) or product.data_ptr() % 256 or residual.data_ptr() % 256:
            raise RuntimeError('Qualified aligned native FP64 reduction descriptor required')
        try:
            code = self.library.mu_product(boundary.data_ptr(), raw.data_ptr(), product.data_ptr(), z.numel(), stream, z.device.index)
            if code:
                raise RuntimeError('Native CABI product launch failed: ' + str(code))
            self.launches += 1
            row_inner = product.sum(1, keepdim=True)
            self.native_sums += 1
            self.fast_sums += 1
            code = self.library.mu_residual(z.data_ptr(), product.data_ptr(), row_inner.data_ptr(), branch.data_ptr(), residual.data_ptr(), z.numel(), stream, z.device.index)
            if code:
                raise RuntimeError('Native CABI residual launch failed: ' + str(code))
            self.launches += 1
            result = residual.sum(0)
            self.native_sums += 1
            self.fast_sums += 1
            self.calls += 1
            self.fast_calls += 1
            if return_intermediates:
                return (result, (product, row_inner, residual))
            return result
        except BaseException:
            self.failures += 1
            raise

    def close(self):
        self.closed = True

def adapter_view(original, measurement):
    old_adapt = original.adapt

    def current_adapt(parent, rule):
        inherited = old_adapt(parent, rule)
        if rule != 'ordered':
            return inherited

        class ElementwiseMeasurement(inherited):

            def _measure(self, raw_lambda):
                if raw_lambda is None:
                    raise RuntimeError('Current raw boundary derivative required')
                self.structural_queries += 1
                function = super()._measure
                original_geometry = function.__func__.__globals__['geometry']
                return measurement.measure(self._z, self._boundary, self._unclamped, raw_lambda, original_geometry.channel_measurement)

            def close(self):
                try:
                    return super().close()
                finally:
                    measurement.close()
        return ElementwiseMeasurement
    old_spec = original.specification

    def current_spec(rule):
        result = dict(old_spec(rule))
        result['mu_elementwise_native_reduce'] = dict(specification(), enabled=rule == 'ordered')
        return result
    result = SimpleNamespace(**vars(original))
    result.adapt = current_adapt
    result.specification = current_spec
    return result
