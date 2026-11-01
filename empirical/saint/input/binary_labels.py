import ctypes
from pathlib import Path
LIBRARY = Path(__file__).resolve().parent / 'libbinary_labels_v1.so'

class Decoder:

    def __init__(self, library=LIBRARY):
        self.library = ctypes.CDLL(str(library))
        self.library.binary_labels_abi_version.restype = ctypes.c_int
        if self.library.binary_labels_abi_version() != 1:
            raise RuntimeError('Exact CPU binary-label ABI1 required')
        self.library.binary_labels_decode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.library.binary_labels_decode.restype = ctypes.c_int
        self.calls = self.fast = self.fallback = 0

    def attempt(self, raw, rows, columns):
        import numpy as np
        if type(raw) is not bytes or rows <= 0 or columns <= 0:
            raise ValueError('Positive exact dimensions and raw bytes required')
        output = np.empty((rows, columns), dtype=np.float64)
        seen = np.zeros(rows, dtype=np.uint8)
        error = ctypes.c_size_t(0)
        code = self.library.binary_labels_decode(raw, len(raw), rows, columns, output.ctypes.data, seen.ctypes.data, ctypes.byref(error))
        if code:
            return (None, dict(code=code, error_at=error.value))
        return (output, dict(code=0, error_at=error.value))
