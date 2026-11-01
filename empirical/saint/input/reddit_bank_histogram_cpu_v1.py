from pathlib import Path
import ctypes
HERE = Path(__file__).resolve().parent
LIBRARY = HERE / 'libreddit_bank_histogram_cpu_v1.so'
VERSION = 'Reddit-native-fixed-bank-direct-int32-CPU-histogram-v1'
_COUNTER = None

class Counter:

    def __init__(self, path=LIBRARY):
        self.library = ctypes.CDLL(str(path))
        self.library.reddit_bank_histogram_abi_version.restype = ctypes.c_int
        if self.library.reddit_bank_histogram_abi_version() != 1:
            raise RuntimeError('Qualified native CPU integer histogram ABI1 required')
        (pointer, count) = (ctypes.c_void_p, ctypes.c_int64)
        self.library.reddit_bank_histogram_count.argtypes = [pointer, count, pointer, count, pointer, count, pointer, count, ctypes.c_int32]
        self.library.reddit_bank_histogram_count.restype = ctypes.c_int

    def count(self, node_ids, edge_ids, node_counts, edge_counts, graph_count):
        code = self.library.reddit_bank_histogram_count(node_ids.ctypes.data, node_ids.size, edge_ids.ctypes.data, edge_ids.size, node_counts.ctypes.data, node_counts.size, edge_counts.ctypes.data, edge_counts.size, graph_count)
        if code == 3:
            raise ValueError('Original bank histogram identity is negative')
        if code:
            raise RuntimeError('Original bank histogram domain/count bound failed: ' + str(code))

def count_flat(node_ids, edge_ids, node_counts, edge_counts, *, graph_count, _fallback=None):
    import numpy as np
    global _COUNTER
    if graph_count != 1000:
        raise ValueError('Only the fixed original1000-graph warm pool is supported')
    if any((type(a) not in (np.ndarray, np.memmap) or a.ndim != 1 or a.dtype != np.int32 or (not a.flags.c_contiguous) or (not a.flags.aligned) for a in (node_ids, edge_ids))):
        if _fallback is not None:
            return _fallback(node_ids, edge_ids, node_counts, edge_counts, graph_count=graph_count)
        raise ValueError('Original aligned contiguous int32 bank identities required')
    if type(node_counts) is not np.ndarray or type(edge_counts) is not np.ndarray or node_counts.ndim != 1 or (edge_counts.ndim != 1) or (node_counts.dtype != np.float64) or (edge_counts.dtype != np.float32) or any((not a.flags.c_contiguous or not a.flags.writeable or (not a.flags.aligned) for a in (node_counts, edge_counts))):
        if _fallback is not None:
            return _fallback(node_ids, edge_ids, node_counts, edge_counts, graph_count=graph_count)
        raise ValueError('Fresh native writable aligned count arrays required')
    if np.shares_memory(node_counts, edge_counts) or any((np.shares_memory(out, inp) for out in (node_counts, edge_counts) for inp in (node_ids, edge_ids))):
        if _fallback is not None:
            return _fallback(node_ids, edge_ids, node_counts, edge_counts, graph_count=graph_count)
        raise ValueError('Independent original native count-array storage required')
    if _COUNTER is None:
        _COUNTER = Counter()
    _COUNTER.count(node_ids, edge_ids, node_counts, edge_counts, graph_count)
    return None
