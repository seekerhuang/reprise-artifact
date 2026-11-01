import math
BLOCK_ROWS = 4096

def transform_rows(node_counts, edge_counts, indptr):
    import numpy as np
    if node_counts.ndim != 1 or edge_counts.ndim != 1 or indptr.ndim != 1 or (len(indptr) != len(node_counts) + 1):
        raise ValueError('Original one-dimensional CSR/count schema required')
    if node_counts.dtype != np.float64 or edge_counts.dtype != np.float32 or (not np.issubdtype(indptr.dtype, np.integer)):
        raise ValueError('Pinned native count/CSR dtypes required')
    if int(indptr[0]) != 0 or int(indptr[-1]) != len(edge_counts):
        raise ValueError('Complete original CSR span required')
    if len(node_counts) and (node_counts[0] / edge_counts[:0]).dtype != np.float64:
        raise RuntimeError('Native scalar promotion differs from qualified NumPy')
    chunks = math.ceil(len(node_counts) / BLOCK_ROWS)
    for lo in range(0, len(node_counts), BLOCK_ROWS):
        hi = min(lo + BLOCK_ROWS, len(node_counts))
        begin = int(indptr[lo])
        end = int(indptr[hi])
        numerator = np.repeat(node_counts[lo:hi], np.diff(indptr[lo:hi + 1]))
        val = np.clip(numerator / edge_counts[begin:end], 0, 10000.0)
        val[np.isnan(val)] = 0.1
        edge_counts[begin:end] = val
    return dict(rows=len(node_counts), edges=len(edge_counts), chunks=chunks, largest_block_edges=None, original_node_dtype=str(node_counts.dtype), original_edge_dtype=str(edge_counts.dtype), arithmetic_dtype='float64', final_edge_dtype=str(edge_counts.dtype), NumPy_version=np.__version__)
