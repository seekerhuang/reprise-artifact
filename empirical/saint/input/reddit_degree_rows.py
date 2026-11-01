def eligible(adj, deg, sort_indices=True):
    import numpy as np
    import scipy.sparse as sp
    if type(adj) is not sp.csr_matrix or sort_indices is not True:
        return 'type_or_sort_mode'
    (p, j, v) = (adj.indptr, adj.indices, adj.data)
    if any((type(a) is not np.ndarray or a.ndim != 1 for a in (p, j, v))) or type(deg) is not np.ndarray or deg.ndim != 1:
        return 'arrays'
    if p.dtype != np.dtype('int32') or j.dtype != np.dtype('int32') or v.dtype != np.dtype('float32') or (deg.dtype != np.dtype('int64')):
        return 'dtype'
    if any((not (a.flags.c_contiguous and a.flags.aligned) for a in (p, j, v, deg))):
        return 'layout'
    (n, m) = adj.shape
    if n != m:
        return 'nonsquare'
    if p.size != n + 1 or deg.size != n or j.size != v.size or (j.size > 2147483647) or (max(n, m) > 2147483647):
        return 'shape'
    if p[0] != 0 or p[-1] != j.size or (p[1:] < p[:-1]).any():
        return 'rowptr'
    if (deg < 0).any() or ((deg == 0) & (p[1:] != p[:-1])).any() or (not np.isfinite(v).all()) or (v <= 0).any():
        return 'nonpositive_or_nonfinite'
    if j.size and (j.min() < 0 or j.max() >= m):
        return 'bounds'
    bad = j[1:] <= j[:-1]
    starts = p[1:-1]
    starts = starts[(starts > 0) & (starts < j.size)]
    bad[starts - 1] = False
    if bad.any():
        return 'order_or_duplicate'
    return None

def apply(original, adj, deg=None, sort_indices=True):
    import numpy as np
    import scipy.sparse as sp
    reason = eligible(adj, deg, sort_indices)
    if reason is not None:
        return (original(adj, deg=deg, sort_indices=sort_indices), reason)
    values = np.repeat(1 / deg, np.diff(adj.indptr))
    np.multiply(values, adj.data, out=values)
    result = sp.csr_matrix((values, adj.indices.copy(), adj.indptr.copy()), shape=adj.shape)
    return (result, None)
