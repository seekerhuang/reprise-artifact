def eligible(adj, deg=None, sort_indices=True):
    import numpy as np
    import scipy.sparse as sp
    if type(adj) is not sp.csr_matrix or deg is not None or sort_indices is not True:
        return 'type_or_explicit_degree_or_sort_mode'
    (p, j, v) = (adj.indptr, adj.indices, adj.data)
    if any((type(a) is not np.ndarray or a.ndim != 1 for a in (p, j, v))):
        return 'arrays'
    if any((a.dtype != np.dtype('int32') for a in (p, j, v))):
        return 'dtype'
    if any((not (a.flags.c_contiguous and a.flags.aligned) for a in (p, j, v))):
        return 'layout'
    (n, m) = adj.shape
    if n != m or n == 0 or p.size != n + 1 or (j.size != v.size):
        return 'shape'
    if p[0] != 0 or p[-1] != j.size or (p[1:] < p[:-1]).any():
        return 'rowptr'
    if (v != 1).any():
        return 'not_positive_unit_data'
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
    D = adj.sum(1).flatten()
    coefficients = np.asarray(1 / D).reshape(-1)
    values = np.repeat(coefficients, np.diff(adj.indptr))
    out = sp.csr_matrix((values, adj.indices.copy(), adj.indptr.copy()), shape=adj.shape)
    return (out, None)
