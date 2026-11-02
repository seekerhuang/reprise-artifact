import numpy as np
import scipy.sparse as sp


def common_gcn_norm(adj, deg=None, sort_indices=True):
    if adj.shape[0] != adj.shape[1]:
        raise ValueError('Common model needs a square sampled graph')
    if np.count_nonzero(adj.diagonal()):
        raise ValueError('Raw sampler edges must exclude self-loops')
    degree = np.asarray(adj.sum(1)).reshape(-1) if deg is None else np.asarray(deg).reshape(-1)
    if len(degree) != adj.shape[0] or np.any(degree < 0):
        raise ValueError('Full-population degree shape/value differs')
    degree_plus = degree.astype(np.float32) + np.float32(1.0)
    inverse = np.reciprocal(np.sqrt(degree_plus))
    scale = sp.diags(inverse, format='csr')
    value = scale.dot(adj.astype(np.float32)).dot(scale)
    value = (value + sp.diags(np.reciprocal(degree_plus), format='csr')).tocsr()
    if sort_indices:
        value.sort_indices()
    return value
