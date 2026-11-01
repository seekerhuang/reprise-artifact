def eligible(adj):
    import numpy as np
    import scipy.sparse as sp
    if type(adj) is not sp.coo_matrix:
        return 'type'
    if len(adj.shape) != 2:
        return 'shape'
    (r, c, v) = (adj.row, adj.col, adj.data)
    if any((type(a) is not np.ndarray or a.ndim != 1 for a in (r, c, v))) or not r.size == c.size == v.size:
        return 'arrays'
    if r.dtype not in (np.dtype('int32'), np.dtype('int64')) or c.dtype not in (np.dtype('int32'), np.dtype('int64')) or v.dtype not in (np.dtype('float32'), np.dtype('float64')):
        return 'dtype'
    if any((not (a.flags.c_contiguous and a.flags.aligned) for a in (r, c, v))):
        return 'layout'
    if r.size:
        if r.min() < 0 or c.min() < 0 or r.max() >= adj.shape[0] or (c.max() >= adj.shape[1]):
            return 'bounds'
        if not ((r[1:] > r[:-1]) | (r[1:] == r[:-1]) & (c[1:] > c[:-1])).all():
            return 'order_or_duplicate'
    return None

def convert(adj):
    import numpy as np
    import torch
    from graphsaint.minibatch import _coo_scipy2torch
    if eligible(adj) is not None:
        return _coo_scipy2torch(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(adj.shape), dtype=v.dtype, device=i.device, is_coalesced=True)
