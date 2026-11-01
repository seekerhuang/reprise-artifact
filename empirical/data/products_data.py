import torch
from torch_sparse import SparseTensor

def normalize_raw_csr(ptr, col, nodes, dtype=torch.float32):
    adjacency = SparseTensor(rowptr=ptr, col=col, sparse_sizes=(nodes, nodes), is_sorted=False)
    adjacency = adjacency.set_diag()
    degree = adjacency.sum(dim=1).to(dtype)
    inverse = degree.pow(-0.5)
    inverse[inverse == float('inf')] = 0
    adjacency = inverse.view(-1, 1) * adjacency * inverse.view(1, -1)
    return adjacency.to_torch_sparse_csr_tensor()
