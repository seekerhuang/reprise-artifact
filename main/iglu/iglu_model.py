import torch
import torch.nn.functional as F
import numpy as np
from main.graph_operator_reuse import ExactPropagation
TF_NAMES = ('R', 'R_1', 'w', 'b1', 'b2', 'b3', 'offset', 'scale', 'Layer2/offset', 'Layer2/scale')
LOCAL_NAMES = ('q', 'r', 'w', 'b1', 'b2', 'b3', 'offset2', 'scale2', 'offset1', 'scale1')
SHAPES = ((128, 256), (256, 256), (256, 40), (256,), (256,), (40,), (1, 256), (1, 256), (1, 256), (1, 256))
GROUP_NAMES = {'Q': ('q', 'b1', 'offset1', 'scale1'), 'R': ('r', 'b2', 'offset2', 'scale2'), 'w': ('w', 'b3')}

class Target(torch.nn.Module):

    def __init__(self, tensors, device='cpu'):
        super().__init__()
        if set(tensors) != set(TF_NAMES):
            raise ValueError('Use exactly the ten original target tensors')
        for (original, local, shape) in zip(TF_NAMES, LOCAL_NAMES, SHAPES):
            value = torch.as_tensor(tensors[original], device=device)
            if value.dtype != torch.float32 or tuple(value.shape) != shape:
                raise ValueError('Original FP32 parameter shape differs: ' + original)
            self.register_parameter(local, torch.nn.Parameter(value.clone()))

    def original_tensors(self):
        return {a: getattr(self, b) for (a, b) in zip(TF_NAMES, LOCAL_NAMES)}

    def groups(self):
        return {key: [getattr(self, name) for name in names] for (key, names) in GROUP_NAMES.items()}

    def norm_relu(self, value, layer):
        mean = value.mean(1, keepdim=True)
        variance = (value - mean.detach()).square().mean(1, keepdim=True)
        inv = torch.rsqrt(variance + 1e-09) * getattr(self, 'scale' + str(layer))
        return F.relu(value * inv + (getattr(self, 'offset' + str(layer)) - mean * inv))

    def forward(self, first_image, matrix, transpose, *, label_matrix=None, label_transpose=None, response=None):
        hidden = self.norm_relu(first_image @ self.q + self.b1, 1)
        mixed = ExactPropagation.apply(hidden, matrix, transpose)
        hidden = self.norm_relu(mixed @ self.r + self.b2, 2)
        final_matrix = matrix if label_matrix is None else label_matrix
        final_transpose = transpose if label_transpose is None else label_transpose
        return ExactPropagation.apply(hidden @ self.w, final_matrix, final_transpose) + self.b3

def original_normalize(adj):
    import scipy.sparse as sp
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def csr(matrix, device):
    matrix = matrix.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(matrix.indptr.astype(np.int64)), torch.from_numpy(matrix.indices.astype(np.int64)), torch.from_numpy(matrix.data.astype(np.float32)), size=matrix.shape, device=device)
