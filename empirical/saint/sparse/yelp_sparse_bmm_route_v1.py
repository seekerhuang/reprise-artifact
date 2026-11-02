from contextlib import contextmanager
from types import MethodType
import os
import sys

WORKSPACE = ':4096:8'

def configure_before_torch():
    if 'torch' in sys.modules:
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before any Torch import')
    current = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if current not in (None, WORKSPACE):
        raise RuntimeError('Different pre-existing cuBLAS workspace configuration')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE

def public_singleton_mm(adj, dense):
    import torch
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError('Strict deterministic mode must cover forward AND backward')
    if (adj.layout != torch.sparse_coo or adj.dim() != 2 or adj.sparse_dim() != 2
            or adj.dense_dim() != 0 or adj.requires_grad or not adj.is_cuda
            or not dense.is_cuda or dense.dim() != 2 or dense.layout != torch.strided
            or adj.device != dense.device or adj.dtype != torch.float32
            or dense.dtype != torch.float32 or adj.shape[1] != dense.shape[0]):
        raise ValueError('Locked 2D FP32 CUDA COO (fixed values) times dense required')
    
    
    
    return torch.bmm(adj.unsqueeze(0), dense.unsqueeze(0)).squeeze(0).contiguous()

@contextmanager
def strict_determinism(torch):
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

@contextmanager
def install(model):

    if model.order_layer != [1, 1, 0] or not model.sigmoid_loss:
        raise ValueError('Original Yelp order1-1-0 sigmoid model required')
    saved = []
    counts = {str(index): 0 for index in range(len(model.aggregators))}
    try:
        for index, layer in enumerate(model.aggregators):
            present = '_spmm' in layer.__dict__
            saved.append((layer, present, layer.__dict__.get('_spmm')))

            def replacement(self, adj, dense, _index=index):
                result = public_singleton_mm(adj, dense)
                counts[str(_index)] += 1
                return result

            layer._spmm = MethodType(replacement, layer)
        yield counts
    finally:
        for layer, present, value in reversed(saved):
            if present:
                layer._spmm = value
            else:
                delattr(layer, '_spmm')

def specification():
    return dict(route='torch.bmm(A.unsqueeze(0),X.unsqueeze(0)).squeeze(0).contiguous()',
        deterministic_algorithms=True, warn_only=False,
        CUBLAS_WORKSPACE_CONFIG=WORKSPACE, workspace_config_before_Torch=True,
        cache_3D_adjacency=False, explicit_coalesce_or_dtype_conversion=False,
        precision='Original FP32, matmul TF32 false/highest; cuDNN flags not assigned',
        autograd='Public bmm native dense-input backward uses sparse transpose.bmm under the same strict flag',
        native_2D_mm_called_by_candidate=False)
