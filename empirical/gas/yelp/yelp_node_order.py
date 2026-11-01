import time

def new_to_old_rows(old_permutation, new_permutation):
    import torch
    if old_permutation.device.type != 'cpu' or new_permutation.device.type != 'cpu' or old_permutation.dtype != torch.int64 or (new_permutation.dtype != torch.int64) or (old_permutation.ndim != 1) or (old_permutation.shape != new_permutation.shape):
        raise ValueError('Require two CPU int64 permutations of the same node population')
    inverse = torch.empty_like(old_permutation)
    inverse[old_permutation] = torch.arange(old_permutation.numel(), dtype=torch.int64)
    return inverse.index_select(0, new_permutation)

def migrate_history(history, rows, *, producer_drained, complete_current_theta_history):
    import torch
    if not producer_drained or not complete_current_theta_history:
        raise RuntimeError('History handoff requires drained, complete current-parameter history')
    emb = history.emb
    if emb.device.type != 'cpu' or emb.dtype != torch.float32 or emb.ndim != 2 or (rows.device.type != 'cpu') or (rows.dtype != torch.int64) or (rows.ndim != 1) or (rows.numel() != emb.shape[0]) or emb.requires_grad:
        raise ValueError('Require original CPU FP32 non-gradient full-node history')
    started = time.perf_counter()
    with torch.no_grad():
        relocated = emb.index_select(0, rows)
        emb.copy_(relocated)
    return {'nodes': emb.shape[0], 'width': emb.shape[1], 'temporary_payload_bytes': relocated.numel() * relocated.element_size(), 'host_seconds': time.perf_counter() - started, 'tensor_object_preserved': True, 'scope': 'Already computed history values only; caller owns drain, geometry, clocks and role readers'}
