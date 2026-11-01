import time
import torch
from types import SimpleNamespace
from empirical.lmc.flickr.lmc_flickr_core import csr

def build_geometry(author, data, ptr, device, *, training_parts=12, evaluation_parts=6):
    from torch_sparse import SparseTensor

    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
    sync()
    phase_start = time.perf_counter()
    loader = author.package.LMCLoader(data, ptr, batch_size=training_parts, compensate=True, merge_cluster=True, score_func_name='const', shuffle=False, num_workers=0, log=False)
    conditions = {}
    total_train = int(data.train_mask.sum())
    for (batch, core_size, nodes, offset, count, params) in loader:
        if len(offset) != 1 or batch.adj_t.size(0) != batch.adj_t.size(1):
            raise ValueError('Original two premerged square conditional graphs required')
        key = int(offset[0])
        rows = batch.train_mask[:batch.adj_t.size(0)].nonzero().flatten()
        conditions[key] = SimpleNamespace(key=key, core_size=core_size, node_ids=nodes.clone(), matrix=batch.adj_t.to(device), features=batch.x.to(device), labels=batch.y[rows].to(device), train_rows=rows.to(device), total_training_labels=total_train, core_training_labels=int(batch.train_mask[:core_size].sum()), pull_size=int(params['pull_mask_id'].numel()))
    if len(conditions) != 2:
        raise ValueError('Two original training conditions required')
    sync()
    training_seconds = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    matrix = csr(data.adj_t, device)
    eval_first_image = torch.sparse.mm(matrix, data.x.to(device))
    sync()
    delivery_seconds = time.perf_counter() - phase_start
    eval_ptr = ptr[::evaluation_parts]
    if int(eval_ptr[-1]) != data.num_nodes:
        eval_ptr = torch.cat([eval_ptr, ptr[-1:]])
    if len(eval_ptr) != 5:
        raise ValueError('Four original inference groups required')
    group = torch.repeat_interleave(torch.arange(4), eval_ptr[1:] - eval_ptr[:-1])
    (row, col, value) = data.adj_t.coo()
    same = group[row] == group[col]
    same_group = SparseTensor(row=row[same], col=col[same], value=value[same], sparse_sizes=data.adj_t.sparse_sizes(), is_sorted=True)
    ids = {name: getattr(data, field).nonzero().flatten().to(device) for (name, field) in (('train', 'train_mask'), ('valid', 'val_mask'), ('test', 'test_mask'))}
    record = None
    result = SimpleNamespace(conditions=conditions, data=data, ptr=ptr, matrix=matrix, same_group=csr(same_group, device), eval_first_image=eval_first_image, labels=data.y.to(device), split_ids=ids, record=record)
    result.training_geometry_seconds = training_seconds
    result.delivery_geometry_seconds = delivery_seconds
    record = {'training_core_offsets': sorted(conditions), 'evaluation_ptr': eval_ptr.tolist(), 'nodes': data.num_nodes, 'normalized_edges': int(data.adj_t.nnz()), 'same_group_edges': int(same.sum()), 'conditions': {str(key): {'core_nodes': c.core_size, 'local_nodes': len(c.node_ids), 'local_edges': c.matrix.nnz(), 'receiver_training_labels': len(c.labels), 'core_training_labels': c.core_training_labels, 'native_pull_size': c.pull_size} for (key, c) in conditions.items()}, 'training_input_rule': 'No staticQX: each currentquery applies original inputdropout.3 to raw localfeatures then currentQ propagation', 'inference_rule': 'Four groups after saved native label permutation; original global weights in P and P_same, no renormalization'}
    result.record = record
    return result
