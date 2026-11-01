from dataclasses import dataclass
import torch
import torch.nn.functional as F
R75_deferred_setup_audits = []

def csr(matrix, device):
    (ptr, col, value) = matrix.csr()
    if value is None:
        raise ValueError('The normalized original operator must have explicit weights')
    return torch.sparse_csr_tensor(ptr.to(device), col.to(device), value.to(device), size=matrix.sparse_sizes(), device=device)

@dataclass
class Condition:
    key: int
    core_size: int
    node_ids: torch.Tensor
    matrix: torch.Tensor
    transpose: torch.Tensor
    label_matrix: torch.Tensor
    label_transpose: torch.Tensor
    first_image: torch.Tensor
    train_rows: torch.Tensor
    labels: torch.Tensor
    total_training_labels: int
    core_training_labels: int
    pull_size: int

@dataclass
class Geometry:
    conditions: dict
    data: object
    ptr: torch.Tensor
    matrix: torch.Tensor
    same_group: torch.Tensor
    first_image: torch.Tensor
    labels: torch.Tensor
    split_ids: dict
    record: dict

def build_geometry(author, data, ptr, device, *, training_parts=40, evaluation_parts=20):
    from torch_sparse import SparseTensor
    conditions = {}
    loader = author.package.LMCLoader(data, ptr, batch_size=training_parts, compensate=True, merge_cluster=True, score_func_name='const', shuffle=False, num_workers=0, log=False)
    ntrain = int(data.train_mask.sum())
    for sub in loader:
        (batch, core, nodes, offset, count, params) = sub
        if len(offset) != 1:
            raise ValueError('One premerged core group expected')
        key = int(offset[0])
        rows = batch.train_mask[:batch.adj_t.size(0)].nonzero().flatten()
        q = csr(batch.adj_t, device)
        qt = csr(batch.adj_t.t(), device)
        labelled = batch.adj_t.index_select(0, rows)
        conditions[key] = Condition(key, core, nodes.clone(), q, qt, csr(labelled, device), csr(labelled.t(), device), torch.sparse.mm(q, batch.x.to(device)), rows.to(device), batch.y[rows].to(device), ntrain, None, None)
        conditions[key]._R75_native_pull_mask_id = params['pull_mask_id']
    if len(conditions) != 2:
        raise ValueError('The declared training model has exactly two conditions')
    eval_ptr = ptr[::evaluation_parts]
    if int(eval_ptr[-1]) != data.num_nodes:
        eval_ptr = torch.cat([eval_ptr, ptr[-1:]])
    if len(eval_ptr) != 5:
        raise ValueError('Original inference must retain exactly four groups')
    group = torch.repeat_interleave(torch.arange(4), eval_ptr[1:] - eval_ptr[:-1])
    (row, col, value) = data.adj_t.coo()
    keep = group[row] == group[col]
    same = SparseTensor(row=row[keep], col=col[keep], value=value[keep], sparse_sizes=data.adj_t.sparse_sizes(), is_sorted=True)
    matrix = csr(data.adj_t, device)
    ids = {name: getattr(data, field).nonzero().flatten().to(device) for (name, field) in (('train', 'train_mask'), ('valid', 'val_mask'), ('test', 'test_mask'))}
    record = {}
    R75_deferred_setup_audits.append(lambda : record.update({'training_core_offsets': sorted(conditions), 'evaluation_ptr': eval_ptr.tolist(), 'nodes': data.num_nodes, 'normalized_edges': int(data.adj_t.nnz()), 'same_group_edges': int(keep.sum()), 'conditions': {str(key): {'core_nodes': c.core_size, 'local_nodes': len(c.node_ids), 'local_edges': c.matrix.values().numel(), 'receiver_training_labels': len(c.labels), 'core_training_labels': c.core_training_labels, 'native_pull_size': c.pull_size} for (key, c) in conditions.items()}, 'execution': 'OWN CSR/static QX and label-row final propagation; original independent LMC is unchanged', 'normalization': 'Original global symmetric GCN weights; neither Q nor P_same is renormalized'}))
    return Geometry(conditions, data, ptr, matrix, csr(same, device), torch.sparse.mm(matrix, data.x.to(device)), data.y.to(device), ids, record)

def conditional_loss(logits, condition):
    return F.cross_entropy(logits, condition.labels, reduction='sum') * (2.0 / condition.total_training_labels)

@torch.no_grad()
def four_group_logits(model, geometry):
    model.eval()
    h = F.linear(geometry.first_image, model.convs[0].lin.weight, model.convs[0].bias)
    for layer in (0, 1):
        inside = model.bns[layer].bn_ib(h).relu()
        outside = model.bns[layer].bn_ob(h).relu()
        mixed = torch.sparse.mm(geometry.matrix, outside)
        mixed = mixed + torch.sparse.mm(geometry.same_group, inside - outside)
        h = F.linear(mixed, model.convs[layer + 1].lin.weight, model.convs[layer + 1].bias)
    return h
