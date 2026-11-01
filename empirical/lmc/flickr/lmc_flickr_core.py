import torch
import torch.nn.functional as F

def csr(matrix, device):
    (ptr, col, value) = matrix.csr()
    if value is None:
        raise ValueError('Original normalized weights required')
    return torch.sparse_csr_tensor(ptr.to(device), col.to(device), value.to(device), size=matrix.sparse_sizes(), device=device)

def conditional_loss(logits, condition):
    return F.cross_entropy(logits[condition.train_rows], condition.labels, reduction='sum') * (2.0 / condition.total_training_labels)

@torch.no_grad()
def four_group_logits(model, geometry):
    model.eval()
    h = F.linear(geometry.eval_first_image, model.convs[0].lin.weight, model.convs[0].bias)
    inside = model.bns[0].bn_ib(h).relu()
    outside = model.bns[0].bn_ob(h).relu()
    mixed = torch.sparse.mm(geometry.matrix, outside)
    mixed = mixed + torch.sparse.mm(geometry.same_group, inside - outside)
    return F.linear(mixed, model.convs[1].lin.weight, model.convs[1].bias)
