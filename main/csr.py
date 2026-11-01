import torch

def select_csr_rows(matrix, rows):
    ptr = matrix.crow_indices()
    counts = ptr[rows + 1] - ptr[rows]
    output_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    size = int(output_ptr[-1])
    offsets = torch.arange(size, device=rows.device, dtype=ptr.dtype)
    offsets += torch.repeat_interleave(ptr[rows] - output_ptr[:-1], counts, output_size=size)
    return torch.sparse_csr_tensor(output_ptr, matrix.col_indices()[offsets], matrix.values()[offsets], size=(rows.numel(), matrix.shape[1]))
