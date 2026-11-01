def balanced_partition(num_nodes, num_parts=40, seed=123):
    import torch
    if type(num_nodes) is not int or num_nodes < num_parts or num_parts != 40 or (seed != 123):
        raise ValueError('Positive complete node population, exactly40 parts and fixed seed123 required')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    permutation = torch.randperm(num_nodes, generator=generator, device='cpu', dtype=torch.long)
    ptr = torch.tensor([i * num_nodes // num_parts for i in range(num_parts + 1)], dtype=torch.long)
    return (permutation, ptr)
