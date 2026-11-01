import torch


def prepare(raw, device):
    import dgl
    rows = torch.repeat_interleave(torch.arange(len(raw['x'])), raw['rowptr'][1:] - raw['rowptr'][:-1])
    graph = dgl.graph((raw['col'], rows), num_nodes=len(raw['x']))
    graph = dgl.to_bidirected(graph)
    graph = graph.remove_self_loop().add_self_loop()
    weight = dgl.nn.EdgeWeightNorm(norm='both')(graph, torch.ones(graph.num_edges()))
    ptr, col, eid = graph.adj_tensors('csc')
    matrix = torch.sparse_csr_tensor(ptr.long().to(device), col.long().to(device),
                                     weight[eid.long()].to(device),
                                     size=(len(raw['x']), len(raw['x'])), device=device)
    features = raw['x']
    mean = features.mean(dim=0)
    inverse = 1.0 / features.std(dim=0)
    inverse[inverse.isinf()] = 0
    return ((features - mean) * inverse).to(device), matrix
