import torch

def make_graph(matrix):
    import dgl
    graph = dgl.graph(('csc', (matrix.crow_indices(), matrix.col_indices(), torch.empty(0, dtype=torch.int64, device=matrix.device))), num_nodes=matrix.shape[0], device=matrix.device)
    graph.edata['weight'] = matrix.values()
    return graph.formats('csc')

def spmm(graph, features):
    import dgl.ops
    return dgl.ops.gspmm(graph, 'mul', 'sum', features, graph.edata['weight'])
