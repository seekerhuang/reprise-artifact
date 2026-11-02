import torch

from empirical.data.datasets import load


def read(args, dataset):
    data = load(dataset, args.data_dir)
    if dataset == 'reddit2':
        data['x'] = (data['x'] - data['x'].mean(0)) / data['x'].std(0)
    return data


def initial(args):
    path = args.data_dir / 'baselines' / 'initial' / args.config / f'{args.initialization}.pt'
    return torch.load(path, map_location='cpu', weights_only=False)


def dgl_graph(data, *, self_loops=False):
    import dgl
    graph = dgl.graph(('csc', (data['sample_rowptr'], data['sample_col'], torch.empty(0, dtype=torch.int64))),
                      num_nodes=len(data['x']))
    if self_loops:
        graph = dgl.add_self_loop(graph)
    graph.ndata['features'] = data['x']
    graph.ndata['labels'] = data['y']
    for name in ('train_mask', 'val_mask', 'test_mask'):
        graph.ndata[name] = data[name]
    return graph
