from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_sparse import SparseTensor


DATASETS = ('arxiv', 'products', 'reddit2', 'flickr', 'yelp', 'ppi')


def download(name, root):
    root = Path(root)
    if name in ('arxiv', 'products'):
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            dataset = PygNodePropPredDataset(name='ogbn-' + name, root=str(root / 'raw'))
        data = dataset[0]
        split = dataset.get_idx_split()
        for key, field in [('train', 'train_mask'), ('valid', 'val_mask'), ('test', 'test_mask')]:
            data[field] = torch.zeros(data.num_nodes, dtype=torch.bool)
            data[field][split[key]] = True
        data.y = data.y.flatten()
        if name == 'arxiv':
            data.edge_index = torch.cat((data.edge_index, data.edge_index.flip(0)), dim=1)
        else:
            edges = data.edge_index[:, ::2]
            data.edge_index = torch.cat((edges, edges.flip(0)), dim=1)
    elif name == 'ppi':
        from torch_geometric.datasets import PPI
        result = {}
        for split in ('train', 'val', 'test'):
            dataset = PPI(str(root / 'raw' / name), split=split)
            data = Batch.from_data_list(list(dataset))
            data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            data.val_mask = data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            result[split] = standard(data)
        return result
    else:
        from torch_geometric.datasets import Flickr, Reddit2, Yelp
        cls = {'flickr': Flickr, 'reddit2': Reddit2, 'yelp': Yelp}[name]
        data = cls(str(root / 'raw' / name))[0]
    return standard(data, name)


def standard(data, name=None):
    edges = data.edge_index
    matrix = SparseTensor(row=edges[1], col=edges[0], sparse_sizes=(data.num_nodes, data.num_nodes))
    ptr, col, _ = matrix.csr()
    result = dict(x=data.x.float(), y=data.y, rowptr=ptr, col=col,
                  train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
    if name in ('arxiv', 'products'):
        row, column = (edges[1], edges[0]) if name == 'arxiv' else (edges[0], edges[1])
        order = np.argsort(row.numpy(), kind='stable')
        rows = row.numpy()[order]
        result['sample_rowptr'] = torch.from_numpy(np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=data.num_nodes)))))
        result['sample_col'] = column[torch.from_numpy(order)].contiguous()
    return result


def load(name, root, split=None):
    path = Path(root) / (name + '.pt')
    data = torch.load(path, map_location='cpu', weights_only=True)
    return data[split] if split else data


def adjacency(data, *, symmetric=False, loops=False, normalization=None, device='cpu'):
    n = len(data['x'])
    matrix = SparseTensor(rowptr=data['rowptr'], col=data['col'], sparse_sizes=(n, n))
    if symmetric:
        matrix = matrix.to_symmetric()
    if loops:
        matrix = matrix.set_diag()
    if normalization == 'gcn':
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        matrix = gcn_norm(matrix, add_self_loops=False)
    elif normalization == 'mean':
        matrix = matrix.fill_value(1.0)
        matrix = matrix / matrix.sum(1).clamp_min(1).view(-1, 1)
    return matrix.to(device)


def splits(data, device):
    return {key: data[field].nonzero().flatten().to(device)
            for key, field in [('train', 'train_mask'), ('valid', 'val_mask'), ('test', 'test_mask')]}


def partition(data, parts, symmetric=False):
    import pymetis
    matrix = adjacency(data, symmetric=symmetric)
    rowptr, col, _ = matrix.csr()
    dtype = pymetis.zero_copy_dtype()
    result = pymetis.part_graph(parts, adjacency=pymetis.CSRAdjacency(
        np.asarray(rowptr.numpy(), dtype=dtype), np.asarray(col.numpy(), dtype=dtype)),
        recursive=False, options=pymetis.Options(), vweights=None, eweights=None, tpwgts=None)
    cluster = torch.from_numpy(np.asarray(result.vertex_part, dtype=np.int64).copy())
    cluster, permutation = cluster.sort()
    ptr = torch.ops.torch_sparse.ind2ptr(cluster, parts)
    return dict(permutation=permutation, ptr=ptr)


def scipy_csr(data):
    import scipy.sparse as sp
    ptr, col = data['rowptr'].numpy(), data['col'].numpy()
    return sp.csr_matrix((np.ones(len(col), dtype=np.float32), col, ptr),
                         shape=(len(data['x']), len(data['x'])))
