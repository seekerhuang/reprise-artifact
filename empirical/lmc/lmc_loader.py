from typing import NamedTuple, List, Tuple, Dict
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.data import Data
import torch_geometric_autoscale
Relabel2hop = torch.classes.torch_geometric_autoscale.Relabel2hop

def const(x):
    return torch.ones_like(x)

class SubData(NamedTuple):
    data: Data
    batch_size: int
    n_id: Tensor
    offset: Tensor
    count: Tensor
    lmc_params: Dict

    def to(self, *args, **kwargs):
        return SubData(self.data.to(*args, **kwargs), self.batch_size, self.n_id, self.offset, self.count, self.lmc_params)

class LMCLoader(DataLoader):

    def __init__(self, data: Data, ptr: Tensor, batch_size: int=1, compensate: bool=True, rm1hop: bool=True, merge_cluster: bool=True, score_func_name: str='linear', bipartite: bool=True, log: bool=True, **kwargs):
        if merge_cluster:
            ptr = ptr[::batch_size]
            if int(ptr[-1]) != data.num_nodes:
                ptr = torch.cat([ptr, torch.tensor([data.num_nodes])], dim=0)
            batch_size = 1
        self.data = data
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log
        self.compensate = compensate
        self.merge_cluster = merge_cluster
        self.score_func_name = score_func_name
        self.deg_subgraph = torch.zeros(data.num_nodes, dtype=torch.int64)
        self.deg = self.data.adj_t.storage.rowcount()
        assert merge_cluster == 0 or batch_size != 0
        (rowptr, col, value) = self.data.adj_t.csr()
        if value is None:
            value = torch.ones_like(col)
            self.value_none = True
        else:
            self.value_none = False
        self.relabel_2hop = Relabel2hop(rowptr, col, value)
        n_id = torch.arange(data.num_nodes)
        batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
        batches = [(i, batches[i]) for i in range(len(batches))]
        if batch_size > 1:
            super().__init__(batches, batch_size=batch_size, collate_fn=self.compute_subgraph, **kwargs)
        else:
            if log:
                t = time.perf_counter()
                print('Pre-processing subgraphs...', end=' ', flush=True)
            data_list = list(DataLoader(batches, collate_fn=self.compute_subgraph, batch_size=batch_size, **kwargs))
            if log:
                print(f'Done! [{time.perf_counter() - t:.2f}s]')
            super().__init__(data_list, batch_size=batch_size, collate_fn=lambda x: x[0], **kwargs)

    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        (batch_ids, n_ids) = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)
        batch_size = n_id.numel()
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset)
        (rowptr, col, value, n_id, lmc_params) = self.lmc_sampler(n_id, self.bipartite)
        lmc_params['num_cluster'] = (len(self.ptr) - 1) / len(batch_ids)
        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value, sparse_sizes=(rowptr.numel() - 1, n_id.numel()), is_sorted=True)
        data = self.data.__class__(adj_t=adj_t)
        for (k, v) in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)
        return SubData(data, batch_size, n_id, offset, count, lmc_params)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def lmc_sampler(self, n_id, bipartite):
        (value_1hop, n_id_result) = self.relabel_2hop.relabel_one_hop(n_id, bipartite)
        batch_size = n_id.numel()
        n_id_outofbatch = n_id_result[batch_size:]
        (rowptr_result, col_result, value_out) = self.relabel_2hop.relabel_one_hop_compensation_rm2hop(n_id_outofbatch, bipartite)
        if self.value_none:
            value_result = None
        else:
            value_result = torch.cat([value_1hop, value_out], dim=0)
            value_result = value_result[:rowptr_result[-1] - rowptr_result[0]]
        deg = rowptr_result[1:] - rowptr_result[:-1]
        alpha = deg[batch_size:] / self.deg[n_id_result[batch_size:]]
        return (rowptr_result, col_result, value_result, n_id_result, {'onehop_size': n_id_result.numel(), 'pull_mask_id': torch.where(alpha != 1)[0] + batch_size, 'pull_nid': n_id_result[batch_size:][alpha != 1], 'alpha': eval(self.score_func_name)(alpha[alpha != 1])})
