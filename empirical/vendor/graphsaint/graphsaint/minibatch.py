from graphsaint.globals import *
from graphsaint.graph_samplers import rw_sampling
from graphsaint.norm_aggr import *
from empirical.saint.data import adj_norm
from empirical.saint.input.row_normalizer import transform_rows
import torch
import scipy.sparse as sp
import scipy
import numpy as np
import time
import math

def _coo_scipy2torch(adj):
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))

class Minibatch:

    def __init__(self, adj_full_norm, adj_train, role, train_params, cpu_eval=False):
        self.use_cuda = args_global.gpu >= 0
        if cpu_eval:
            self.use_cuda = False
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])
        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())
        self.adj_train = adj_train
        if self.use_cuda:
            self.adj_full_norm = self.adj_full_norm.cuda()
        self.node_subgraph = None
        self.batch_num = -1
        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        _denom = len(self.node_train) + len(self.node_val) + len(self.node_test)
        self.norm_loss_test[self.node_train] = 1.0 / _denom
        self.norm_loss_test[self.node_val] = 1.0 / _denom
        self.norm_loss_test[self.node_test] = 1.0 / _denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.cuda()
        self.norm_aggr_train = np.zeros(self.adj_train.size)
        self.sample_coverage = train_params['sample_coverage']
        self.reddit = train_params.get('dataset') == 'reddit'
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()

    def set_sampler(self, train_phases):
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root'] * train_phases['depth']
            self.graph_sampler = rw_sampling(self.adj_train, self.node_train, self.size_subg_budget, int(train_phases['num_root']), int(train_phases['depth']))
        else:
            raise ValueError('rw sampler required')
        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)
        tot_sampled_nodes = 0
        while True:
            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break
        print()
        num_subg = len(self.subgraphs_remaining_nodes)
        if self.reddit:
            from empirical.saint.input.reddit_bank_histogram_cpu_v1 import count_flat
            count_flat(np.concatenate(self.subgraphs_remaining_nodes),
                       np.concatenate(self.subgraphs_remaining_edge_index),
                       self.norm_loss_train, self.norm_aggr_train, graph_count=num_subg)
        else:
            for i in range(num_subg):
                self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
                self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        transform_rows(self.norm_loss_train, self.norm_aggr_train, self.adj_train.indptr)
        self.norm_loss_train[np.where(self.norm_loss_train == 0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.cuda()

    def par_graph_sample(self, phase):
        t0 = time.time()
        (_indptr, _indices, _data, _v, _edge_index) = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end='\r')
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def one_batch(self, mode='train'):
        if mode in ['val', 'test', 'valtest']:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])
            adj = self.adj_full_norm
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')
                print()
            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix((self.subgraphs_remaining_data.pop(), self.subgraphs_remaining_indices.pop(), self.subgraphs_remaining_indptr.pop()), shape=(self.size_subgraph, self.size_subgraph))
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=args_global.num_cpu_core)
            if self.reddit:
                from empirical.saint.input import reddit_degree_rows, reddit_canonical_coo
                adj, _ = reddit_degree_rows.apply(adj_norm, adj, deg=self.deg_train[self.node_subgraph])
                adj = reddit_canonical_coo.convert(adj.tocoo())
            else:
                adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])
                adj = _coo_scipy2torch(adj.tocoo())
            if self.use_cuda:
                adj = adj.cuda()
            self.batch_num += 1
        norm_loss = self.norm_loss_test if mode in ['val', 'test', 'valtest'] else self.norm_loss_train
        nodes = self.node_subgraph
        if mode == 'train' and not self.reddit:
            nodes = torch.as_tensor(nodes, dtype=torch.long, device=norm_loss.device)
        norm_loss = norm_loss[nodes]
        return (nodes, adj, norm_loss)

    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]
