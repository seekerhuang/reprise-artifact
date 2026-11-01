from graphsaint.globals import *
import numpy as np
import math
from math import ceil
import graphsaint.cython_sampler as cy

class GraphSampler:

    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)

    def preproc(self, **kwargs):
        pass

    def par_sample(self, stage, **kwargs):
        return self.cy_sampler.par_sample()

class rw_sampling(GraphSampler):

    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.cy_sampler = cy.RW(self.adj_train.indptr, self.adj_train.indices, self.node_train, NUM_PAR_SAMPLER, SAMPLES_PER_PROC, self.size_root, self.size_depth)

    def preproc(self, **kwargs):
        pass
