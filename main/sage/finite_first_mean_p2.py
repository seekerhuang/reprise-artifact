import torch.nn.functional as F
from main.sage.sage_boundary import FrozenMeanP2
from main.sage.sampled_mean_p2 import block_mean

class FiniteFirstMeanP2(FrozenMeanP2):

    def __init__(self, model, optimizer, dgl, graph, features, labels, *, first_fanout=10, fanouts=(10, 10), sampler='neighbor'):
        if len(model.convs) != 3 or len(model.bns) != 2 or len(fanouts) != 2:
            raise ValueError('Require original Fresh19 SAGE3/BN')
        (self.model, self.optimizer, self.dgl) = (model, optimizer, dgl)
        (self.graph, self.features, self.all_labels) = (graph, features, labels)
        if sampler == 'neighbor':
            self.sampler = dgl.dataloading.NeighborSampler([first_fanout, *fanouts], fused=False)
        elif sampler == 'labor':
            self.sampler = dgl.dataloading.LaborSampler([first_fanout, *fanouts], importance_sampling=0, layer_dependency=False, batch_dependency=1)
        else:
            raise ValueError(sampler)
        self.actual_updates = 0
        self.work = dict(prefix_forwards=0, prefix_VJPs=0, head_queries=0, head_updates=0, body_updates=0, supervised_presentations=0, boundary_payload_peak_bytes=None, input_root_presentations=None, sampled_raw_input_nodes=None, sampled_first_hidden_nodes=None, sampled_second_hidden_nodes=None, sampled_first_edges=None, sampled_middle_edges=None, sampled_last_edges=None, context_samples=0)
        self.context = None

    def set_context(self, roots):
        (inputs, outputs, blocks) = self.sampler.sample_blocks(self.graph, roots)
        self.context = (inputs, outputs, blocks)
        self.labels = self.all_labels[outputs.long()]
        self.work['context_samples'] += 1

    def prefix(self):
        if self.context is None:
            raise RuntimeError('Set a real sampled context')
        (inputs, outputs, blocks) = self.context
        hidden = self.features[inputs.long()]
        for i in range(2):
            (layer, block) = (self.model.convs[i], blocks[i])
            neighbor = block_mean(self.dgl, block, hidden)
            hidden = layer.lin_l(neighbor) + layer.lin_r(hidden[:block.num_dst_nodes()])
            hidden = self.model.bns[i](hidden).relu()
            hidden = F.dropout(hidden, p=self.model.dropout, training=self.model.training)
        return (block_mean(self.dgl, blocks[2], hidden), hidden[:len(outputs)])
