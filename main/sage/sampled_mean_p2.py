import torch
import torch.nn.functional as F
from main.sage.sage_boundary import FrozenMeanP2

def block_mean(dgl, block, features):
    import dgl.function as fn
    with block.local_scope():
        block.srcdata['m221_h'] = features
        if 'edge_weights' in block.edata:
            weights = block.edata['edge_weights']
            block.edata['m221_w'] = weights
            message = fn.u_mul_e('m221_h', 'm221_w', 'm221_m')
        else:
            message = fn.copy_u('m221_h', 'm221_m')
        block.update_all(message, fn.mean('m221_m', 'm221_mean'))
        return block.dstdata['m221_mean']

class SampledMeanP2(FrozenMeanP2):

    def __init__(self, model, optimizer, dgl, graph, features, first_image, labels, *, sampler='neighbor', fanouts=(10, 10)):
        if len(model.convs) != 3 or len(model.bns) != 2:
            raise ValueError('Require the unchanged original Fresh SAGE3/BN model')
        (self.model, self.optimizer, self.dgl) = (model, optimizer, dgl)
        (self.graph, self.features) = (graph, features)
        (self.first_image, self.all_labels) = (first_image, labels)
        if sampler == 'neighbor':
            self.sampler = dgl.dataloading.NeighborSampler(list(fanouts), fused=False)
        elif sampler == 'labor':
            self.sampler = dgl.dataloading.LaborSampler(list(fanouts), importance_sampling=0, layer_dependency=False, batch_dependency=1)
        else:
            raise ValueError(sampler)
        self.actual_updates = 0
        self.work = dict(prefix_forwards=0, prefix_VJPs=0, head_queries=0, head_updates=0, body_updates=0, supervised_presentations=0, boundary_payload_peak_bytes=None, input_root_presentations=None, sampled_first_hidden_nodes=None, sampled_second_hidden_nodes=None, sampled_middle_edges=None, sampled_last_edges=None, context_samples=0)
        self.context = None

    def set_context(self, roots):
        (inputs, outputs, blocks) = self.sampler.sample_blocks(self.graph, roots)
        self.context = (inputs, outputs, blocks)
        self.training_ids = outputs
        self.labels = self.all_labels[outputs.long()]
        self.work['context_samples'] += 1

    def prefix(self):
        if self.context is None:
            raise RuntimeError('Set an actual sampled context before each M221 block')
        (inputs, outputs, blocks) = self.context
        (first, second) = self.model.convs[:2]
        ids = inputs.long()
        hidden = first.lin_l(self.first_image[ids]) + first.lin_r(self.features[ids])
        hidden = self.model.bns[0](hidden).relu()
        hidden = F.dropout(hidden, p=self.model.dropout, training=self.model.training)
        mean = block_mean(self.dgl, blocks[0], hidden)
        hidden = second.lin_l(mean) + second.lin_r(hidden[:blocks[0].num_dst_nodes()])
        hidden = self.model.bns[1](hidden).relu()
        hidden = F.dropout(hidden, p=self.model.dropout, training=self.model.training)
        return (block_mean(self.dgl, blocks[1], hidden), hidden[:len(outputs)])
