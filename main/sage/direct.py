import torch.nn.functional as F


def ordinary_mean(block, features):
    import dgl.function as fn
    with block.local_scope():
        block.srcdata['h'] = features
        if 'edge_weights' in block.edata:
            block.edata['ordinary_w'] = block.edata['edge_weights'].to(features.dtype)
            message = fn.u_mul_e('h', 'ordinary_w', 'm')
        else:
            message = fn.copy_u('h', 'm')
        block.update_all(message, fn.mean('m', 'mean'))
        return block.dstdata['mean']


class Direct:
    def __init__(self, model, optimizer, dgl, graph, features, labels, *, fanouts):
        self.model, self.optimizer = model, optimizer
        self.graph, self.features, self.all_labels = graph, features, labels
        self.sampler = dgl.dataloading.LaborSampler(list(fanouts), importance_sampling=0,
                                                  layer_dependency=False, batch_dependency=1)
        self.actual_updates = 0
        self.loss_weights = None
        self.context = None

    def set_context(self, roots):
        inputs, outputs, blocks = self.sampler.sample_blocks(self.graph, roots)
        self.context = inputs, outputs, blocks
        self.labels = self.all_labels[outputs.long()]

    def forward(self):
        inputs, _, blocks = self.context
        hidden = self.features[inputs.long()]
        for index, (layer, block) in enumerate(zip(self.model.convs, blocks)):
            neighbors = ordinary_mean(block, hidden)
            hidden = layer.lin_l(neighbors) + layer.lin_r(hidden[:block.num_dst_nodes()])
            if index < 2:
                hidden = self.model.bns[index](hidden).relu()
                hidden = F.dropout(hidden, p=self.model.dropout, training=self.model.training)
        return hidden

    def update_block(self, queries=1, scheduler=None):
        if queries != 1:
            raise ValueError('Direct training requires one fresh context per update')
        self.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.forward(), self.labels, weight=self.loss_weights)
        loss.backward()
        self.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        self.actual_updates += 1
        return [loss.detach()]

