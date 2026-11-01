import torch.nn.functional as F
HEAD_NAMES = ('convs.2.lin_l.weight', 'convs.2.lin_l.bias', 'convs.2.lin_r.weight')
import torch
from main.graph_operator_reuse import ExactPropagation
from main.csr import select_csr_rows

class FrozenMeanP2:

    def __init__(self, model, optimizer, matrix, features, training_ids, labels):
        if len(model.convs) != 3 or len(model.bns) != 2:
            raise ValueError('This adapter implements the source three-layer SAGE/BN')
        (self.model, self.optimizer) = (model, optimizer)
        (self.matrix, self.features) = (matrix, features)
        self.training_ids = training_ids
        self.labels = labels[training_ids]
        self.transpose = matrix.transpose(0, 1).to_sparse_csr()
        self.label_matrix = select_csr_rows(matrix, training_ids)
        self.label_transpose = self.label_matrix.transpose(0, 1).to_sparse_csr()
        self.first_image = torch.sparse.mm(matrix, features)
        names = dict(model.named_parameters())
        assert len(names) == 13 and all((name in names for name in HEAD_NAMES))
        self.actual_updates = 0
        self.work = dict(prefix_forwards=0, prefix_VJPs=0, head_queries=0, head_updates=0, body_updates=0, supervised_presentations=0, boundary_payload_peak_bytes=None)

    def prefix(self):
        x = self.features
        for (i, layer) in enumerate(self.model.convs[:-1]):
            neighbor = self.first_image if i == 0 else ExactPropagation.apply(x, self.matrix, self.transpose)
            x = layer.lin_l(neighbor) + layer.lin_r(x)
            x = self.model.bns[i](x).relu()
            x = F.dropout(x, p=self.model.dropout, training=self.model.training)
        return (ExactPropagation.apply(x, self.label_matrix, self.label_transpose), x[self.training_ids])

    def head(self, ports):
        layer = self.model.convs[-1]
        return layer.lin_l(ports[0]) + layer.lin_r(ports[1])

    def verify_clocks(self):
        clocks = {name: int(self.optimizer.state[p]['step']) for (name, p) in self.model.named_parameters()}
        if len(clocks) != 13 or set(clocks.values()) != {self.actual_updates}:
            raise AssertionError(clocks)
        expected = self.work['prefix_forwards']
        assert [int(bn.num_batches_tracked) for bn in self.model.bns] == [expected] * 2
        return clocks
