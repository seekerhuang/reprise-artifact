import torch
import torch.nn.functional as F
from main.graph_operator_reuse import ExactPropagation
from main.csr import select_csr_rows

class CommonBNMeanP2:

    def __init__(self, model, optimizer, matrix, features, train_ids, labels):
        assert model.num_layers == 3 and model.batch_norm
        assert not model.linear and (not model.residual) and (not model.drop_input)
        assert len(model.convs) == len(model.bns) == 3
        (self.model, self.optimizer) = (model, optimizer)
        (self.matrix, self.features) = (matrix, features)
        self.transpose = matrix.transpose(0, 1).to_sparse_csr()
        self.train_matrix = select_csr_rows(matrix, train_ids)
        self.train_transpose = self.train_matrix.transpose(0, 1).to_sparse_csr()
        self.first_image = torch.sparse.mm(matrix, features)
        self.labels = labels[train_ids]
        self.updates = self.blocks = 0
        self.work = dict(prefix_forwards=0, prefix_VJPs=0, head_queries=0, head_updates=0, body_updates=0, supervised_presentations=0)

    def prefix(self):
        h = self.features
        for (i, conv) in enumerate(self.model.convs[:2]):
            aggregate = self.first_image if i == 0 else ExactPropagation.apply(h, self.matrix, self.transpose)
            h = F.linear(aggregate, conv.lin.weight, conv.bias)
            h = F.dropout(self.model.bns[i](h).relu(), p=self.model.dropout, training=True)
        return ExactPropagation.apply(h, self.train_matrix, self.train_transpose)

    @torch.no_grad()
    def logits(self):
        self.model.eval()
        h = self.features
        for (i, conv) in enumerate(self.model.convs):
            if i == 0:
                h = F.linear(self.first_image, conv.lin.weight, conv.bias)
            else:
                h = torch.sparse.mm(self.matrix, F.linear(h, conv.lin.weight)) + conv.bias
            if i < 2:
                h = self.model.bns[i](h).relu()
        return h

    def verify(self):
        unused = {id(p) for p in self.model.bns[2].parameters()}
        clocks = {name: int(self.optimizer.state.get(p, {}).get('step', 0)) for (name, p) in self.model.named_parameters()}
        assert len(clocks) == 12
        for (name, p) in self.model.named_parameters():
            assert clocks[name] == (0 if id(p) in unused else self.updates)
            if id(p) in unused:
                assert p not in self.optimizer.state
        assert [int(b.num_batches_tracked) for b in self.model.bns] == [self.blocks, self.blocks, 0]
        return clocks
