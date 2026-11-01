class Training:

    def __init__(self, model, optimizer):
        if model.num_layers != 2 or model.linear or model.residual:
            raise ValueError('This first adapter covers GAS L2/no-linear/no-residual')
        (self.model, self.optimizer) = (model, optimizer)
        self.readout = model.convs[-1]
        if self.readout.normalize or self.readout.lin.bias is not None:
            raise ValueError('Require original pre-normalized GCN and separate output bias')
        named = dict(model.named_parameters())
        head_ids = {id(p) for p in self.readout.parameters()}
        unused_ids = {id(p) for bn in (model.bns[1:] if model.batch_norm else model.bns) for p in bn.parameters()}
        self.head = {k: p for (k, p) in named.items() if id(p) in head_ids}
        self.unused = {k: p for (k, p) in named.items() if id(p) in unused_ids}
        self.body = {k: p for (k, p) in named.items() if id(p) not in head_ids | unused_ids}
        if len(self.head) != 2 or len(self.body) != (4 if model.batch_norm else 2):
            raise ValueError('Unexpected active prefix/readout schema')
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in named.values()}:
            raise ValueError('Original optimizer coverage differs')
        self.blocks = self.visits = self.optimizer_calls = 0
        self.labels = self.cache_peak_payload_bytes = 0
        self.cache_alive = False

    def tail(self, query):
        import torch.nn.functional as F
        return F.linear(query, self.readout.lin.weight, self.readout.bias)

    def _step(self):
        self.optimizer.step()
        self.optimizer_calls += 1
        self.optimizer.zero_grad(set_to_none=True)
