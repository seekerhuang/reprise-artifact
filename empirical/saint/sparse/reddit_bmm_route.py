from contextlib import contextmanager
from types import MethodType
from empirical.saint.sparse import yelp_sparse_bmm_route_v1 as PUBLIC
strict_determinism = PUBLIC.strict_determinism

@contextmanager
def install(model):
    if model.order_layer != [1, 0, 1, 0] or model.sigmoid_loss or len(model.aggregators) != 4:
        raise ValueError('Only locked Reddit order1-0-1-0 softmax model')
    saved = []
    counts = {str(index): 0 for index in range(4)}
    try:
        for (index, layer) in enumerate(model.aggregators):
            saved.append((layer, '_spmm' in layer.__dict__, layer.__dict__.get('_spmm')))

            def replacement(self, adj, dense, _index=index):
                result = PUBLIC.public_singleton_mm(adj, dense)
                counts[str(_index)] += 1
                return result
            layer._spmm = MethodType(replacement, layer)
        yield counts
    finally:
        for (layer, present, value) in reversed(saved):
            if present:
                layer._spmm = value
            else:
                delattr(layer, '_spmm')


def adapt(Base):
    class TrainBackend(Base):
        def close(self):
            import torch
            device = next(self.model.parameters()).device
            before = (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled())
            torch.cuda.synchronize(device)
            try:
                with strict_determinism(torch), install(self.model) as counts:
                    result = super().close()
            finally:
                torch.cuda.synchronize(device)
            after = (torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled())
            if after != before or any(counts.values()):
                raise RuntimeError('Close changed flags or executed unexpected SpMM')
            return result
    return TrainBackend
