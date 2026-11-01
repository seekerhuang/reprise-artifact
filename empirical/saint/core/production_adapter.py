import importlib.util
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
_DECODERS = {}

def _module(dataset):
    if dataset not in _DECODERS:
        name = 'R75_' + dataset + '_production_graph'
        spec = importlib.util.spec_from_file_location(name, HERE / (dataset + '_production_graph_v3.py'))
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _DECODERS[dataset] = module
    return _DECODERS[dataset]

def _clip_without_observer(self, labels, structural):
    import torch
    for label in labels:
        norm = torch.nn.utils.clip_grad_norm_(list(self.groups[label].values()), 5.0)
        self.clip_calls[label] += 1
        if structural and label == 'L':
            self._L_coefficient = torch.clamp(5.0 / (norm.detach() + 1e-06), max=1.0)

def adapter_view(original, dataset):
    if dataset not in ('yelp', 'reddit'):
        raise ValueError('Only the two frozen SAINT architectures')
    causal = original

    def specification(rule):
        return {}

    def adapt(parent, rule):
        if rule not in ('ordered', 'off', 'reversed'):
            raise ValueError('Local requires its original extra pullback, not this radial route')
        if dataset == 'yelp' and rule == 'reversed':
            raise ValueError('No new Yelp reversed arm is defined')
        frozen = causal.adapt(parent, rule)
        if rule == 'off':
            raise ValueError(rule)

        class ProductionRadial(frozen):

            def __init__(self, model, method):
                super().__init__(model, method)
                self._graph_decoder = _module(dataset).GraphDecode(selected=self.S, late_parameters=tuple(self.groups['L'].values()), warmup_iterations=3)

            def _clip(self, labels):
                _clip_without_observer(self, labels, True)

            def _measure_difference_half(self, lambda0, lambda1):
                if dataset != 'reddit':
                    raise RuntimeError('Only the original Reddit half measurement')
                if lambda0 is None or lambda1 is None or lambda0.shape != lambda1.shape:
                    raise RuntimeError('Two actual raw preclip boundary derivatives required')
                self.structural_queries += 2
                self.structural_applications += 1
                return causal.structural.channel_half_difference(causal.geometry, self._z, self._boundary, self._unclamped, lambda0, lambda1)

            def _prepare_pair(self, acquired):
                import torch
                if any((p.grad is None for p in self.groups['L'].values())):
                    raise RuntimeError('Complete actual clipped mean L support required')
                if dataset == 'reddit':
                    if acquired is None or acquired.shape != self.S[0].shape or acquired.dtype != torch.float64:
                        raise RuntimeError('Original FP64 half difference required')
                    decoded = self._graph_decoder.decode_half(acquired, self._L_coefficient)
                else:
                    if len(acquired) != 2:
                        raise RuntimeError('Original two Yelp measurements required')
                    decoded = self._graph_decoder.decode(acquired, self._L_coefficient)
                self._pair = decoded.pair
                self.structural_contexts += 1

            def flush_structural_stats(self):
                pass

            def close(self):
                try:
                    return super().close()
                finally:
                    self._graph_decoder.close()
        return ProductionRadial
    view = SimpleNamespace(**vars(original))
    view.adapt = adapt
    view.specification = specification
    return view
