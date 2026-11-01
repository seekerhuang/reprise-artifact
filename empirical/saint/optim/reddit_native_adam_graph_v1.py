from types import SimpleNamespace
VERSION = 'Reddit-SMA5-native-fusedAdam-graph-original-real-three-APIs-v1'

def frozen():
    from empirical.saint.optim import yelp_native_adam_graph
    return yelp_native_adam_graph

def install(optimizer, *, head_parameters, body_parameters, inactive_parameters):
    (head, body, unused) = map(tuple, (head_parameters, body_parameters, inactive_parameters))
    if len(head) != 2 or len(body) != 24 or len(unused) != 2 or ([len(g['params']) for g in optimizer.param_groups] != [22, 2, 4]) or any((g['betas'] != (0.9, 0.999) or g['eps'] != 1e-06 or g['weight_decay'] != 0 or (g['fused'] is not True) or (g['foreach'] is not False) for g in optimizer.param_groups)):
        raise RuntimeError('Original Reddit Head2/Body24/unused2 native22/2/4 fused Adam/eps6/beta09 required')
    if optimizer.state:
        raise RuntimeError('Capture dispatcher installs only on original empty Adam')
    if {id(p) for p in optimizer.param_groups[2]['params']} != {id(p) for p in head + unused}:
        raise RuntimeError('Original classifier group must contain actual active Head and unused parameters')
    if any((p.grad is not None for p in unused)):
        raise RuntimeError('Original unused parameters require None gradients')
    return frozen().install_native_adam(optimizer, head_parameters=head, body_parameters=body, inactive_parameters=unused)

def adapted_class(original):

    class RedditNativeAdamGraph(original):

        def __init__(self, model, method):
            super().__init__(model, method)
            self._reddit_native_adam = install(model.optimizer, head_parameters=self.head.values(), body_parameters=self.body.values(), inactive_parameters=self.unused.values())

        def close(self):
            try:
                return super().close()
            finally:
                self._reddit_native_adam.close()
    return RedditNativeAdamGraph

def adapter_view(original):
    original_adapt = original.adapt

    def adapt(parent, rule):
        result = original_adapt(parent, rule)
        return adapted_class(result) if rule == 'ordered' else result
    result = SimpleNamespace(**vars(original))
    result.adapt = adapt
    result.specification = original.specification
    return result
