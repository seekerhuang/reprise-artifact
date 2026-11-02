def adapt(parent):

    class ThreeAPI(parent):

        def __init__(self, model, method):
            if method != 'lateA':
                raise ValueError('Only late-affine M221 is supported')
            super().__init__(model, method)
            if {id(p) for p in self.body.values()} & {id(p) for p in self.head.values()}:
                raise RuntimeError('Disjoint prefix/head supports required')

        def block(self, nodes, adj, weights):
            if self.closed:
                raise RuntimeError('Training engine already closed')
            m = self.model
            m.train()
            m.optimizer.zero_grad(set_to_none=True)
            self.randomness.begin()
            self.active_cache = True
            losses = []
            try:
                prefix = self._prefix(nodes, adj)
                leaf = prefix.detach().requires_grad_(True)
                labels = m.label_full[nodes] if m.sigmoid_loss else m.label_full_cat[nodes]
                cotangent = None
                for query in range(2):
                    leaf.grad = None
                    loss = m._loss(self._tail(leaf, adj), labels, weights)
                    loss.backward()
                    losses.append(loss.detach())
                    if cotangent is None:
                        cotangent = leaf.grad.detach().clone()
                    else:
                        cotangent.add_(leaf.grad)
                    self._clip(self.head_labels)
                    if query == 0:
                        self._step()
                leaf.grad = None
                cotangent.mul_(0.5)
                prefix.backward(cotangent)
                self._clip(self.body_labels)
                gradient = {k: p.grad for (k, p) in self.body.items()}
                self._step()
                for (k, p) in self.body.items():
                    p.grad = gradient[k]
                self._step()
                self.prefix_forwards += 1
                self.body_vjps += 1
                self.local_forwards += 1
                self.local_vjps += 1
                self.head_queries += 2
                del gradient, cotangent, leaf, prefix
                self.blocks += 1
                self.visits += 2
                self.loss_node_presentations += 2 * len(nodes)
                self.prefix_node_presentations += len(nodes)
                return losses
            finally:
                self.randomness.end()
                self.active_cache = False

        def work(self):
            return dict(super().work(), native_Adam_call_grouping='head1; head2+body1; body2', core_query_and_per_parameter_update_sequence_unchanged=True)
    return ThreeAPI
