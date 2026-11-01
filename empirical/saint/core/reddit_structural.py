from empirical.saint.core import geometry

def selected_names():
    return {'S': ('conv_layers.3.params.2', 'conv_layers.3.params.1')}

def channel_half_difference(geometry, z, boundary, unclamped, lambda0, lambda1):
    import torch
    half = lambda1.detach().to(dtype=torch.float64, copy=True)
    half.sub_(lambda0.detach()).mul_(0.5)
    return geometry.channel_measurement(z, boundary, unclamped, half)

def adapt(parent):

    class StructuralTraining(parent):

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
                prefix = self._structured_prefix(nodes, adj)
                leaf = prefix.detach().requires_grad_(True)
                labels = m.label_full[nodes] if m.sigmoid_loss else m.label_full_cat[nodes]
                cotangent = None
                difference_half = None
                for query in range(2):
                    leaf.grad = None
                    loss = m._loss(self._tail(leaf, adj), labels, weights)
                    loss.backward()
                    losses.append(loss.detach())
                    if cotangent is None:
                        cotangent = leaf.grad.detach().clone()
                    else:
                        difference_half = self._measure_difference_half(cotangent, leaf.grad.detach())
                        cotangent.add_(leaf.grad)
                    self._clip(self.head_labels)
                    if query == 0:
                        self._step()
                leaf.grad = None
                cotangent.mul_(0.5)
                prefix.backward(cotangent)
                self._clip(self.body_labels)
                gradient = {k: p.grad for (k, p) in self.body.items()}
                self._prepare_pair(difference_half)
                self._install_pair(0)
                self._step()
                for (k, p) in self.body.items():
                    p.grad = gradient[k]
                self._install_pair(1)
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
                self._release_structural()
                self.randomness.end()
                self.active_cache = False

        def __init__(self, model, method):
            super().__init__(model, method)
            if model.order_layer != [1, 0, 1, 0] or model.aggregators[3].order != 0 or model.sigmoid_loss:
                raise ValueError('Only the locked Reddit order1-0-1-0 softmax backbone')
            layer = model.aggregators[3]
            if layer.bias != 'norm' or layer.f_lin[0].weight.shape[0] != 128 or list(model.named_buffers()):
                raise ValueError('Require last128 feature-axis affine norm without running buffers')
            names = dict(model.named_parameters())
            self.S = (layer.scale[0], layer.offset[0])
            if any((names[n] is not p for (n, p) in zip(selected_names()['S'], self.S))):
                raise RuntimeError('Actual last scale/offset identity changed')
            self.structural_contexts = self.structural_queries = self.structural_installs = 0
            self.structural_applications = 0
            self.half_difference_peak_FP64_bytes = 0
            self.structural_sum = self.structural_max = self.structural_min_tau = None
            self.structural_recorded = 0
            self.last_structural_record = None
            self._release_structural()

        def _release_structural(self):
            self._z = self._boundary = self._unclamped = self._pair = None

        def _structured_prefix(self, nodes, adj):
            if self._z is not None:
                raise RuntimeError('Prior structural context retained')
            (_, hidden) = self.model.conv_layers((adj, self.model.feat_full[nodes]))
            (z, unclamped) = geometry.normalize_capture(hidden)
            boundary = self.model.classifier.f_dropout(z)
            self._z = z.detach()
            self._unclamped = unclamped
            self._boundary = boundary.detach()
            return boundary

        def _install_pair(self, position):
            if self._pair is None or position not in (0, 1):
                raise RuntimeError('Ordered pair missing')
            for (p, g) in zip(self.S, self._pair[position]):
                p.grad = g
            self.structural_installs += 1
    return StructuralTraining
