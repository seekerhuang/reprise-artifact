from pathlib import Path
import inspect
import sys
import textwrap
HERE = Path(__file__).resolve().parent
from empirical.saint.core import geometry
STAT_NAMES = ('tau', 'tau_below_one', 'raw_r_norm', 'applied_r_norm', 'native_L_clip_coefficient', 'native_mean_L_norm', 'postcast_pair_max_L_norm', 'postcast_selected_mean_error_norm', 'native_mean_squared_cap_excess', 'native_mean_outside_cap', 'clamped_normalization_rows', 'raw_measurement_difference_norm', 'scaled_mean_constraint_residual_norm', 'zero_direction_channels')
BLOCK_CHANGES = (('prefix=self._prefix(nodes,adj)', 'prefix=self._structured_prefix(nodes,adj)'), ('cotangent=None', 'cotangent=None\n                measurements=[]'), ('losses.append(loss.detach())', 'losses.append(loss.detach())\n                    measurements.append(self._measure(leaf.grad.detach()))'), ('gradient={k:p.grad for k,p in self.body.items()}', 'gradient={k:p.grad for k,p in self.body.items()}\n                self._prepare_pair(measurements)'), ('self._step()  # head2 + body1, native Adam per-parameter clocks', 'self._install_pair(0)\n                self._step()  # head2 + body1, native Adam per-parameter clocks'), ('self._step()  # body2 only', 'self._install_pair(1)\n                self._step()  # body2 only'), ('            finally:\n                self.randomness.end()', '            finally:\n                self._release_structural()\n                self.randomness.end()'))

def adapt(parent):

    class StructuralTraining(parent):
        block = parent.block

        def __init__(self, model, method):
            super().__init__(model, method)
            if model.order_layer != [1, 1, 0] or model.aggregators[2].order != 0 or (not model.sigmoid_loss):
                raise ValueError('Only the locked Yelp order1-1-0 sigmoid backbone')
            layer = model.aggregators[2]
            if layer.bias != 'norm' or layer.f_lin[0].weight.shape[0] != 512 or list(model.named_buffers()):
                raise ValueError('Require last512 feature-axis affine norm without running buffers')
            names = dict(model.named_parameters())
            self.S = (layer.scale[0], layer.offset[0])
            if any((names[n] is not p for (n, p) in zip(geometry.specification()['S'], self.S))):
                raise RuntimeError('Actual last scale/offset identity changed')
            self.structural_contexts = self.structural_queries = self.structural_installs = 0
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

        def _measure(self, raw_lambda):
            if raw_lambda is None:
                raise RuntimeError('Current raw boundary derivative is missing')
            self.structural_queries += 1
            return geometry.channel_measurement(self._z, self._boundary, self._unclamped, raw_lambda)

        def _clip(self, labels):
            import torch
            for label in labels:
                norm = torch.nn.utils.clip_grad_norm_(list(self.groups[label].values()), 5.0)
                self.clip_calls[label] += 1
                self.clip_active[('P', 'L', 'C').index(label)].add_(norm.detach().gt(5.0))
                if label == 'L':
                    self._L_coefficient = torch.clamp(5.0 / (norm.detach() + 1e-06), max=1.0)

        def _prepare_pair(self, measurements):
            import torch
            if len(measurements) != 2 or any((p.grad is None for p in self.groups['L'].values())):
                raise RuntimeError('Two current feedbacks and complete mean L support required')
            n = torch.stack([p.detach().double() for p in self.S])
            q = torch.stack([p.grad.detach().double() for p in self.S])
            (raw_r, zero) = geometry.euclidean(n, (measurements[1] - measurements[0]) * 0.5)
            r = raw_r * self._L_coefficient.double()
            full_q_square = torch.stack([p.grad.detach().double().square().sum() for p in self.groups['L'].values()]).sum()
            (tau, excess, outside) = geometry.maximal_tau(full_q_square, q, r)
            self._pair = tuple((tuple(((q[j] + sign * tau * r[j]).to(p.dtype) for (j, p) in enumerate(self.S))) for sign in (-1.0, 1.0)))
            p0 = torch.stack([g.double() for g in self._pair[0]])
            p1 = torch.stack([g.double() for g in self._pair[1]])
            remainder = (full_q_square - q.square().sum()).clamp_min(0.0)
            pair_norm = torch.maximum((remainder + p0.square().sum()).sqrt(), (remainder + p1.square().sum()).sqrt())
            mean_error = ((p0 + p1) * 0.5 - q).norm()
            constraint = (n * q).sum(0) - self._L_coefficient.double() * (measurements[0] + measurements[1]) * 0.5
            values = torch.stack((tau, (tau < 1).double(), raw_r.norm(), (tau * r).norm(), self._L_coefficient.double(), full_q_square.sqrt(), pair_norm, mean_error, excess, outside, (~self._unclamped).sum().double(), (measurements[1] - measurements[0]).norm(), constraint.norm(), zero))
            self.structural_sum = values.clone() if self.structural_sum is None else self.structural_sum + values
            self.structural_max = values.clone() if self.structural_max is None else torch.maximum(self.structural_max, values)
            self.structural_min_tau = tau.clone() if self.structural_min_tau is None else torch.minimum(self.structural_min_tau, tau)
            self.structural_contexts += 1
            self.cache_peak_payload_bytes = max(self.cache_peak_payload_bytes, sum((x.numel() * x.element_size() for x in (self._z, self._boundary, self._unclamped, n, q, raw_r, r, *measurements, *self._pair[0], *self._pair[1]))))

        def _install_pair(self, position):
            if self._pair is None or position not in (0, 1):
                raise RuntimeError('Ordered pair missing')
            for (p, g) in zip(self.S, self._pair[position]):
                p.grad = g
            self.structural_installs += 1

        def flush_structural_stats(self):
            import torch
            import math
            if self.structural_recorded == self.structural_contexts:
                return
            count = self.structural_contexts - self.structural_recorded
            values = torch.cat((self.structural_sum / count, self.structural_max, self.structural_min_tau.reshape(1))).detach().cpu().tolist()
            if not all((math.isfinite(x) for x in values)):
                raise ArithmeticError('Nonfinite structural geometry statistics')
            n = len(STAT_NAMES)
            self.last_structural_record = dict(contexts=count, mean=dict(zip(STAT_NAMES, values[:n])), max=dict(zip(STAT_NAMES, values[n:2 * n])), min_tau=values[-1], scope='Per-context device statistics; epoch-boundary copy paid inside training. Native mean outside cap is an explicitly recorded rounding domain, not a strict-cap success. FP32 pair mean/norm errors are recorded.')
            self.structural_recorded = self.structural_contexts
            self.structural_sum = self.structural_max = self.structural_min_tau = None
    return StructuralTraining
