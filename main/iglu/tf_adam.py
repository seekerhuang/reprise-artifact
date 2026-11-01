import torch

class TF1Adam(torch.optim.Optimizer):

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(parameters, dict(lr=lr, betas=betas, eps=eps))
        for group in self.param_groups:
            reference = group['params'][0]
            assert reference.dtype == torch.float32
            group['tf_beta1_power'] = reference.new_tensor(betas[0])
            group['tf_beta2_power'] = reference.new_tensor(betas[1])
            group['tf_step'] = 0
        self._set_hyper_tensors()

    def _set_hyper_tensors(self):
        self._hyper_tensors = []
        for group in self.param_groups:
            reference = group['params'][0]
            assert all((p.device == reference.device and p.dtype == reference.dtype for p in group['params']))
            for name in ('tf_beta1_power', 'tf_beta2_power'):
                group[name] = group[name].to(reference.device, dtype=reference.dtype)
            (b1, b2) = [reference.new_tensor(value) for value in group['betas']]
            self._hyper_tensors.append(dict(lr_python=group['lr'], lr=reference.new_tensor(group['lr']), b1=b1, b2=b2, one_minus_b1=1 - b1, one_minus_b2=1 - b2))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_hyper_tensors()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if any((group['lr'] != values['lr_python'] for (group, values) in zip(self.param_groups, self._hyper_tensors))):
            self._set_hyper_tensors()
        for (group, hyper) in zip(self.param_groups, self._hyper_tensors):
            rate = hyper['lr'] * torch.sqrt(1 - group['tf_beta2_power']) / (1 - group['tf_beta1_power'])
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError('This original target has dense parameter gradients.')
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
                state['step'] += 1
                (m, v) = (state['exp_avg'], state['exp_avg_sq'])
                m.add_((gradient - m) * hyper['one_minus_b1'])
                v.add_((gradient.square() - v) * hyper['one_minus_b2'])
                parameter.sub_(rate * m / (v.sqrt() + group['eps']))
            group['tf_step'] += 1
            group['tf_beta1_power'].mul_(hyper['b1'])
            group['tf_beta2_power'].mul_(hyper['b2'])
        return loss
