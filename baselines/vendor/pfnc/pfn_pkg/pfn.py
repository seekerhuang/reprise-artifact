import torch

def get_attr_recursively(obj, attr_name):
    name_split = attr_name.split('.')[1:]
    for name in name_split:
        obj = getattr(obj, name)
    return obj

class PFN(object):

    def __init__(self, num_nodes, num_layers, queue_size=8, queue_device=torch.device('cuda'), grad_update_again=False, add_padding_queues=False, add_param_value=False, use_padding_grad=False, use_grad_distances_check=False, point_wise_distance_check=False, distance_func=None, check_grad_distances_lambda=0.1, padding_beta=0.9, grad_inflation=False, grad_add=False, grad_normlized_add=False, grad_normlized_add_alpha=0.9, not_use_nid_access=False, tmp_scale_up=1.0, **kwargs):
        super().__init__()
        self.hds = []
        self.meta = []
        self.queue_size = queue_size
        self.queue_device = queue_device
        self.growing_current_queue_pos = -1
        self.current_queue_pos = -1
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.add_padding_queues = add_padding_queues
        self.add_param_value = add_param_value
        self.grad_update_again = grad_update_again
        assert not grad_update_again if add_padding_queues else True, 'grad_update_again should be False when add_padding_queues is True. because the padding update should use non stall grad.'
        self.directly_use_padding_grad = use_padding_grad
        assert add_padding_queues if use_padding_grad else True, 'directly_use_padding_grad need to have mantained padding queues.'
        self.use_grad_distances_check = use_grad_distances_check
        self.point_wise_distance_check = point_wise_distance_check
        if distance_func is not None:
            self.distance_func = distance_func
        elif use_grad_distances_check:
            assert ValueError('distance_func should be provided when use_grad_distances_check is True.')
        self.padding_beta = padding_beta
        self.grad_normlized_add_alpha = grad_normlized_add_alpha
        self.check_grad_distances_lambda = check_grad_distances_lambda
        self.grad_inflation = grad_inflation
        self.grad_add = grad_add
        self.grad_normlized_add = grad_normlized_add
        self.not_use_nid_access = not_use_nid_access
        self.tmp_scale_up = tmp_scale_up
        assert grad_inflation or grad_add or grad_normlized_add, 'At least choose one of the grad calculation options.'

    def generate_meta(self, layer_list, layer_name_list, layer_num_list, cfg=None):
        assert len(layer_list) == len(layer_name_list), 'layer_list and layer_name_list should have the same length.'
        assert len(layer_list) == len(layer_num_list), 'layer_list and layer_num_list should have the same length.'
        assert all((0 <= layer_num < self.num_layers for layer_num in layer_num_list)), 'layer_num_list should be in [0, num_layers).'
        self.layer_name_list = layer_name_list
        self.layer_num_list = layer_num_list
        if len(self.meta) > 0:
            raise ValueError('Meta info already exists.')
        for layer in layer_list:
            self.meta.append(self.get_history_meta_one_layer(layer, cfg))
        self.queue_pos_for_nodes = {layer_num_list[i]: {para_meta[0]: torch.zeros(self.num_nodes, dtype=torch.int32, device=self.queue_device) - 1 for para_meta in layer_meta} for (i, layer_meta) in enumerate(self.meta)}
        print(f'[INFO] {[{para_meta[0]: 1 for para_meta in layer_meta} for layer_meta in self.meta]}')

    def init_all_history_queues(self):
        assert len(self.meta) > 0, 'Meta info not generated.'
        for (i, layer_meta) in enumerate(self.meta):
            for parameter_meta in layer_meta:
                queue_name = self.layer_name_list[i] + parameter_meta[0].replace('.', '_')
                setattr(self, queue_name + '_grad', torch.zeros((self.queue_size, *parameter_meta[1]), device=self.queue_device))
                if self.add_padding_queues:
                    setattr(self, queue_name + '_grad_padding', torch.zeros(parameter_meta[1], device=self.queue_device))
                if self.add_param_value:
                    setattr(self, queue_name + '_value', torch.zeros((self.queue_size, *parameter_meta[1]), device=self.queue_device))

    @staticmethod
    def get_history_meta_one_layer(layer, cfg=None):
        one_by_one = False
        if one_by_one:
            if layer.__class__ is SAGEConv:
                meta = [('.lin_l.weight', layer.lin_l.weight.size(), get_attr_recursively(layer, '.lin_l.weight'))]
                if hasattr(layer.lin_l, 'bias') and layer.lin_l.bias is not None:
                    meta.append(('.lin_l.bias', layer.lin_l.bias.size(), get_attr_recursively(layer, '.lin_l.bias')))
                if hasattr(layer, 'lin') and layer.lin is not None:
                    meta.append(('.lin.weight', layer.lin.weight.size(), get_attr_recursively(layer, '.lin.weight')))
                    meta.append(('.lin.bias', layer.lin.bias.size(), get_attr_recursively(layer, '.lin.bias')))
                if hasattr(layer, 'lin_r') and layer.lin_r is not None:
                    meta.append(('.lin_r.weight', layer.lin_r.weight.size(), get_attr_recursively(layer, '.lin_r.weight')))
            elif layer.__class__ is GCNConv:
                meta = [('.lin.weight', layer.lin.weight.size(), get_attr_recursively(layer, '.lin.weight'))]
                if hasattr(layer, 'bias') and layer.bias is not None:
                    meta.append(('.bias', layer.bias.size(), get_attr_recursively(layer, '.bias')))
        else:
            meta = []
            for (name, param) in layer.named_parameters():
                meta.append(('.' + name, param.size(), param))
        return meta

    def detached_grad_correction(self, n_id_list):
        self.growing_current_queue_pos = self.growing_current_queue_pos + 1
        self.current_queue_pos = self.growing_current_queue_pos % self.queue_size

        def conv_hook_generator(layer_name, layer_num, param_name, param_obj, grad_update_again=False, grad_add=False):

            def backward_hook(grad):
                new_grad = self.calculate_weighted_sum_of_grad_hist_part(layer_name, layer_num, param_name, n_id_list[layer_num], grad)
                if not grad_update_again:
                    self.update_queues(grad, layer_name, layer_num, param_name, param_obj, n_id_list[layer_num])
                else:
                    self.update_queues(new_grad, layer_name, layer_num, param_name, param_obj, n_id_list[layer_num])
                return new_grad
            return backward_hook
        for (i, layer_meta) in enumerate(self.meta):
            for parameter_meta in layer_meta:
                parameter_meta[2].grad.data = conv_hook_generator(self.layer_name_list[i], self.layer_num_list[i], parameter_meta[0], parameter_meta[2], self.grad_update_again, self.grad_add)(parameter_meta[2].grad.data)

    def update_queues(self, grad_to_save, layer_name, layer_num, param_name, param_obj, nid):
        queue_name = layer_name + param_name.replace('.', '_')
        queue_name_grad = queue_name + '_grad'
        queue_name_grad_padding = queue_name + '_grad_padding'
        queue_name_value = queue_name + '_value'
        eval('self.' + queue_name_grad)[self.current_queue_pos] = grad_to_save
        if self.add_padding_queues:
            setattr(self, queue_name_grad_padding, getattr(self, queue_name_grad_padding) * (1 - self.padding_beta) + grad_to_save * self.padding_beta)
        if self.add_param_value:
            eval('self.' + queue_name_value)[self.current_queue_pos] = param_obj.detach()
        self.queue_pos_for_nodes[layer_num][param_name][nid] = self.growing_current_queue_pos

    def calculate_weighted_sum_of_grad_hist_part(self, layer_name, layer_num, param_name, nid=None, grad=None):
        queue_name = layer_name + param_name.replace('.', '_')
        queue_name_grad = queue_name + '_grad'
        queue_name_grad_padding = queue_name + '_grad_padding'
        queue_name_value = queue_name + '_value'
        if self.directly_use_padding_grad:
            return getattr(self, queue_name_grad_padding)
        if self.not_use_nid_access:
            nid = None
        if nid is None:
            effective_queue_pos = self.queue_pos_for_nodes[layer_num][param_name][self.queue_pos_for_nodes[layer_num][param_name] > max(self.growing_current_queue_pos - 1 - self.queue_size, -1)] % self.queue_size
            effective_queue_pos = effective_queue_pos.unique(return_counts=True)
        else:
            within_queue_mask = self.queue_pos_for_nodes[layer_num][param_name] > max(self.growing_current_queue_pos - 1 - self.queue_size, -1)
            effective_queue_pos = self.queue_pos_for_nodes[layer_num][param_name][nid][within_queue_mask[nid]] % self.queue_size
            effective_queue_pos = effective_queue_pos.unique(return_counts=True)
        effective_queue_pos = list(effective_queue_pos)
        queue_idx = effective_queue_pos[0].long()
        if not self.point_wise_distance_check:
            if self.use_grad_distances_check:
                value_queue = getattr(self, queue_name_value)
                distance_queue = self.distance_func(value_queue[self.current_queue_pos], value_queue)
                grad_distances_check_mask = distance_queue < self.check_grad_distances_lambda
                queue_idx = queue_idx[grad_distances_check_mask]
                effective_queue_pos[1] = effective_queue_pos[1][grad_distances_check_mask]
            if self.grad_inflation:
                param_num_dims = len(getattr(self, queue_name_grad)[queue_idx].shape)
                old_grad = (getattr(self, queue_name_grad)[queue_idx] * (effective_queue_pos[1] / effective_queue_pos[1][queue_idx == self.current_queue_pos]).reshape((-1,) + (1,) * (param_num_dims - 1))).sum(dim=0)
                return old_grad + grad
            elif self.grad_add:
                param_num_dims = len(getattr(self, queue_name_grad)[queue_idx].shape)
                old_grad = (getattr(self, queue_name_grad)[queue_idx] * (effective_queue_pos[1] / effective_queue_pos[1].sum()).reshape((-1,) + (1,) * (param_num_dims - 1))).sum(dim=0)
                return old_grad * self.tmp_scale_up + grad
            elif self.grad_normlized_add:
                param_num_dims = len(getattr(self, queue_name_grad)[queue_idx].shape)
                old_grad = (getattr(self, queue_name_grad)[queue_idx] * (effective_queue_pos[1] / effective_queue_pos[1].sum()).reshape((-1,) + (1,) * (param_num_dims - 1))).sum(dim=0)
                return old_grad + self.grad_normlized_add_alpha * (grad - old_grad)
        else:
            raise NotImplementedError('seem redundant, not implemented yet.')
            assert self.point_wise_distance_check, 'point_wise_distance_check should be True.'
            value_queue = getattr(self, queue_name_value)
            distance_queue = self.distance_func(value_queue[self.current_queue_pos], value_queue)
            grad_distances_check_mask = distance_queue < self.check_grad_distances_lambda
            if self.grad_inflation:
                raise ValueError('grad_inflation is not supported in point_wise_distance_check.')
            elif self.grad_add:
                return (getattr(self, queue_name_grad) * grad_distances_check_mask).sum(dim=0)
            else:
                raise NotImplementedError('Complexity is high, not implemented yet, need calculate number of nodes for each point differently.')
