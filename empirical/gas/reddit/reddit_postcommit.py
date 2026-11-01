from empirical.gas.gas_boundary import Training as PreviousTraining
import time
import torch

class Training(PreviousTraining):
    post_commit_enabled = True

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        if model.batch_norm or model.drop_input or model.linear or model.residual or (len(model.histories) != 1) or model.convs[0].normalize or (model.convs[0].lin.bias is not None) or (model.convs[0].bias is None):
            raise ValueError('Fixed-feature two-layer, no input dropout/BN/skip GCN only')
        self.static_input = None
        self.corrected_blocks = self.corrected_rows = 0
        self.cache_initialization_seconds = self.post_commit_wall_seconds = 0.0
        self.static_record = None

    def install_static_input(self, features, matrix):
        import torch
        from torch_sparse import matmul
        if self.static_input is not None or tuple(matrix.sizes()) != (self.model.num_nodes, self.model.num_nodes):
            raise RuntimeError('One complete fixed input propagation only')
        if features.ndim != 2 or features.shape[0] != self.model.num_nodes or features.requires_grad:
            raise ValueError('Unmodified fixed full features required')
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize(self.model.device)
        stamp = time.perf_counter()
        with torch.no_grad():
            self.static_input = matmul(matrix.to(self.model.device), features.to(self.model.device), reduce='sum').detach()
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize(self.model.device)
        self.cache_initialization_seconds = time.perf_counter() - stamp
        self.static_record = dict(nodes=int(self.static_input.shape[0]), features=int(self.static_input.shape[1]), dtype=str(self.static_input.dtype), device=str(self.static_input.device), bytes=self.static_input.numel() * self.static_input.element_size(), full_input_SpMM_calls=1, original_features_version=features._version, original_features_ptr=features.data_ptr())
        self.original_features = features

    def correct_core(self, core, args):
        import torch
        import torch.nn.functional as F
        if self.static_input is None or len(args) != 3:
            raise RuntimeError('Static input and original n_id/offset/count are required')
        (n_id, offset, count) = args
        if n_id.device.type != 'cpu' or offset.device.type != 'cpu' or count.device.type != 'cpu':
            raise RuntimeError('Original CPU partition coordinates required')
        stamp = time.perf_counter()
        with torch.no_grad():
            core_input = self.static_input.index_select(0, n_id[:core].to(self.model.device))
            conv = self.model.convs[0]
            corrected = F.linear(core_input, conv.lin.weight, conv.bias).relu_()
            (pool, history) = (self.model.pool, self.model.histories[0])
            if pool is None:
                history.push(corrected, n_id[:core], offset, count)
            else:
                if corrected.is_cuda:
                    index = (pool._push_index + 1) % pool.pool_size
                    pool._push_stream(index).wait_stream(torch.cuda.current_stream(self.model.device))
                pool.async_push(corrected, offset, count, history.emb)
                pool.synchronize_push()
        self.post_commit_wall_seconds += time.perf_counter() - stamp
        self.corrected_blocks += 1
        self.corrected_rows += core
