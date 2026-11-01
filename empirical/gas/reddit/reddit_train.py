def make_train(engine):

    def train(model, loader, criterion, optimizer, max_steps, grad_norm=None, edge_dropout=0.0):
        import torch
        if model is not engine.model or optimizer is not engine.optimizer or grad_norm is not None or (edge_dropout != 0.0) or (max_steps != 2) or (len(loader) != 2):
            raise ValueError('Locked two-batch input interval followed immediately by native full refresh only')
        visited = 0
        try:
            for (i, (batch, core, *args)) in enumerate(loader):
                (x, adj) = (batch.x.to(model.device), batch.adj_t.to(model.device))
                labels = batch.y[:core].to(model.device)
                host_mask = batch.train_mask[:core]
                if host_mask.device.type != 'cpu':
                    raise RuntimeError('Original CPU core mask required')
                mask = host_mask
                count = int(host_mask.sum())
                row_ids = host_mask.nonzero(as_tuple=False).flatten().to(model.device)
                if not count:
                    raise RuntimeError('The locked two-prefix schedule requires both actual cores to have supervision')
                engine.next_prefix_before_refresh = i == 0
                losses = engine.block(x, adj, core, args, labels, mask, criterion, label_count=count, row_ids=row_ids)
                visited += 1
            if visited != 2:
                raise RuntimeError('Input interval did not complete two prefixes')
            engine.completed_input_intervals += 1
            return 0.0
        finally:
            engine.next_prefix_before_refresh = None
    return train
