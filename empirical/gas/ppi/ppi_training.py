from empirical.gas.ppi import ppi_radial as radial
RADIAL_METRIC = 'euclidean'

def make_train(edge_dropout_fn, mode, period):
    import torch
    if mode != 'prefix' or period != 2:
        raise ValueError('R59 preserves Head2/mean VJP1/Body2; only final BN structural gradient completion is new')

    def train(model, loader, criterion, optimizer, max_steps, grad_norm=None, edge_dropout=0.0):
        if grad_norm is not None or edge_dropout != 0.0 or model.dropout != 0.0:
            raise ValueError('This fixed PPI adapter requires no grad clip, edge/input dropout')
        model.train()
        if not hasattr(model, '_r59_radial'):
            model._r59_radial = radial.RadialFeedback(model, RADIAL_METRIC)
        structured = model._r59_radial
        if not hasattr(model, '_ppi_all_labeled_loss_work'):
            model._ppi_all_labeled_loss_work = dict(version='PPI-S-all-labeled-Ngt1-no-mask-upload-loss-work-v2', batches_seen=0, empty_skips=0, direct_contexts=0, singleton_fallback_contexts=0, partial_contexts=0, contexts_completed=0, queries=0, direct_queries=0, indexed_queries=0, mask_uploads=0, mask_upload_skips=0)
        _PPI_loss_work = model._ppi_all_labeled_loss_work
        for (i, (batch, batch_size, *args)) in enumerate(loader):
            x = batch.x.to(model.device)
            adj = batch.adj_t.to(model.device)
            y = batch.y[:batch_size].to(model.device)
            host_mask = batch.train_mask[:batch_size]
            if host_mask.device.type != 'cpu':
                raise RuntimeError('Original GAS loader must supply CPU masks')
            if host_mask.dtype != torch.bool or host_mask.ndim != 1:
                raise RuntimeError('The native loader CPU boolean row mask is required')
            count = int(host_mask.sum())
            all_labeled_core_rows = count == batch_size and batch_size > 1
            _PPI_loss_work['batches_seen'] += 1
            if count == 0:
                _PPI_loss_work['empty_skips'] += 1
            elif all_labeled_core_rows:
                _PPI_loss_work['direct_contexts'] += 1
            elif batch_size == 1:
                _PPI_loss_work['singleton_fallback_contexts'] += 1
            else:
                _PPI_loss_work['partial_contexts'] += 1
            mask = None if all_labeled_core_rows else host_mask.nonzero(as_tuple=True)[0].to(model.device)
            _PPI_loss_work['mask_uploads'] += int(not all_labeled_core_rows)
            _PPI_loss_work['mask_upload_skips'] += int(all_labeled_core_rows)
            if count == 0:
                continue
            head = model.lins[1]
            saved_forward = head.forward
            try:
                head.forward = lambda h: h
                with structured.capture():
                    cut = model(x, adj, batch_size, *args)
            finally:
                head.forward = saved_forward
            structured.prepare(cut)
            query = cut.detach().requires_grad_(True)
            cotangent = torch.zeros_like(query)
            pending_head2_grad = None
            for query_position in range(period):
                optimizer.zero_grad(set_to_none=True)
                query.grad = None
                loss = criterion(head(query), y) if all_labeled_core_rows else criterion(head(query)[mask], y[mask])
                loss.backward()
                _PPI_loss_work['queries'] += 1
                _PPI_loss_work['direct_queries'] += int(all_labeled_core_rows)
                _PPI_loss_work['indexed_queries'] += int(not all_labeled_core_rows)
                cotangent.add_(query.grad.detach(), alpha=1.0 / period)
                structured.measure(query.grad)
                if query_position == 0:
                    optimizer.step()
                else:
                    pending_head2_grad = tuple((p.grad for p in head.parameters()))
                    if any((g is None for g in pending_head2_grad)):
                        raise RuntimeError('The actual affine Head must have both explicit gradients')
            optimizer.zero_grad(set_to_none=True)
            cut.backward(cotangent)
            structured.complete_measurement(optimizer)
            structured.install(0)
            optimizer.step()
            structured.install(1)
            for (p, g) in zip(head.parameters(), pending_head2_grad):
                p.grad = g
            optimizer.step()
            for p in head.parameters():
                p.grad = None
            pending_head2_grad = None
            structured.finish()
            _PPI_loss_work['contexts_completed'] += 1
            if i + 1 >= max_steps and i + 1 < len(loader):
                break
        return 0.0
    return train
