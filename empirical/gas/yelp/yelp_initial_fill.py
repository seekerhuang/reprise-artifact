import time

def initial_hidden_only(model, loader):
    import torch
    if model.num_layers != 2 or model.batch_norm or model.linear or (model.dropout != 0.0) or (len(model.histories) != 1) or (model.pool is None) or (model.residual and model.convs[0].in_channels == model.convs[0].out_channels):
        raise RuntimeError('Only original Yelp L2/drop0/noBN/inactive residual is covered')
    model.eval()
    with torch.no_grad():
        batches = [sub_data + ({},) for sub_data in loader]
        covered = 0
        for (data, core, n_id, offset, count, state) in batches:
            x = data.x.to(model.device)
            adj_t = data.adj_t.to(model.device)
            out = model.forward_layer(0, x, adj_t, state)[:core]
            model.pool.async_push(out, offset, count, model.histories[0].emb)
            covered += core
        model.pool.synchronize_push()
        if covered != model.num_nodes or len(batches) != 8:
            raise RuntimeError('Initial hidden fill must cover every node exactly once')
    return dict(first_layer_calls=len(batches), last_layer_calls=0, hidden_nodes=covered, subsequent_refresh_policy='All original500 complete layerwise mini_test calls unchanged', scope='Original same-order layer0 and history push/drain. Initial unused _out absent; pull streams/buffers/output allocation may be deferred to later required work.')
