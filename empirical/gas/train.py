import copy
from contextlib import nullcontext

import torch
from torch_geometric.data import Data

from empirical.data.datasets import adjacency, load, partition
from main.common import Output, Timer, score
from main.initialization import load_initial, restore
from main.recipes import set_rate


def _data(raw, normalize_features=False):
    x = raw['x']
    if normalize_features:
        x = (x - x.mean(dim=0)) / x.std(dim=0)
    return Data(x=x, y=raw['y'], adj_t=adjacency(raw),
                train_mask=raw['train_mask'], val_mask=raw['val_mask'], test_mask=raw['test_mask'])


def _loaders(data, ptr, recipe, device):
    from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader
    train_class, eval_class = SubgraphLoader, EvalSubgraphLoader
    dataset = recipe['dataset']
    if dataset in ('flickr', 'yelp'):
        if dataset == 'flickr':
            from empirical.gas.flickr.flickr_resident import Registry
            from empirical.gas.flickr.flickr_fixed_groups import cached_class
        else:
            from empirical.gas.yelp.yelp_resident import Registry
        registry = Registry(device, lambda name: nullcontext())
        train_class, eval_class, _ = registry.classes(train_class, eval_class)
        if dataset == 'flickr':
            train_class = cached_class(train_class)
    train = train_class(data, ptr, batch_size=recipe['batch_parts'],
                        shuffle=dataset != 'flickr', num_workers=0)
    evaluation = eval_class(data, ptr, batch_size=recipe['batch_parts'], log=False)
    if dataset in ('flickr', 'yelp'):
        for sub in evaluation.dataset:
            sub.data.adj_t = sub.data.adj_t.to(device)
    if dataset == 'flickr':
        graphs = {id(sub.n_id): sub.data.adj_t.to(device) for sub in train._fixed_cache.values()}
        original = train.collate_fn

        def collate(batches):
            sub = original(batches)
            sub.data.adj_t = graphs[id(sub.n_id)]
            return sub
        train.collate_fn = collate
    return train, evaluation


def _refresh(model, loader):
    model.eval()
    with torch.no_grad():
        return model(loader=loader)


def run(args, recipe):
    from torch_geometric_autoscale import permute
    from torch_geometric_autoscale.models import GCN
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from empirical.gas.stream_dependency import Installation
    from empirical.gas import calibration
    device = torch.device(args.device)
    timer = Timer(device)
    dataset = recipe['dataset']
    ppi = dataset == 'ppi'
    with timer:
        raw = load(dataset, args.data_dir, 'train' if ppi else None)
        data = _data(raw, dataset in ('reddit2', 'yelp'))
        if dataset == 'yelp':
            from empirical.gas.yelp.yelp_partition import balanced_partition
            permutation, ptr = balanced_partition(data.num_nodes)
        else:
            saved = torch.load(args.data_dir / (dataset + '-partition.pt'), weights_only=True)
            permutation, ptr = saved['permutation'], saved['ptr']
        data = permute(data, permutation, log=False)
        data.adj_t = gcn_norm(data.adj_t.set_diag(), add_self_loops=False)
        if ppi:
            recipe = dict(recipe, batch_norm=True, residual=True, drop_input=False)
            evaluation_data = {split: _data(load('ppi', args.data_dir, split)) for split in ('val', 'test')}
            for value in evaluation_data.values():
                value.adj_t = gcn_norm(value.adj_t.set_diag(), add_self_loops=False)
        architecture = dict(hidden_channels=recipe['hidden'], num_layers=2,
                            dropout=recipe['dropout'], batch_norm=recipe['batch_norm'],
                            residual=recipe['residual'], drop_input=recipe['drop_input'], linear=ppi)
        classes = data.y.shape[1] if data.y.ndim == 2 else int(data.y.max()) + 1
        if dataset == 'reddit2':
            from empirical.gas.reddit.reddit_feature_elision import training_view_without_X
            view, _ = training_view_without_X(data)
            from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader
            train = SubgraphLoader(view, ptr, batch_size=recipe['batch_parts'], shuffle=True, num_workers=0)
            evaluation = EvalSubgraphLoader(data, ptr, batch_size=recipe['batch_parts'], log=False)
        else:
            train, evaluation = _loaders(data, ptr, recipe, device)
        buffer_size = max(sub.n_id.numel() for sub in evaluation)
        model = GCN(data.num_nodes, data.x.shape[1], out_channels=classes,
                    pool_size=2, buffer_size=buffer_size, **architecture).to(device)
        rng = load_initial(model, args)
        if dataset == 'flickr':
            for bn in model.bns:
                bn.momentum = recipe['bn_momentum']
        optimizer = torch.optim.Adam([
            dict(params=model.reg_modules.parameters(), weight_decay=recipe.get('reg_weight_decay', 0.0)),
            dict(params=model.nonreg_modules.parameters(), weight_decay=0.0),
        ], lr=recipe['lr'])
        dependency = Installation(model.pool) if dataset in ('reddit2', 'ppi') else None
        if dataset == 'reddit2':
            from empirical.gas.reddit.reddit_structural import Training
            from empirical.gas.reddit.reddit_train import make_train
            from empirical.gas.reddit.reddit_head_average import HeadAverage
            engine = Training(model, optimizer)
            engine.install_static_input(data.x, data.adj_t)
            train_epoch = make_train(engine)
            average = HeadAverage(model, engine, optimizer)
        elif ppi:
            from empirical.gas.ppi import ppi_radial
            from empirical.gas.ppi import ppi_training
            from empirical.gas.ppi.ppi_radial_recovery_graph_v1 import adapt
            from empirical.gas.ppi.ppi_terminal_beta import install
            from empirical.gas.ppi.ppi_sma21 import CausalSMA21
            ppi_radial.RadialFeedback = adapt(ppi_radial.RadialFeedback, total_contexts=4500)
            install(model, optimizer, beta1=recipe['terminal_beta1'])
            train_epoch = ppi_training.make_train(None, 'prefix', 2)
            average = CausalSMA21(model, device)
        else:
            if dataset == 'flickr':
                from empirical.gas.flickr.flickr_structural import FlickrTraining as Training
            else:
                from empirical.gas.yelp.yelp_structural import Training
            engine = Training(model, optimizer)
        if dataset in ('flickr', 'ppi'):
            calibration.compile_native(GCN)
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                clone = GCN(0, data.x.shape[1], out_channels=classes, buffer_size=None,
                            **architecture).to(device)
        criterion = (torch.nn.CrossEntropyLoss() if data.y.ndim == 1 else
                     torch.nn.BCEWithLogitsLoss(pos_weight=data.x.new_tensor(recipe['positive_weight']).to(device))
                     if dataset == 'yelp' else torch.nn.BCEWithLogitsLoss())
        restore(rng, device)
        if dataset == 'yelp':
            from empirical.gas.yelp.yelp_initial_fill import initial_hidden_only
            initial_hidden_only(model, evaluation)
        else:
            _refresh(model, evaluation)
    output = Output(args.out)
    updates = 0
    phases = []
    try:
        for epoch in range(recipe['epochs']):
            with timer:
                if dataset == 'yelp' and updates == recipe['random_partition_until']:
                    from empirical.gas.yelp.yelp_node_order import new_to_old_rows, migrate_history
                    model.pool.synchronize_push()
                    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                        new = partition(raw, recipe['parts'])
                        phases.append((data, train, evaluation))
                        rows = new_to_old_rows(permutation, new['permutation'])
                        data = permute(data, rows, log=False)
                        train, evaluation = _loaders(data, new['ptr'], recipe, device)
                        needed = max(max(sub.n_id.numel() for sub in evaluation.dataset),
                                     data.num_nodes - int((new['ptr'][1:] - new['ptr'][:-1]).sort().values[:5].sum()))
                        if needed > model.pool.buffer_size:
                            from torch_geometric_autoscale import AsyncIOPool
                            model.pool = AsyncIOPool(model.pool.pool_size, needed, model.pool.embedding_dim).to(device)
                            model.buffer_size = needed
                        migrate_history(model.histories[0], rows, producer_drained=True, complete_current_theta_history=True)
                model.train()
                if dataset in ('reddit2', 'ppi'):
                    train_epoch(model, train, criterion, optimizer, recipe['max_steps'])
                    updates += 2 * recipe['max_steps']
                else:
                    for index, (batch, core, *coordinates) in enumerate(train):
                        mask = batch.train_mask[:core]
                        count = int(mask.sum())
                        if count:
                            engine.block(batch.x.to(device), batch.adj_t.to(device), core,
                                         coordinates, batch.y[:core].to(device), mask.to(device), criterion,
                                         label_count=count)
                            updates += 2
                        if index + 1 >= recipe['max_steps']:
                            break
                set_rate(optimizer, recipe, updates)
                if dataset == 'reddit2':
                    average.begin(updates)
                refresh = Timer(device)
                with refresh:
                    refreshed = _refresh(model, evaluation)
                if ppi:
                    average.capture(updates)
            delivery = Timer(device)
            with torch.no_grad():
                if dataset in ('flickr', 'ppi'):
                    clone.load_state_dict(model.state_dict(), strict=True)
                    clone.eval()
                    with delivery:
                        if ppi:
                            average.install_on(clone)
                            payload = calibration.ppi_prepare(clone, data)
                        else:
                            payload = calibration.flickr_prepare(clone, data)
                    if ppi:
                        logits = {key: clone(value.x.to(device), value.adj_t.to(device)).cpu()
                                  for key, value in evaluation_data.items()}
                        valid = score(logits['val'], evaluation_data['val'].y)
                        test = score(logits['test'], evaluation_data['test'].y)
                    else:
                        logits = calibration.flickr_finish(clone, payload)
                        valid = score(logits[data.val_mask], data.y[data.val_mask])
                        test = score(logits[data.test_mask], data.y[data.test_mask])
                    delivered = clone
                else:
                    logits = refreshed
                    threshold = 0.45 if dataset == 'yelp' else 0.5
                    valid = score(logits[data.val_mask], data.y[data.val_mask], threshold)
                    test = score(logits[data.test_mask], data.y[data.test_mask], threshold)
                    delivered = model
                output.observe(delivered, updates, timer.seconds - refresh.seconds + delivery.seconds, valid, test)
            if dataset == 'reddit2':
                average.end()
    finally:
        if dependency is not None:
            dependency.restore()
        output.close()
