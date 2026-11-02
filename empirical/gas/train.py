import copy
from contextlib import nullcontext

import torch
from torch_geometric.data import Data

from empirical.data.datasets import adjacency, load, partition
from main.common import Output, Timer, score
from main.initialization import load_initial, restore
from main.recipes import set_rate
from main.recording import Recording
from main.verification import expected_clocks, expected_bn


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
    dataset = recipe['dataset']
    ppi = dataset == 'ppi'
    direct = getattr(args, 'method', 'reprise') == 'direct'
    recording = Recording(device, memory=direct)
    timer = Timer(device, recording, 'setup_seconds')
    with timer:
        with recording.phase('raw_input_seconds', group='setup'):
            raw = load(dataset, args.data_dir, 'train' if ppi else None)
        with recording.phase('input_and_partition_seconds', group='setup'):
            data = _data(raw, dataset in ('reddit2', 'yelp'))
            if dataset == 'yelp':
                from empirical.gas.yelp.yelp_partition import balanced_partition
                permutation, ptr = balanced_partition(data.num_nodes)
            else:
                saved = torch.load(args.data_dir / (dataset + '-partition.pt'), weights_only=True)
                permutation, ptr = saved['permutation'], saved['ptr']
        with recording.phase('permutation_and_normalization_seconds', group='setup'):
            data = permute(data, permutation, log=False)
            data.adj_t = gcn_norm(data.adj_t.set_diag(), add_self_loops=False)
            if ppi:
                recipe = dict(recipe, batch_norm=True, residual=True, drop_input=False)
                with recording.phase('inductive_data_preparation_seconds', group='setup'):
                    evaluation_data = {split: _data(load('ppi', args.data_dir, split)) for split in ('val', 'test')}
                with recording.phase('inductive_normalization_seconds', group='setup'):
                    for value in evaluation_data.values():
                        value.adj_t = gcn_norm(value.adj_t.set_diag(), add_self_loops=False)
        architecture = dict(hidden_channels=recipe['hidden'], num_layers=2,
                            dropout=recipe['dropout'], batch_norm=recipe['batch_norm'],
                            residual=recipe['residual'], drop_input=recipe['drop_input'], linear=ppi)
        classes = data.y.shape[1] if data.y.ndim == 2 else int(data.y.max()) + 1
        with recording.phase('loader_construction_seconds', group='setup'):
            if dataset == 'reddit2':
                from empirical.gas.reddit.reddit_feature_elision import training_view_without_X
                view, _ = training_view_without_X(data)
                from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader
                train = SubgraphLoader(view, ptr, batch_size=recipe['batch_parts'], shuffle=True, num_workers=0)
                evaluation = EvalSubgraphLoader(data, ptr, batch_size=recipe['batch_parts'], log=False)
            else:
                train, evaluation = _loaders(data, ptr, recipe, device)
        with recording.phase('model_and_initial_load_seconds', group='setup'):
            buffer_size = max(sub.n_id.numel() for sub in evaluation)
            model = GCN(data.num_nodes, data.x.shape[1], out_channels=classes,
                        pool_size=2, buffer_size=buffer_size, **architecture).to(device)
            rng = load_initial(model, args)
            if dataset == 'flickr':
                for bn in model.bns:
                    bn.momentum = recipe['bn_momentum']
        with recording.phase('optimizer_constructor_seconds', group='setup'):
            optimizer = torch.optim.Adam([
                dict(params=model.reg_modules.parameters(), weight_decay=recipe.get('reg_weight_decay', 0.0)),
                dict(params=model.nonreg_modules.parameters(), weight_decay=0.0),
            ], lr=recipe['lr'])
        with recording.phase('engine_and_delivery_setup_seconds', group='setup'):
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
                install(model, optimizer, beta1=recipe['terminal_beta1'])
                if direct:
                    from empirical.gas.ppi.direct import train as train_epoch
                else:
                    ppi_radial.RadialFeedback = adapt(ppi_radial.RadialFeedback, total_contexts=4500)
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
        from empirical.gas.clock import Clock
        clock = Clock(model)
        recording.close_callbacks.append(clock.close)
        with recording.phase('initial_rng_restore_seconds', group='setup'):
            restore(rng, device)
        with recording.phase('initial_history_fill_seconds', group='setup'):
            if dataset == 'yelp':
                from empirical.gas.yelp.yelp_initial_fill import initial_hidden_only
                initial_hidden_only(model, evaluation)
            else:
                _refresh(model, evaluation)
    timer.name = 'training_outer_seconds'
    if ppi:
        recording.costs['evaluation_setup_excluded_seconds'] = sum(recording.costs['setup'][name] for name in
            ('inductive_data_preparation_seconds', 'inductive_normalization_seconds'))
    recording.identify(args, recipe, model, optimizer)
    output = Output(args.out, recording)

    def evaluate_metrics(candidate, checkpoint=None):
        candidate.eval()
        current_data, current_evaluation = data, evaluation
        if (dataset == 'yelp' and checkpoint is not None and phases
                and checkpoint['updates'] <= recipe['random_partition_until']):
            current_data, _, current_evaluation = phases[0]
        if dataset in ('flickr', 'ppi'):
            clone.load_state_dict(candidate.state_dict(), strict=True)
            clone.eval()
            if ppi:
                logits = {key: clone(value.x.to(device), value.adj_t.to(device)).cpu()
                          for key, value in evaluation_data.items()}
                return score(logits['val'], evaluation_data['val'].y), score(logits['test'], evaluation_data['test'].y)
            logits = clone(data.x.to(device), data.adj_t.to(device)).cpu()
        else:
            logits = _refresh(candidate, current_evaluation)
        return (score(logits[current_data.val_mask], current_data.y[current_data.val_mask]),
                score(logits[current_data.test_mask], current_data.y[current_data.test_mask], 0.45 if dataset == 'yelp' else 0.5))

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
                refresh = Timer(device, recording, 'history_refresh_seconds')
                with refresh:
                    refreshed = _refresh(model, evaluation)
                if ppi:
                    average.capture(updates)
            delivery = Timer(device, recording, 'delivery_preparation_seconds')
            with torch.no_grad(), recording.phase('delivery_observer_seconds'):
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
                    valid = score(logits[data.val_mask], data.y[data.val_mask])
                    test = score(logits[data.test_mask], data.y[data.test_mask], threshold)
                    delivered = model
                recording.costs['pure_history_output_seconds'] = clock.output_seconds
                recording.costs['shared_pool_setup_seconds'] = clock.shared_setup_seconds
                recording.costs['current_evaluation_refresh_excluded_seconds'] = refresh.seconds
                recording.costs['prior_history_outputs_excluded_seconds'] = clock.prior_output(True)
                recording.costs['single_delivery_preparation_seconds'] = delivery.seconds
                if ppi:
                    recording.costs['averaging_seconds'] = average.required_snapshot_seconds
                output.observe(delivered, updates,
                               timer.seconds - refresh.seconds + delivery.seconds - clock.prior_output(True), valid, test)
            if dataset == 'reddit2':
                average.end()
        expected = 2 * recipe['max_steps'] * recipe['epochs']
        unused = tuple(name for name, _ in model.named_parameters()
                       if not ppi and name.startswith('bns.') and (not recipe['batch_norm'] or name.startswith('bns.1.')))
        unused_bn = tuple(name for name, _ in model.named_buffers()
                          if not ppi and name.startswith('bns.') and (not recipe['batch_norm'] or name.startswith('bns.1.')))
        if ppi:
            output.full_seconds = output.last_row['seconds'] - delivery.seconds + output.best_delivery_seconds
        tolerance = ((1 / int(data.val_mask.sum()), 1 / int(data.test_mask.sum()))
                     if dataset in ('flickr', 'reddit2') else
                     tuple(2 / max(1, int((data.y[mask] > 0).sum())) for mask in (data.val_mask, data.test_mask))
                     if dataset == 'yelp' else (2e-5, 2e-5))
        output.finish(model, updates, evaluate_metrics, expected_updates=expected, optimizer=optimizer,
                      clocks=expected_clocks(model, expected, unused),
                      bn_clocks=expected_bn(model, expected // (1 if direct else 2), unused_bn),
                      tolerances=tolerance,
                      counts=(int(data.val_mask.sum()), int(data.test_mask.sum())) if data.y.ndim == 1 else None,
                      readback=evaluate_metrics)
    finally:
        if dependency is not None:
            dependency.restore()
        output.close()
