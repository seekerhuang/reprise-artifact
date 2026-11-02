import copy
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from empirical.data.datasets import adjacency, load
from empirical.lmc.lmc_loader import LMCLoader
from main.common import Output, Timer, score
from main.initialization import load_initial, restore
from main.models import DoubleBNGCN
from main.recipes import set_rate
from main.recording import Recording
from main.verification import expected_clocks, expected_bn


def run(args, recipe):
    from torch_geometric_autoscale import EvalSubgraphLoader, permute
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    device = torch.device(args.device)
    recording = Recording(device)
    timer = Timer(device, recording, 'setup_seconds')
    with timer:
        with recording.phase('data_partition_and_normalization_seconds', group='setup'):
            raw = load(recipe['dataset'], args.data_dir)
            saved = torch.load(args.data_dir / (recipe['dataset'] + '-lmc-partition.pt'), weights_only=True)
            matrix = adjacency(raw, symmetric=recipe['dataset'] == 'arxiv')
            data = Data(x=raw['x'], y=raw['y'], adj_t=matrix,
                        train_mask=raw['train_mask'], val_mask=raw['val_mask'], test_mask=raw['test_mask'])
            data = permute(data, saved['permutation'], log=False)
            data.adj_t = gcn_norm(data.adj_t.set_diag(), add_self_loops=False)
            ptr = saved['ptr']
            author = SimpleNamespace(package=SimpleNamespace(LMCLoader=LMCLoader))
        with recording.phase('model_initial_optimizer_seconds', group='setup'):
            model = DoubleBNGCN(data.x.shape[1], int(data.y.max()) + 1, recipe['layers'],
                                recipe['dropout'], recipe['dataset'] == 'flickr').to(device)
            rng = load_initial(model, args)
            optimizer = torch.optim.Adam([
                dict(params=model.reg_modules.parameters(), weight_decay=0.0),
                dict(params=model.nonreg_modules.parameters(), weight_decay=0.0),
            ], lr=0.01)
        with recording.phase('geometry_and_delivery_setup_seconds', group='setup'):
            if recipe['dataset'] == 'arxiv':
                from empirical.lmc.arxiv import lmc_arxiv_core as core
                from empirical.lmc.arxiv.lmc_arxiv_structural import LMCStructural
                geometry = core.build_geometry(author, data, ptr, device)
                engine = LMCStructural(model, optimizer)
            else:
                from empirical.lmc.flickr import lmc_flickr_core as core
                from empirical.lmc.flickr.lmc_flickr_geometry import build_geometry
                from empirical.lmc.flickr.lmc_flickr_structural import FrozenMeanP2
                from empirical.lmc.lmc_population import off_statistics, install_statistics
                geometry = build_geometry(author, data, ptr, device)
                engine = FrozenMeanP2(model, optimizer)
                loader = EvalSubgraphLoader(data, ptr, batch_size=recipe['eval_parts'], log=False)
                populations = {branch: torch.cat([
                    item.n_id[:item.batch_size] if branch == 'bn_ib' else item.n_id[item.batch_size:]
                    for item in loader.dataset]).to(device) for branch in ('bn_ib', 'bn_ob')}
                with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                    clone = copy.deepcopy(model)
        for group in optimizer.param_groups:
            group['lr'] = recipe['lr']
        optimizer.defaults['lr'] = recipe['lr']
        with recording.phase('sequence_load_seconds', group='setup'):
            sequence = torch.load(args.data_dir / (recipe['dataset'] + '-lmc-sequence.pt'), weights_only=True)
        with recording.phase('initial_rng_restore_seconds', group='setup'):
            restore(rng, device)
    timer.name = 'training_outer_seconds'
    recording.identify(args, recipe, model, optimizer)
    output = Output(args.out, recording)

    def evaluate_metrics(candidate):
        logits = core.four_group_logits(candidate, geometry)
        ids, labels = geometry.split_ids, geometry.labels
        return (score(logits[ids['valid']], labels[ids['valid']]),
                score(logits[ids['test']], labels[ids['test']]))

    updates = 0
    try:
        for epoch in range(recipe['epochs']):
            with timer:
                model.train()
                for key in sequence[2 * epoch:2 * epoch + 2]:
                    condition = geometry.conditions[int(key)]
                    if recipe['dataset'] == 'arxiv':
                        engine.update_block(condition)
                    else:
                        engine.block(condition)
                    updates += 2
                    set_rate(optimizer, recipe, updates)
            with torch.no_grad():
                if recipe['dataset'] == 'flickr':
                    clone.load_state_dict(model.state_dict(), strict=True)
                    clone.eval()
                    delivery = Timer(device, recording, 'delivery_preparation_seconds')
                    with delivery:
                        values = off_statistics(clone, geometry, populations)
                        install_statistics(clone, values)
                    delivered = clone
                else:
                    delivered = model
                with recording.phase('evaluation_seconds'):
                    valid, test = evaluate_metrics(delivered)
                recording.costs['single_delivery_preparation_seconds'] = delivery.seconds if recipe['dataset'] == 'flickr' else 0.0
                output.observe(delivered, updates, timer.seconds + (delivery.seconds if recipe['dataset'] == 'flickr' else 0.0),
                               valid, test)
        expected = 4 * recipe['epochs']
        if recipe['dataset'] == 'flickr':
            engine.clocks()
        unused_prefix = f"bns.{recipe['layers'] - 1}."
        unused = tuple(name for name, _ in model.named_parameters() if name.startswith(unused_prefix))
        unused_bn = tuple(name for name, _ in model.named_buffers() if name.startswith(unused_prefix))
        output.finish(model, updates, evaluate_metrics, expected_updates=expected, optimizer=optimizer,
                      clocks=expected_clocks(model, expected, unused),
                      bn_clocks=expected_bn(model, expected // 2, unused_bn),
                      counts=(len(geometry.split_ids['valid']), len(geometry.split_ids['test'])))
    finally:
        output.close()
