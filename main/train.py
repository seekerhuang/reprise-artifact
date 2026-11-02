from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from empirical.data.datasets import adjacency, load, scipy_csr, splits
from main.common import Output, Timer, score
from main.recipes import DIRECT_CONFIGS, set_rate
from main.initialization import load_initial, restore
from main.recording import Recording
from main.verification import expected_clocks, expected_bn


def run(args, recipe):
    if recipe['family'] == 'iglu':
        return run_iglu(args, recipe)
    if recipe['family'] == 'sage':
        return run_sage(args, recipe)
    return run_gcn(args, recipe)


def run_gcn(args, recipe):
    from main.models import GCN, TOPGCN
    direct = getattr(args, 'method', 'reprise') == 'direct'
    device = torch.device(args.device)
    recording = Recording(device, memory=args.config in DIRECT_CONFIGS)
    timer = Timer(device, recording, 'setup_seconds')
    with timer:
        with recording.phase('data_load_and_transfer_seconds', group='setup'):
            raw = load(recipe['dataset'], args.data_dir)
            labels, ids = raw['y'].to(device), splits(raw, device)
        if recipe['family'] == 'top':
            with recording.phase('graph_preparation_seconds', group='setup'):
                from main.native.dgl_adapter import configure
                configure()
                from main.top.arxiv_coupled_structural_r64_v1 import CoupledStructural
                from empirical.data.top_data import prepare
                x, matrix = prepare(raw, device)
            with recording.phase('model_initial_optimizer_seconds', group='setup'):
                model = TOPGCN(x.shape[1], 256, int(labels.max()) + 1, recipe['dropout']).to(device)
                rng = load_initial(model, args)
                optimizer = torch.optim.Adam(model.parameters(), lr=recipe['lr'])
            with recording.phase('engine_construction_seconds', group='setup'):
                if direct:
                    from main.top.direct import Direct
                    engine = Direct(model, matrix, x, training_ids=ids['train'])
                else:
                    engine = CoupledStructural(model, matrix, x, training_ids=ids['train'])
            engine.features = x.new_empty((0, x.shape[1]))
            del x
            evaluate = engine.forward
        else:
            with recording.phase('graph_preparation_seconds', group='setup'):
                x = raw['x'].to(device)
                from main.products_gcn.products_equal_body_small_head_graph_v1 import EqualBodySmallSecondStructural
                from empirical.data.products_data import normalize_raw_csr
                matrix = normalize_raw_csr(raw['sample_rowptr'].to(device), raw['sample_col'].to(device), len(x))
            with recording.phase('model_initial_optimizer_seconds', group='setup'):
                model = GCN(x.shape[1], 256, int(labels.max()) + 1, 3, recipe['dropout']).to(device)
                rng = load_initial(model, args)
                optimizer = torch.optim.Adam([
                    dict(params=model.reg_modules.parameters(), weight_decay=0.0),
                    dict(params=model.nonreg_modules.parameters(), weight_decay=0.0),
                ], lr=recipe['lr'], eps=recipe['eps'])
            with recording.phase('engine_construction_seconds', group='setup'):
                engine = EqualBodySmallSecondStructural(model, optimizer, matrix, x, ids['train'], labels)
                evaluate = engine.logits
        with recording.phase('initial_rng_restore_seconds', group='setup'):
            restore(rng, device)
    timer.name = 'training_outer_seconds'
    recording.identify(args, recipe, model, optimizer)
    output = Output(args.out, recording)

    def evaluate_metrics(candidate):
        logits = evaluate()
        return (score(logits[ids['valid']], labels[ids['valid']]),
                score(logits[ids['test']], labels[ids['test']]))

    try:
        for update in range(1 if direct else 2, recipe['updates'] + 1, 1 if direct else 2):
            with timer:
                model.train()
                if recipe['family'] == 'top':
                    engine.update_block(optimizer, labels[ids['train']], 'A')
                else:
                    engine.update_block()
                set_rate(optimizer, recipe, update)
            if update % recipe['eval_every']:
                continue
            with torch.no_grad(), recording.phase('evaluation_seconds'):
                model.eval()
                valid, test = evaluate_metrics(model)
            output.observe(model, update, timer.seconds, valid, test)
        unused = ('bns.2.weight', 'bns.2.bias')
        if recipe['family'] == 'products_gcn':
            with recording.phase('training_close_seconds'):
                engine.verify()
        output.finish(model, update, evaluate_metrics, expected_updates=recipe['updates'], optimizer=optimizer,
                      clocks=expected_clocks(model, recipe['updates'], unused),
                      bn_clocks=expected_bn(model, recipe['updates'] // (1 if direct else 2),
                                            ('bns.2.num_batches_tracked',)), counts=(len(ids['valid']), len(ids['test'])))
    finally:
        output.close()


@torch.no_grad()
def sage_logits(model, matrix, features):
    value = features
    for index, layer in enumerate(model.convs):
        mean = torch.sparse.mm(matrix, value)
        value = F.linear(mean, layer.lin_l.weight, layer.lin_l.bias) + F.linear(value, layer.lin_r.weight)
        if index < len(model.convs) - 1:
            bn = model.bns[index]
            value = F.relu(F.batch_norm(value, bn.running_mean, bn.running_var,
                                       bn.weight, bn.bias, training=False, momentum=0.0, eps=bn.eps))
    return value.log_softmax(-1)


def run_sage(args, recipe):
    from main.native.dgl_adapter import configure
    configure()
    import dgl
    from main.sage.sage_model import HistorySAGE
    from main.sage.train_frequency_loss_v1 import weights_from_train_labels
    direct = getattr(args, 'method', 'reprise') == 'direct'
    device = torch.device(args.device)
    recording = Recording(device, memory=args.config in DIRECT_CONFIGS)
    timer = Timer(device, recording, 'setup_seconds')
    with timer:
        with recording.phase('data_load_and_transfer_seconds', group='setup'):
            from main.sage.sage_paid_startup_overlap_v1 import load_training_arrays
            raw = load_training_arrays(lambda directory, name, device: load(name, directory),
                                       args.data_dir, recipe['dataset'], device)
            features, labels = raw['x'].to(device), raw['y'].to(device)
            ids = splits(raw, device)
        with recording.phase('evaluation_graph_preparation_seconds', group='setup'):
            import scipy.sparse as sp
            sampled = sp.csr_matrix((np.ones(len(raw['sample_col']), dtype=np.float32),
                                     raw['sample_col'].numpy(), raw['sample_rowptr'].numpy()),
                                    shape=(len(features), len(features)))
            normalized = (sampled + sp.eye(len(features), dtype=np.float32)).tocsr()
            degree = np.asarray(normalized.sum(1)).ravel().astype(np.float32)
            normalized.data *= np.repeat(1.0 / degree, np.diff(normalized.indptr))
            matrix = torch.sparse_csr_tensor(torch.from_numpy(normalized.indptr.astype(np.int64)),
                                             torch.from_numpy(normalized.indices.astype(np.int64)),
                                             torch.from_numpy(normalized.data), size=normalized.shape, device=device)
        with recording.phase('model_initial_optimizer_seconds', group='setup'):
            model = HistorySAGE(features.shape[1], 256, int(labels.max()) + 1, 3,
                                recipe['dropout'], 'CSR', {}, SimpleNamespace(method='grad')).to(device)
            rng = load_initial(model, args)
            optimizer = torch.optim.Adam(model.parameters(), lr=recipe['lr'], betas=(recipe.get('beta1', 0.9), 0.999),
                                         **({'fused': True} if not recipe['static_first'] else {}))
            scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe['cosine_horizon'])
                         if 'cosine_horizon' in recipe else None)
        with recording.phase('sampling_graph_and_engine_seconds', group='setup'):
            graph = dgl.graph(('csc', (raw['sample_rowptr'], raw['sample_col'], torch.empty(0, dtype=torch.int64))), num_nodes=len(features))
            if recipe['static_first']:
                from main.sage.fresh_arxiv_structural import FreshArxivStructural
                graph = dgl.add_self_loop(graph).int().formats(['csc']).to(device)
                dgl.seed(123)
                engine = FreshArxivStructural(model, optimizer, dgl, graph, features,
                                              torch.sparse.mm(matrix, features), labels,
                                              sampler=recipe['sampler'], fanouts=recipe['fanouts'])
            else:
                from main.sage.sage_direct_self_loop_csc_v1 import candidate
                from main.sage.sage_exact_head_graph_v1 import ExactHeadSAGE
                graph = candidate(dgl, graph).to(device)
                dgl.seed(123)
                if direct:
                    from main.sage.direct import Direct
                    engine = Direct(model, optimizer, dgl, graph, features, labels, fanouts=recipe['fanouts'])
                else:
                    engine = ExactHeadSAGE(model, optimizer, dgl, graph, features, labels,
                                           first_fanout=10, fanouts=(10, 10), sampler='labor')
                engine.sampler.set_seed(123)
                engine.loss_weights, _ = weights_from_train_labels(labels[ids['train']], int(labels.max()) + 1, recipe['loss_alpha'])
        with recording.phase('initial_rng_restore_seconds', group='setup'):
            torch.set_rng_state(rng[0])
            if rng[1] is not None:
                torch.cuda.set_rng_state(rng[1], device)
    timer.name = 'training_outer_seconds'
    if not recipe['static_first']:
        recording.costs['evaluation_setup_excluded_seconds'] = recording.costs['setup']['evaluation_graph_preparation_seconds']
    recording.identify(args, recipe, model, optimizer)
    output = Output(args.out, recording)

    def evaluate_metrics(candidate):
        torch.backends.cuda.matmul.allow_tf32 = False
        logits = sage_logits(candidate, matrix, features)
        return (score(logits[ids['valid']], labels[ids['valid']]),
                score(logits[ids['test']], labels[ids['test']]))

    evaluation_steps = set(range(recipe['eval_every'], recipe['updates'] + 1, recipe['eval_every'])) | {recipe['updates']}
    evaluation_steps |= {n for n in (20, 50, 100, 150, 200, 300, 500, 750, 986, 1000) if n <= recipe['updates']}
    order, cursor = None, 0
    try:
        while engine.actual_updates < recipe['updates']:
            with timer:
                torch.backends.cuda.matmul.allow_tf32 = True
                model.train()
                if order is None or cursor >= len(order):
                    order = ids['train'][torch.randperm(len(ids['train']), device=device)]
                    cursor = 0
                roots = order[cursor:cursor + recipe['batch_size']].to(graph.idtype)
                cursor += len(roots)
                engine.set_context(roots)
                engine.update_block(min(1 if direct else 2, recipe['updates'] - engine.actual_updates), scheduler)
                if engine.actual_updates == recipe['updates'] and hasattr(engine, '_exact_head'):
                    engine._exact_head.close()
            if engine.actual_updates not in evaluation_steps:
                continue
            with torch.no_grad(), recording.phase('evaluation_seconds'):
                torch.backends.cuda.matmul.allow_tf32 = False
                model.eval()
                valid, test = evaluate_metrics(model)
            output.observe(model, engine.actual_updates, timer.seconds, valid, test)
        output.finish(model, engine.actual_updates, evaluate_metrics, expected_updates=recipe['updates'],
                      optimizer=optimizer, clocks=expected_clocks(model, recipe['updates']),
                      bn_clocks=expected_bn(model, (recipe['updates'] + (0 if direct else 1)) // (1 if direct else 2)),
                      tolerances=(0.0, 1 / len(ids['test'])) if recipe['dataset'] == 'products' else (0.0, 0.0),
                      counts=(len(ids['valid']), len(ids['test'])))
    finally:
        output.close()


def run_iglu(args, recipe):
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler
    from main.iglu.iglu_model import Target, TF_NAMES, SHAPES, csr, original_normalize
    from main.iglu.iglu_structural_r66_v1 import IGLUStructural
    device = torch.device(args.device)
    recording = Recording(device)
    timer = Timer(device, recording, 'setup_seconds')
    with timer:
        with recording.phase('data_and_geometry_preparation_seconds', group='setup'):
            raw = load('arxiv', args.data_dir)
            split = splits(raw, device)
            train = raw['train_mask'].nonzero().flatten().numpy()
            scaler = StandardScaler().fit(raw['x'].numpy()[train])
            features = torch.as_tensor(scaler.transform(raw['x'].numpy()), dtype=torch.float32, device=device)
            labels = raw['y'].to(device)
            full = scipy_csr(raw)
            full.sum_duplicates()
            full.data = np.ones(full.nnz, dtype=np.float64)
            full = full.maximum(full.T)
            full_norm = original_normalize(full + sp.eye(len(features))).astype(np.float32).tocsr()
            mask = raw['train_mask'].numpy().astype(np.float64)
            training = sp.diags(mask) @ full @ sp.diags(mask)
            training = original_normalize(training + sp.diags(mask)).astype(np.float32).tocsr()
            p, pt = csr(full_norm, device), csr(full_norm.T, device)
            replay = np.random.RandomState(2026092901)
            shuffled = train[replay.permutation(np.arange(len(train)))]
            batches = [shuffled[i:i + 10000] for i in range(0, len(shuffled), 10000)]
            all_ids = np.concatenate([raw[field].nonzero().flatten().numpy()
                                      for field in ('train_mask', 'val_mask', 'test_mask')])
            replay.permutation(np.arange(len(all_ids)))
            conditions = []
            for batch in batches:
                local = training[batch].tocsr()
                conditions.append(SimpleNamespace(matrix=csr(local, device), transpose=csr(local.T, device),
                                                  labels=labels[torch.as_tensor(batch, device=device)]))
            geometry = SimpleNamespace(first=torch.sparse.mm(p, features), matrix=p, transpose=pt,
                                       labels=labels, splits=split, conditions=conditions)
        with recording.phase('model_initial_optimizer_seconds', group='setup'):
            tensors = {name: np.zeros(shape, dtype=np.float32) for name, shape in zip(TF_NAMES, SHAPES)}
            model = Target(tensors, device=device)
            rng = load_initial(model, args)
            engine = IGLUStructural(model, geometry, dropout=recipe['dropout'])
        with recording.phase('initial_rng_restore_seconds', group='setup'):
            restore(rng, device)
    timer.name = 'training_outer_seconds'
    recording.identify(args, recipe, model, engine.optimizers)
    output = Output(args.out, recording)

    def evaluate_metrics(candidate):
        logits = candidate(geometry.first, p, pt)
        return (score(logits[split['valid']], labels[split['valid']]),
                score(logits[split['test']], labels[split['test']]))

    try:
        for epoch in range(recipe['epochs']):
            with timer:
                torch.backends.cuda.matmul.allow_tf32 = True
                model.train()
                rate = 0.001 if epoch < 10 else 0.0005 if epoch < 100 else 0.0001
                order = replay.choice(np.arange(10), 10, replace=False)
                replay.choice(np.arange(17), 17, replace=False)
                for batch in order:
                    engine.update_block(int(batch), rate)
            with torch.no_grad(), recording.phase('evaluation_seconds'):
                model.eval()
                valid, test = evaluate_metrics(model)
            output.observe(model, engine.actual_updates, timer.seconds, valid, test)
        engine.clocks()
        expected = 2 * recipe['batches'] * recipe['epochs']
        output.finish(model, engine.actual_updates, evaluate_metrics, expected_updates=expected,
                      optimizer=engine.optimizers, clocks=expected_clocks(model, expected), bn_clocks={},
                      counts=(len(split['valid']), len(split['test'])))
    finally:
        output.close()
