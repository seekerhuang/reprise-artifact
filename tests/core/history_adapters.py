from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

from graph_adapters import affine_case
from reference import check_block


def check(name, device):
    from torch_geometric_autoscale import SubgraphLoader
    from torch_geometric_autoscale.models import GCN
    from empirical.gas.flickr.flickr_structural import FlickrTraining
    from empirical.gas.yelp.yelp_structural import Training as YelpTraining
    from empirical.gas.reddit.reddit_structural import Training as RedditTraining
    from empirical.gas.ppi import ppi_training, ppi_radial
    from empirical.gas.ppi.ppi_radial_recovery_graph_v1 import adapt
    from empirical.gas.ppi.ppi_terminal_beta import install
    from empirical.gas.stream_dependency import Installation
    from empirical.pna.model import PNA
    from empirical.pna.training import Training as PNATraining
    from main.recipes import RECIPES
    from prepare_initial import DIMENSIONS
    recipe = RECIPES[name]
    dataset, pna, ppi = recipe['dataset'], recipe['family'] == 'pna', name == 'PPI-GCN'
    inputs, classes = DIMENSIONS[dataset]
    n = 80
    ids = torch.arange(n, device=device)
    rows = ids.repeat_interleave(4)
    cols = torch.stack([(ids + 1) % n, (ids - 1) % n, (ids + 5) % n, (ids - 5) % n], 1).reshape(-1)
    matrix = SparseTensor(row=rows, col=cols, sparse_sizes=(n, n))
    if not pna:
        matrix = gcn_norm(matrix.set_diag(), add_self_loops=False)
    labels = torch.randint(0, 2, (n, classes), device=device).float() if dataset in ('yelp', 'ppi') else ids % classes
    data = Data(x=torch.randn(n, inputs, device=device).cpu(), y=labels.cpu(), adj_t=matrix.cpu(),
                train_mask=(torch.ones(n, dtype=torch.bool) if ppi else (ids % 3 != 0).cpu()))
    loader = SubgraphLoader(data, torch.arange(0, n + 1, 4), batch_size=10, shuffle=False, log=False)
    batch, core, n_id, offset, count = next(iter(loader))
    x, adj, target = batch.x.to(device), batch.adj_t.to(device), batch.y[:core].to(device)
    mask = batch.train_mask[:core]
    coordinates = (n_id, offset, count)
    if ppi:
        ppi_radial.RadialFeedback = adapt(ppi_radial.RadialFeedback)
        head_names = {'lins.1.weight', 'lins.1.bias'}
        layout = [('bns.1.weight', 0, 0, 1), ('bns.1.bias', 0, 0, 1)]
    elif pna:
        head_names = {f'convs.3.post_lins.{i}.{field}' for i in range(4) for field in ('weight', 'bias')}
        head_names |= {'convs.3.lin.weight', 'convs.3.lin.bias'}
        layout = [(f'convs.3.pre_lins.{i}.{field}', i, 0, 1) for i in range(4) for field in ('weight', 'bias')]
    else:
        head_names = {'convs.1.lin.weight', 'convs.1.bias'}
        layout = ([('bns.0.weight', 0, 0, 1), ('bns.0.bias', 0, 0, 1)] if dataset == 'flickr' else
                  [('convs.0.lin.weight', 0, 0, 1), ('convs.0.bias', 0, 0, 1)])

    def build():
        if pna:
            model = PNA(n, inputs, 64, classes, 4, ['mean', 'max'], ['identity', 'amplification'],
                        data.adj_t.storage.rowcount(), dropout=.5, drop_input=True, batch_norm=True,
                        residual=False, pool_size=2, buffer_size=n, gamma=.5).to(device)
        else:
            model = GCN(n, inputs, recipe['hidden'], classes, 2, dropout=recipe['dropout'],
                        batch_norm=True if ppi else recipe['batch_norm'], residual=ppi or recipe['residual'],
                        drop_input=False if ppi else recipe['drop_input'], linear=ppi,
                        pool_size=2, buffer_size=n).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.reg_modules.parameters(), weight_decay=recipe.get('reg_weight_decay', 0.)),
            dict(params=model.nonreg_modules.parameters(), weight_decay=0.),
        ], lr=.005 if pna else recipe['lr'])
        if ppi:
            install(model, optimizer, beta1=.7)
            model._r59_radial = ppi_radial.RadialFeedback(model, 'euclidean')
            engine = SimpleNamespace(model=model)
        elif pna:
            engine = PNATraining(model, optimizer, 'A')
            for group in optimizer.param_groups:
                group['lr'] = recipe['lr']
            optimizer.defaults['lr'] = recipe['lr']
        elif dataset == 'flickr':
            engine = FlickrTraining(model, optimizer)
        elif dataset == 'yelp':
            engine = YelpTraining(model, optimizer)
        else:
            engine = RedditTraining(model, optimizer)
            engine.install_static_input(data.x, data.adj_t)
            engine.next_prefix_before_refresh = True
        if dataset == 'reddit2' or ppi:
            engine.dependency = Installation(model.pool)
        for bn in model.bns:
            bn.momentum = recipe.get('bn_momentum', .1)
        return engine, optimizer

    def prefix(engine):
        if pna:
            return engine.prefix(x, adj, core, n_id)
        if ppi:
            readout = engine.model.lins[1]
        else:
            readout = engine.readout
        if dataset == 'reddit2':
            engine.supervised_rows = mask.nonzero(as_tuple=True)[0].to(device)
            return engine.prefix(x, adj, core, *coordinates)
        old = readout.forward
        readout.forward = (lambda hidden: hidden) if ppi else (lambda hidden, matrix: matmul(matrix, hidden))
        try:
            return engine.model(x, adj, core, *coordinates)
        finally:
            readout.forward = old

    def objective(engine, ports):
        if pna:
            boundary, scale = ports
            logits = engine.tail(boundary, scale)
        elif ppi:
            logits = engine.model.lins[1](ports)
        else:
            logits = engine.tail(ports)
        if dataset == 'reddit2':
            return F.cross_entropy(logits, target[mask.to(device)])
        if dataset == 'yelp':
            return F.binary_cross_entropy_with_logits(logits[mask.to(device)], target[mask.to(device)],
                                                       pos_weight=logits.new_tensor(2.))
        if ppi:
            return F.binary_cross_entropy_with_logits(logits, target)
        return F.cross_entropy(logits[mask.to(device)], target[mask.to(device)])

    for zero in (False, True):
        engine, optimizer = build()
        affine_case(engine.model, [row[0] for row in layout], zero)
        for history in engine.model.histories:
            history.emb.copy_(torch.randn(history.emb.shape, device=device).cpu() * .1)
        for _ in range(1 if zero else 2):
            def invoke(current):
                if ppi:
                    ppi_training.make_train(None, 'prefix', 2)(current.model, [(batch, core, *coordinates)],
                                                              torch.nn.BCEWithLogitsLoss(), optimizer, 1)
                elif pna:
                    current.block(x, adj, core, n_id, target, mask.to(device), torch.nn.CrossEntropyLoss(),
                                  label_count=int(mask.sum()))
                else:
                    criterion = (torch.nn.BCEWithLogitsLoss(pos_weight=x.new_tensor(2.)) if dataset == 'yelp'
                                 else torch.nn.CrossEntropyLoss())
                    current.block(x, adj, core, coordinates, target,
                                  mask if dataset == 'reddit2' else mask.to(device), criterion,
                                  label_count=int(mask.sum()))

            measured = (lambda: tuple(value.clone() for value in engine.model._r59_radial.queries)) if ppi else None
            reference = check_block(engine, optimizer, build, prefix, objective, invoke, head_names, layout,
                                    measure=measured, atol=8e-5, rtol=3e-3)
            if hasattr(reference, 'dependency'):
                reference.dependency.restore()
            if ppi:
                reference.model._r59_radial._recovery_graph.close()
        if hasattr(engine, 'dependency'):
            engine.dependency.restore()
        if ppi:
            engine.model._r59_radial._recovery_graph.close()
