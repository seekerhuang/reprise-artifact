import torch
from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader, permute

from empirical.data.datasets import load
from empirical.gas.train import _data
from empirical.pna.averaging import Window
from empirical.pna.delivery import Delivery, compile_native
from empirical.pna.model import PNA
from empirical.pna.training import Training
from main.common import Output, Timer, score
from main.initialization import load_initial, restore
from main.recipes import set_rate


def architecture(recipe):
    return dict(hidden_channels=recipe['hidden'], num_layers=recipe['layers'],
                aggregators=['mean', 'max'], scalers=['identity', 'amplification'],
                dropout=recipe['dropout'], drop_input=True, batch_norm=True, residual=False)


@torch.no_grad()
def refresh(model, loader):
    model.eval()
    return model(loader=loader)


def run(args, recipe):
    direct = getattr(args, 'method', 'reprise') == 'direct'
    device = torch.device(args.device)
    timer = Timer(device)
    compile_native(PNA)
    with timer:
        raw = load('flickr', args.data_dir)
        data = _data(raw)
        saved = torch.load(args.data_dir / 'flickr-partition.pt', weights_only=True)
        data = permute(data, saved['permutation'], log=False)
        train = SubgraphLoader(data, saved['ptr'], batch_size=recipe['batch_parts'],
                               shuffle=True, num_workers=0)
        evaluation = EvalSubgraphLoader(data, saved['ptr'], batch_size=recipe['batch_parts'], log=False)
        buffer_size = max(sub.n_id.numel() for sub in evaluation)
        options = architecture(recipe)

        def factory():
            return PNA(data.num_nodes, data.x.shape[1], out_channels=7,
                       deg=data.adj_t.storage.rowcount(), pool_size=2,
                       buffer_size=buffer_size, gamma=recipe['gamma'], **options).to(device)

        model = factory()
        optimizer = torch.optim.Adam([
            dict(params=model.reg_modules.parameters(), weight_decay=0.0),
            dict(params=model.nonreg_modules.parameters(), weight_decay=0.0),
        ], lr=0.005)
        if direct:
            from empirical.pna.direct import Direct
            engine = Direct(model, optimizer)
        else:
            engine = Training(model, optimizer, 'A')
        rng = load_initial(model, args)
        for group in optimizer.param_groups:
            group['lr'] = recipe['lr']
        optimizer.defaults['lr'] = recipe['lr']
        for bn in model.bns:
            bn.momentum = recipe['bn_momentum']
        average = Window(model)
        delivery = Delivery(factory, data, evaluation, device)
        restore(rng, device)
        refresh(model, evaluation)
    output = Output(args.out)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        for _ in range(recipe['epochs']):
            with timer:
                set_rate(optimizer, recipe, engine.visits)
                for _ in range(2 if direct else 1):
                    for index, (batch, core, ids, _offset, _count) in enumerate(train):
                        host_mask = batch.train_mask[:core]
                        count = int(host_mask.sum())
                        if not count:
                            continue
                        engine.block(batch.x.to(device), batch.adj_t.to(device), core, ids,
                                     batch.y[:core].to(device), host_mask.to(device), criterion,
                                     label_count=count)
                        if index + 1 >= recipe['max_steps'] and index + 1 < len(train):
                            break
                if direct:
                    optimizer.zero_grad(set_to_none=True)
                average.capture(model, engine.visits)
                current_refresh = Timer(device)
                with current_refresh:
                    refresh(model, evaluation)
            logits, preparation_seconds = delivery.observe(model, average)
            output.observe(delivery.model, engine.visits,
                           timer.seconds - current_refresh.seconds + preparation_seconds,
                           score(logits[data.val_mask], data.y[data.val_mask]),
                           score(logits[data.test_mask], data.y[data.test_mask]))
    finally:
        output.close()
