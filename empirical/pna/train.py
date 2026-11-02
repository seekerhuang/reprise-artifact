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
from main.recording import Recording
from main.verification import expected_clocks, expected_bn


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
    recording = Recording(device, memory=direct)
    timer = Timer(device, recording, 'setup_seconds')
    compile_native(PNA)
    with timer:
        with recording.phase('data_and_partition_seconds', group='setup'):
            raw = load('flickr', args.data_dir)
            data = _data(raw)
            saved = torch.load(args.data_dir / 'flickr-partition.pt', weights_only=True)
            data = permute(data, saved['permutation'], log=False)
        with recording.phase('loader_construction_seconds', group='setup'):
            train = SubgraphLoader(data, saved['ptr'], batch_size=recipe['batch_parts'],
                                   shuffle=True, num_workers=0)
            evaluation = EvalSubgraphLoader(data, saved['ptr'], batch_size=recipe['batch_parts'], log=False)
            buffer_size = max(sub.n_id.numel() for sub in evaluation)
        options = architecture(recipe)

        def factory():
            return PNA(data.num_nodes, data.x.shape[1], out_channels=7,
                       deg=data.adj_t.storage.rowcount(), pool_size=2,
                       buffer_size=buffer_size, gamma=recipe['gamma'], **options).to(device)

        with recording.phase('model_constructor_seconds', group='setup'):
            model = factory()
        with recording.phase('optimizer_constructor_seconds', group='setup'):
            optimizer = torch.optim.Adam([
                dict(params=model.reg_modules.parameters(), weight_decay=0.0),
                dict(params=model.nonreg_modules.parameters(), weight_decay=0.0),
            ], lr=0.005)
        with recording.phase('engine_initial_and_delivery_setup_seconds', group='setup'):
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
            delivery = Delivery(factory, data, evaluation, device, recording)
            from empirical.gas.clock import Clock
            clock = Clock(model)
            recording.close_callbacks.append(clock.close)
        with recording.phase('initial_rng_restore_seconds', group='setup'):
            restore(rng, device)
        with recording.phase('initial_history_fill_seconds', group='setup'):
            refresh(model, evaluation)
    timer.name = 'training_outer_seconds'
    recording.identify(args, recipe, model, optimizer)
    output = Output(args.out, recording)

    def evaluate_metrics(candidate):
        logits = refresh(candidate, evaluation)
        return (score(logits[data.val_mask], data.y[data.val_mask]),
                score(logits[data.test_mask], data.y[data.test_mask]))

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
                current_refresh = Timer(device, recording, 'history_refresh_seconds')
                with current_refresh:
                    refresh(model, evaluation)
            with recording.phase('delivery_observer_seconds'):
                logits, preparation_seconds = delivery.observe(model, average)
                valid = score(logits[data.val_mask], data.y[data.val_mask])
                test = score(logits[data.test_mask], data.y[data.test_mask])
            recording.costs['pure_history_output_seconds'] = clock.output_seconds
            recording.costs['shared_pool_setup_seconds'] = clock.shared_setup_seconds
            recording.costs['current_evaluation_refresh_excluded_seconds'] = current_refresh.seconds
            recording.costs['prior_history_outputs_excluded_seconds'] = clock.prior_output(True)
            recording.costs['single_delivery_preparation_seconds'] = preparation_seconds
            output.observe(delivery.model, engine.visits,
                           timer.seconds - current_refresh.seconds + preparation_seconds - clock.prior_output(True), valid, test)
        expected = 2 * recipe['max_steps'] * recipe['epochs']
        if not direct:
            engine.verify(expected)
        output.finish(model, engine.visits, evaluate_metrics, expected_updates=expected, optimizer=optimizer,
                      clocks=expected_clocks(model, expected),
                      bn_clocks=expected_bn(model, expected // (1 if direct else 2)),
                      tolerances=(1 / int(data.val_mask.sum()), 1 / int(data.test_mask.sum())),
                      counts=(int(data.val_mask.sum()), int(data.test_mask.sum())))
    finally:
        output.close()
