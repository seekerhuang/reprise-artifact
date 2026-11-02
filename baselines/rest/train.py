import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric_autoscale import permute

from baselines.common import Output, metrics
from baselines.data import initial, read
from empirical.data.datasets import adjacency


def run(args, recipe):
    from baselines.rest.engine import Engine
    output = Output(args, recipe)
    with output.phase('input_and_partition'):
        raw = read(args, recipe['dataset'])
        data = Data(x=raw['x'], y=raw['y'], adj_t=adjacency(raw),
                    train_mask=raw['train_mask'], val_mask=raw['val_mask'], test_mask=raw['test_mask'])
        saved = torch.load(args.data_dir / 'baselines' / f"{recipe['dataset']}-partition.pt",
                           weights_only=True)
        data = permute(data, saved['permutation'], log=False)
        data.adj_t = gcn_norm(data.adj_t.set_diag(), add_self_loops=False)
    with output.phase('model_and_initial_history'):
        engine = Engine(data, saved['ptr'], initial(args)['model'], recipe, args.device)
    record = dict(y=data.y, val_mask=data.val_mask, test_mask=data.test_mask)
    for epoch in range(recipe['epochs']):
        with output.phase('training_and_history'):
            engine.epoch_step()
        with output.phase('evaluation'):
            valid, test = metrics(engine.last_logits, record)
        output.observe(engine.model, epoch + 1, engine.counters['optimizer_updates'], valid, test)
    expected = recipe['epochs'] * (recipe['parts'] // recipe['batch_parts'])
    if engine.counters['optimizer_updates'] != expected or engine.rest_counters['forward_only_calls'] != expected:
        raise RuntimeError('REST update or refresh count differs')

    @torch.no_grad()
    def evaluate(candidate):
        candidate.eval()
        return metrics(candidate(loader=engine.eval_loader), record)

    output.finish(engine.model, evaluate, engine.optimizer)
    engine.close()
