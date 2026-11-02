import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from main.recipes import RECIPES
from prepare_initial import DIMENSIONS, factory


def prepare(name, recipe, directory):
    dataset = recipe['dataset']
    inputs, classes = DIMENSIONS[dataset]
    counts = {'arxiv': 240, 'products': 640, 'reddit2': 400,
              'flickr': 240, 'yelp': 240, 'ppi': 80}
    parts = {'arxiv': 80, 'reddit2': 200, 'flickr': 24, 'yelp': 40, 'ppi': 20}
    n = counts[dataset]
    ids = torch.arange(n)
    columns = torch.stack([ids.roll(1), ids, ids.roll(-1)], dim=1).flatten()
    train_end, valid_end = (615, 627) if dataset == 'products' else (n * 2 // 3, n * 5 // 6)
    multilabel = dataset in ('yelp', 'ppi')
    labels = torch.randint(0, 2, (n, classes)).float() if multilabel else ids % classes
    raw = dict(x=torch.randn(n, inputs), y=labels, rowptr=torch.arange(0, 3 * n + 1, 3),
               col=columns, train_mask=ids < train_end,
               val_mask=(ids >= train_end) & (ids < valid_end), test_mask=ids >= valid_end)
    raw.update(sample_rowptr=raw['rowptr'], sample_col=raw['col'])
    if dataset == 'ppi':
        raw['train_mask'] = torch.ones(n, dtype=torch.bool)
    torch.save(dict(train=raw, val=raw, test=raw) if dataset == 'ppi' else raw,
               directory / (dataset + '.pt'))
    if dataset in parts:
        ptr = torch.tensor([i * n // parts[dataset] for i in range(parts[dataset] + 1)])
        partition = dict(permutation=ids, ptr=ptr)
        torch.save(partition, directory / (dataset + '-partition.pt'))
        if recipe['family'] == 'lmc':
            torch.save(partition, directory / (dataset + '-lmc-partition.pt'))
            torch.save(torch.tensor([0, int(ptr[recipe['batch_parts']])]),
                       directory / (dataset + '-lmc-sequence.pt'))
    model = factory(recipe)()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    target = directory / 'initial' / name
    target.mkdir(parents=True)
    torch.save(dict(model=initial, rng=(torch.get_rng_state(), torch.cuda.get_rng_state())), target / '1.pt')
    return initial, set(dict(model.named_parameters()))


def check(name, device, method='reprise'):
    recipe = dict(RECIPES[name])
    recipe.update(updates=4, eval_every=2, epochs=5 if name == 'Yelp-GCN' else 2 if name == 'PPI-GCN' else 1)
    if recipe['family'] == 'sage':
        recipe['batch_size'] = 615 if name == 'products-SAGE' else 32
    with tempfile.TemporaryDirectory(prefix='reprise-tensors-') as temporary:
        directory = Path(temporary)
        initial, parameters = prepare(name, recipe, directory)
        args = SimpleNamespace(config=name, data_dir=directory, out=directory / 'output',
                               initialization=1, device=device, threads=2, method=method)
        if recipe['family'] in ('top', 'products_gcn', 'sage'):
            from main.train import run
        elif recipe['family'] == 'lmc':
            from empirical.lmc.train import run
        elif recipe['family'] == 'pna':
            from empirical.pna.train import run
        else:
            from empirical.gas.train import run
        run(args, recipe)
        from recording import check_output
        check_output(args.out, method)
        checkpoint = torch.load(args.out / 'best.pt', map_location='cpu', weights_only=False)
        if method == 'direct':
            expected = 2 * recipe['max_steps'] * recipe['epochs'] if recipe['family'] in ('ppi', 'pna') else recipe['updates']
            with (args.out / 'trajectory.csv').open() as stream:
                assert int(list(csv.DictReader(stream))[-1]['updates']) == expected
        actual = checkpoint['model']
        if method == 'direct':
            for key, value in actual.items():
                if key.endswith('num_batches_tracked'):
                    expected = 0 if name == 'arxiv-TOP' and key == 'bns.2.num_batches_tracked' else checkpoint['updates']
                    assert int(value) == expected, (key, int(value), expected)
        assert actual.keys() == initial.keys()
        changed = False
        for key, value in actual.items():
            assert value.shape == initial[key].shape and value.dtype == initial[key].dtype, key
            if value.is_floating_point():
                assert torch.isfinite(value).all(), key
                changed |= key in parameters and not torch.equal(value, initial[key])
        assert changed
