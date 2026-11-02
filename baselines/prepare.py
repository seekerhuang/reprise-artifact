import argparse
from pathlib import Path

import torch

from empirical.data.datasets import load, partition
from main.recipes import RECIPES
from prepare_initial import factory


CONFIGS = ('products-GCN', 'products-SAGE', 'Reddit2-GCN')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.data_dir / 'baselines'
    root.mkdir(parents=True, exist_ok=True)
    for dataset, parts in [('products', 70), ('reddit2', 200)]:
        path = root / (dataset + '-partition.pt')
        if not path.exists():
            torch.save(partition(load(dataset, args.data_dir), parts), path)
    for name in CONFIGS:
        target = root / 'initial' / name
        target.mkdir(parents=True, exist_ok=True)
        for slot in (1, 2, 3):
            path = target / (str(slot) + '.pt')
            if path.exists():
                continue
            source = args.data_dir / 'initial' / name / (str(slot) + '.pt')
            state = torch.load(source, map_location='cpu', weights_only=False)
            if slot == 1 and name == 'products-GCN':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(123)
                    for _ in range(4):
                        torch.empty((), dtype=torch.int64).random_()
                    model = factory(RECIPES[name])()
                    state = dict(state, model=model.state_dict())
            if name != 'products-SAGE':
                state = dict(model=state['model'])
            torch.save(state, path)


if __name__ == '__main__':
    main()
