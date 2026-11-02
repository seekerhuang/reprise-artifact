import argparse
from pathlib import Path

import torch

from empirical.data.datasets import load, partition


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


if __name__ == '__main__':
    main()
