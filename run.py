import argparse
import os
from pathlib import Path

from main.recipes import DIRECT_CONFIGS, RECIPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=RECIPES)
    parser.add_argument('--method', choices=('reprise', 'direct'), default='reprise')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--initialization', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.method == 'direct' and args.config not in DIRECT_CONFIGS:
        parser.error('--method direct supports ' + ', '.join(DIRECT_CONFIGS))
    if RECIPES[args.config]['family'] == 'saint':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from main.common import seed
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(123)
    recipe = RECIPES[args.config]
    family = recipe['family']
    if family in ('top', 'products_gcn', 'sage', 'iglu'):
        from main.train import run
    elif family == 'lmc':
        from empirical.lmc.train import run
    elif family in ('history', 'ppi'):
        from empirical.gas.train import run
    elif family == 'pna':
        from empirical.pna.train import run
    else:
        from empirical.saint.train import run
    run(args, recipe)


if __name__ == '__main__':
    main()
