import argparse
import importlib
import os
from pathlib import Path

from baselines.recipes import RECIPES
from run import configure_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=sorted({key[0] for key in RECIPES}))
    parser.add_argument('--config', required=True, choices=sorted({key[1] for key in RECIPES}))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--initialization', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cpu-cores', type=int, nargs='+')
    args = parser.parse_args()
    key = args.method, args.config
    if key not in RECIPES:
        parser.error('Unsupported method/configuration pair')
    configure_cpu(args, parser)
    os.environ['DGLBACKEND'] = 'pytorch'
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = RECIPES[key]
    from main.common import seed
    seed(recipe['seed'])
    if args.method == 'pfnc':
        from main.native.dgl_adapter import configure
        configure()
    importlib.import_module('baselines.' + args.method + '.train').run(args, recipe)


if __name__ == '__main__':
    main()
