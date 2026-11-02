import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(ROOT))
from main.recipes import DIRECT_CONFIGS, RECIPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', choices=RECIPES)
    parser.add_argument('--method', choices=('reprise', 'direct'), default='reprise')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.configs is None:
        args.configs = list(DIRECT_CONFIGS if args.method == 'direct' else RECIPES)
    if args.method == 'direct' and any(name not in DIRECT_CONFIGS for name in args.configs):
        parser.error('--method direct supports ' + ', '.join(DIRECT_CONFIGS))
    if len(args.configs) > 1:
        for name in args.configs:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--configs', name, '--device', args.device, '--method', args.method], check=True)
        return
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from main.common import seed
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(123)
    name = args.configs[0]
    family = RECIPES[name]['family']
    if family == 'saint':
        from saint import check
    elif family == 'iglu':
        from iglu import check
    elif family == 'pna' and args.method != 'direct':
        from pna import check
    else:
        from training import check
    if args.method == 'direct':
        check(name, args.device, method='direct')
    else:
        check(name, args.device)
    torch.cuda.synchronize(args.device)
    print('PASS', name, flush=True)


if __name__ == '__main__':
    main()
