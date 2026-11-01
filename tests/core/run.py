import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main.recipes import RECIPES

ADAPTERS = {
    'arxiv-TOP': 'main.top.arxiv_coupled_structural_r64_v1.CoupledStructural',
    'products-GCN': 'main.products_gcn.products_equal_body_small_head_graph_v1.EqualBodySmallSecondStructural',
    'arxiv-Fresh': 'main.sage.fresh_arxiv_structural.FreshArxivStructural',
    'products-SAGE': 'main.sage.sage_exact_head_graph_v1.ExactHeadSAGE',
    'arxiv-IGLU': 'main.iglu.iglu_structural_r66_v1.IGLUStructural',
    'arxiv-LMC': 'empirical.lmc.arxiv.lmc_arxiv_structural.LMCStructural',
    'Flickr-LMC': 'empirical.lmc.flickr.lmc_flickr_structural.FrozenMeanP2',
    'Reddit2-GCN': 'empirical.gas.reddit.reddit_structural.Training',
    'Flickr-GCN': 'empirical.gas.flickr.flickr_structural.FlickrTraining',
    'Yelp-GCN': 'empirical.gas.yelp.yelp_structural.Training',
    'PPI-GCN': 'empirical.gas.ppi.ppi_training.make_train',
    'Flickr-PNA': 'empirical.pna.training.Training',
    'Reddit2-SAINT': 'empirical.saint.train.reddit_engine_class',
    'Yelp-SAINT': 'empirical.saint.train.engine_class',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', choices=ADAPTERS, default=list(RECIPES))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    assert set(ADAPTERS) == set(RECIPES)
    if len(args.configs) > 1:
        for name in args.configs:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--configs', name,
                            '--device', args.device], check=True)
        return
    import torch
    from main.common import seed
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(731)
    name = args.configs[0]
    family = RECIPES[name]['family']
    if family in ('iglu', 'lmc'):
        from iglu_lmc import check
    elif family in ('saint',):
        from saint import check
    elif family in ('history', 'ppi', 'pna'):
        from history_adapters import check
    else:
        from graph_adapters import check
    check(name, args.device)
    torch.cuda.synchronize(args.device)
    print('PASS', name, ADAPTERS[name], 'mean-VJP / measurements / decoding', flush=True)


if __name__ == '__main__':
    main()
