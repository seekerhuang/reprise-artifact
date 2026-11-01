import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from main.initialization import SEEDS, construct, iglu_tensors
from main.models import DoubleBNGCN, GCN, TOPGCN
from main.recipes import RECIPES
from main.sage.sage_model import HistorySAGE


DIMENSIONS = {'arxiv': (128, 40), 'products': (100, 47), 'reddit2': (602, 41),
              'flickr': (500, 7), 'yelp': (300, 100), 'ppi': (50, 121)}


def factory(recipe):
    inputs, outputs = DIMENSIONS[recipe['dataset']]
    family = recipe['family']
    if family == 'pna':
        from empirical.pna.model import PNA
        from empirical.pna.train import architecture
        return lambda: PNA(0, inputs, out_channels=outputs, deg=torch.ones(1, dtype=torch.long),
                           gamma=recipe['gamma'], **architecture(recipe))
    if family == 'top':
        from main.native.dgl_adapter import configure
        configure()
        return lambda: TOPGCN(inputs, 256, outputs, recipe['dropout'])
    if family == 'sage':
        return lambda: HistorySAGE(inputs, 256, outputs, 3, recipe['dropout'], 'CSR', {}, SimpleNamespace(method='grad'))
    if family == 'lmc':
        return lambda: DoubleBNGCN(inputs, outputs, recipe['layers'], recipe['dropout'], recipe['dataset'] == 'flickr')
    if family == 'saint':
        from empirical.saint.model import GraphSAINT
        architecture = dict(dim=recipe['hidden'], aggr='concat', loss=recipe['loss'],
                            arch=recipe['architecture'], act='relu', bias='norm')
        parameters = dict(lr=0.01, dropout=0.1, weight_decay=0.0)
        return lambda: GraphSAINT(outputs, architecture, parameters, np.zeros((1, inputs)), np.zeros((1, outputs)), cpu_eval=True)
    return lambda: GCN(inputs, recipe.get('hidden', 256), outputs,
                        3 if family == 'products_gcn' else 2, recipe['dropout'],
                        residual=recipe.get('residual', family == 'ppi'),
                        batch_norm=recipe.get('batch_norm', True),
                        drop_input=recipe.get('drop_input', False), linear=family == 'ppi')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', choices=RECIPES, default=list(RECIPES))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    for name in args.configs:
        recipe = RECIPES[name]
        target = args.data_dir / 'initial' / name
        target.mkdir(parents=True, exist_ok=True)
        for slot in (1, 2, 3):
            path = target / (str(slot) + '.pt')
            if path.exists():
                continue
            settings = SimpleNamespace(config=name, initialization=slot)
            if recipe['family'] == 'iglu':
                from main.iglu.iglu_model import Target
                model = Target(iglu_tensors(SEEDS[slot - 1] or 2026092901))
                torch.manual_seed(2026092901)
                rng = torch.get_rng_state(), torch.cuda.get_rng_state(args.device)
            else:
                model, rng = construct(factory(recipe), settings, recipe, device=args.device)
            torch.save(dict(model={key: value.detach().cpu() for key, value in model.state_dict().items()}, rng=rng), path)
            del model
        print(name, flush=True)


if __name__ == '__main__':
    main()
