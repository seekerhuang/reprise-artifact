from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch


def check(name, device, method):
    from empirical.saint import data, train
    from empirical.saint.delivery.averaging import CausalAverage
    from main.recipes import RECIPES
    from prepare_initial import factory
    from recording import check_output
    reddit = name == 'Reddit2-SAINT'
    recipe = dict(RECIPES[name], epochs=2, updates=2 if method == 'direct' else 4)
    cadence = 1 if method == 'direct' else 2
    recipe['eval_every'] = cadence
    if reddit:
        recipe['sample_coverage'] = 800
    full = sp.eye(80, format='csr', dtype=np.int32)
    training = sp.diags(np.r_[np.ones(60), np.zeros(20)]).astype(np.int32).tocsr()
    roles = dict(tr=np.arange(60), va=np.arange(60, 70), te=np.arange(70, 80))
    rng = np.random.default_rng(7)
    features = rng.standard_normal((80, 602 if reddit else 300))
    labels = np.eye(41)[np.arange(80) % 41] if reddit else rng.integers(0, 2, (80, 100)).astype(np.float64)

    def average(model, **kwargs):
        return CausalAverage(model, after_updates=cadence, cadence=cadence, window=2)

    with tempfile.TemporaryDirectory(prefix='reprise-saint-training-') as temporary:
        directory = Path(temporary)
        target = directory / 'initial' / name
        target.mkdir(parents=True)
        initial = factory(recipe)()
        torch.save(dict(model=initial.state_dict(), rng=(torch.get_rng_state(), torch.cuda.get_rng_state(device))),
                   target / '1.pt')
        args = SimpleNamespace(config=name, method=method, data_dir=directory, out=directory / 'output',
                               initialization=1, device=device)
        with patch.object(data, 'load_reddit' if reddit else 'load_yelp',
                          return_value=(full, training, features, labels, roles)), \
             patch('empirical.saint.delivery.averaging.CausalAverage', side_effect=average):
            train.run(args, recipe)
        check_output(args.out, method)
