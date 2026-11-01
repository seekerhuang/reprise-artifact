from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch


def check(name, device):
    from main.iglu.iglu_model import Target, TF_NAMES, SHAPES, csr
    from main.iglu.iglu_structural_r66_v1 import IGLUStructural
    rng = np.random.RandomState(7)
    arrays = {key: rng.normal(0, .03, shape).astype(np.float32) for key, shape in zip(TF_NAMES, SHAPES)}
    for key in ('scale', 'Layer2/scale'):
        arrays[key].fill(1.)
    model = Target(arrays, device=device)
    matrix = sp.eye(80, format='csr', dtype=np.float32)
    propagation, transpose = csr(matrix, device), csr(matrix.T, device)
    local, local_transpose = csr(matrix[:32], device), csr(matrix[:32].T, device)
    features = torch.randn(80, 128, device=device)
    geometry = SimpleNamespace(first=torch.sparse.mm(propagation, features), matrix=propagation,
                               transpose=transpose, conditions=[SimpleNamespace(
                                   matrix=local, transpose=local_transpose,
                                   labels=torch.arange(32, device=device) % 40)])
    engine = IGLUStructural(model, geometry, dropout=.2)
    before = model.w.detach().clone()
    pairs = []

    def observe(index, current, measurements, directions, means, corrections):
        gradients = tuple(p.grad.detach().clone() for p in (model.scale2, model.offset2))
        if index == 0:
            pairs[:] = gradients
        else:
            for first, second, mean in zip(pairs, gradients, means):
                torch.testing.assert_close((first.double() + second.double()) * .5,
                                           mean.double(), rtol=1e-5, atol=1e-7)

    engine.observe = observe
    for _ in range(2):
        engine.update_block(0, .001)
    assert engine.actual_updates == 4 and not torch.equal(before, model.w)
    model.eval()
    with torch.no_grad():
        logits = model(geometry.first, propagation, transpose)
    assert logits.shape == (80, 40) and torch.isfinite(logits).all()
