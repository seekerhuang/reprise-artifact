import copy
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from reference import close, local_measurements, projection


def _close(actual, expected, device, *, fp32=False, state=False):
    if isinstance(actual, torch.Tensor):
        actual, expected = actual.to(device), expected.to(device)
        assert actual.device.type == expected.device.type == 'cuda'
        tolerance = (3e-4, 3e-5) if fp32 else ((2e-7, 2e-8) if state else (1e-9, 2e-11))
        close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _close(actual[key], expected[key], device, fp32=fp32, state=state)
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _close(first, second, device, fp32=fp32, state=state)
    else:
        assert actual == expected


def _matrix(nodes, device, dtype):
    identity = torch.eye(nodes, device=device, dtype=dtype)
    return (identity * .4 + identity.roll(1, 1) * .2 + identity.roll(-1, 1) * .2
            + identity.roll(5, 1) * .1 + identity.roll(-5, 1) * .1)


def _block(actual, reference, optimizers, reference_optimizers, selected, head_names,
           unused, condition, prefix, head, loss, execute, device, engine, *, fp32, zero):
    reference.load_state_dict(actual.state_dict(), strict=True)
    for name, optimizer in optimizers.items():
        reference_optimizers[name].load_state_dict(copy.deepcopy(optimizer.state_dict()))
    actual.train()
    reference.train()
    named = dict(actual.named_parameters())
    ref_named = dict(reference.named_parameters())
    body = tuple(key for key in named if key not in head_names + unused)
    head_optimizer = 'w' if fp32 else 'Adam'
    body_optimizers = ('Q', 'R') if fp32 else ('Adam',)
    incoming_rng = torch.cuda.get_rng_state(device)
    boundary = prefix(reference, condition)
    raw, head_gradients = [], []
    for _ in range(2):
        reference.zero_grad(set_to_none=True)
        objective = loss(head(reference, boundary), condition)
        names = body + head_names
        values = torch.autograd.grad(objective, tuple(ref_named[key] for key in names),
                                     retain_graph=True)
        gradients = {key: value.detach().clone() for key, value in zip(names, values)}
        raw.append({key: gradients[key] for key in body})
        head_gradients.append({key: gradients[key] for key in head_names})
        for key in head_names:
            ref_named[key].grad = gradients[key]
        reference_optimizers[head_optimizer].step()
    reference.zero_grad(set_to_none=True)
    means = {key: (raw[0][key] + raw[1][key]) * .5 for key in body}
    parameters = {key: ref_named[key] for key in selected}
    layout = [(key, 0, parameter.ndim - 1, 1) for key, parameter in parameters.items()]
    correction, measurements = projection(parameters, layout, raw[0], raw[1])
    assert torch.stack([value.double().square().sum() for value in means.values()]).sum() > 0
    assert torch.stack([value.double().square().sum() for value in correction.values()]).sum() > 0
    if zero:
        assert all(torch.all(parameters[key].reshape(-1)[0] == 0) for key in selected)
        assert all(torch.all(correction[key].reshape(-1)[0] == 0) for key in selected)
    submissions = []
    for sign in (-1, 1):
        reference.zero_grad(set_to_none=True)
        gradients = {key: value + sign * correction[key] if key in correction else value
                     for key, value in means.items()}
        submissions.append(gradients)
        for key, value in gradients.items():
            ref_named[key].grad = value
        for name in body_optimizers:
            reference_optimizers[name].step()
    expected_rng = torch.cuda.get_rng_state(device)
    events, reported_measurements, observed_mean = [], [], {}
    counts = dict.fromkeys(body, 0)
    handles = []

    def capture_mean(gradient, key):
        assert gradient.device.type == 'cuda'
        observed_mean[key] = gradient.detach().clone()
        counts[key] += 1

    for key in body:
        handles.append(named[key].register_hook(lambda gradient, key=key: capture_mean(gradient, key)))

    def capture_step(name):
        def capture(optimizer, args, kwargs):
            gradients = {key: parameter.grad.detach().clone() if parameter.grad is not None else None
                         for key, parameter in named.items()}
            measurement = None
            if all(gradients[key] is not None for key in head_names):
                weight = actual.w if fp32 else actual.convs[-1].lin.weight
                measurement = (weight.detach().double() * weight.grad.detach().double()).sum(
                    1 if fp32 else 0).reshape(-1)
            elif not reported_measurements:
                values = local_measurements(engine)
                assert values is not None and len(values) == 2
                reported_measurements.extend(values)
            events.append((name, gradients, measurement))
        return capture

    for name, optimizer in optimizers.items():
        handles.append(optimizer.register_step_pre_hook(capture_step(name)))
    torch.cuda.set_rng_state(incoming_rng, device)
    try:
        execute(condition)
    finally:
        for handle in handles:
            handle.remove()
    _close(torch.cuda.get_rng_state(device), expected_rng, device)
    assert set(counts.values()) == {1}
    _close(observed_mean, means, device, fp32=fp32)
    expected_apis = ('w', 'w', 'Q', 'R', 'Q', 'R') if fp32 else ('Adam',) * 4
    assert tuple(name for name, _, _ in events) == expected_apis
    for index in range(2):
        _, gradients, measured = events[index]
        assert all(gradients[key] is None for key in body + unused)
        _close({key: gradients[key] for key in head_names}, head_gradients[index], device, fp32=fp32)
        _close(measured, measurements[index], device, fp32=fp32)
        _close(reported_measurements[index], measurements[index], device, fp32=fp32)
    body_events = events[2:]
    for index, (_, gradients, measured) in enumerate(body_events):
        assert measured is None and all(gradients[key] is None for key in head_names + unused)
        _close({key: gradients[key] for key in body}, submissions[index // 2 if fp32 else index],
               device, fp32=fp32)
    first, second = body_events[0][1], body_events[2 if fp32 else 1][1]
    for key in selected:
        _close((first[key].double() + second[key].double()) * .5,
               means[key].double(), device, fp32=fp32)
        _close((second[key].double() - first[key].double()) * .5,
               correction[key].double(), device, fp32=fp32)
    _close(actual.state_dict(), reference.state_dict(), device, fp32=fp32, state=True)
    for name, optimizer in optimizers.items():
        _close(optimizer.state_dict(), reference_optimizers[name].state_dict(),
               device, fp32=fp32, state=True)


def _iglu(device):
    from main.iglu.iglu_model import Target, TF_NAMES, SHAPES
    from main.iglu.iglu_prefix import prefix
    from main.iglu.iglu_structural_r66_v1 import IGLUStructural
    from main.iglu.tf_adam import TF1Adam
    generator = torch.Generator(device=device).manual_seed(731)
    dense = _matrix(19, device, torch.float32)
    matrix, transpose = dense.to_sparse_csr(), dense.T.contiguous().to_sparse_csr()
    conditions = []
    for offset in (0, 1):
        rows = dense[offset::3]
        conditions.append(SimpleNamespace(matrix=rows.to_sparse_csr(),
            transpose=rows.T.contiguous().to_sparse_csr(),
            labels=torch.arange(len(rows), device=device) * 3))
    geometry = SimpleNamespace(first=dense @ torch.randn(19, 128, generator=generator, device=device,
                                                        dtype=torch.float32),
                               matrix=matrix, transpose=transpose, conditions=conditions)
    for zero in (False, True):
        tensors = {key: torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * .03
                   for key, shape in zip(TF_NAMES, SHAPES)}
        tensors['scale'].fill_(1.)
        tensors['Layer2/scale'].fill_(1.)
        actual = Target(tensors, device=device)
        if zero:
            with torch.no_grad():
                actual.scale2[0, 0] = actual.offset2[0, 0] = 0
        reference = copy.deepcopy(actual)
        engine = IGLUStructural(actual, geometry, dropout=.2)
        reference_optimizers = {key: TF1Adam(parameters, lr=.001)
                                for key, parameters in reference.groups().items()}
        for index, condition in enumerate(conditions):
            rate = .001 if index == 0 else .0005
            for optimizer in engine.optimizers.values():
                for group in optimizer.param_groups:
                    group['lr'] = rate
            _block(actual, reference, engine.optimizers, reference_optimizers,
                   ('scale2', 'offset2'), ('w', 'b3'), (), condition,
                   lambda model, current: prefix(model, geometry, current, .2),
                   lambda model, boundary: boundary @ model.w + model.b3,
                   lambda logits, current: F.cross_entropy(logits, current.labels, reduction='sum'),
                   lambda current: engine.update_block(index, rate), device, engine, fp32=True, zero=zero)
            assert set(engine.clocks().values()) == {2 * (index + 1)}
        assert engine.work['prefix_forwards'] == engine.work['prefix_VJPs'] == 2
        assert engine.work['head_queries'] == 4


def _lmc(name, device):
    from main.models import DoubleBNGCN
    from main.recipes import RECIPES
    recipe = RECIPES[name]
    flickr = name == 'Flickr-LMC'
    if flickr:
        from torch_sparse import SparseTensor
        from empirical.lmc.flickr.lmc_flickr_boundary import prefix, head, core, UNUSED_NAMES
        from empirical.lmc.flickr.lmc_flickr_structural import FrozenMeanP2, S_NAMES
        selected, unused = S_NAMES, UNUSED_NAMES
    else:
        from empirical.lmc.arxiv.lmc_arxiv_boundary import prefix, head, core
        from empirical.lmc.arxiv.lmc_arxiv_structural import LMCStructural
        selected = tuple('bns.1.' + branch + '.' + field
                         for branch in ('bn_ib', 'bn_ob') for field in ('weight', 'bias'))
        unused = tuple('bns.2.' + branch + '.' + field
                       for branch in ('bn_ib', 'bn_ob') for field in ('weight', 'bias'))
    generator = torch.Generator(device=device).manual_seed(732)
    dense = _matrix(24, device, torch.float64)
    features = torch.randn(24, 500 if flickr else 128, device=device,
                           dtype=torch.float64, generator=generator)
    rows = torch.arange(1, 24, 3, device=device)
    labels = rows % (7 if flickr else 40)
    condition = SimpleNamespace(core_size=12, train_rows=rows, labels=labels,
                                total_training_labels=16, core_training_labels=4)
    if flickr:
        row, col = dense.nonzero(as_tuple=True)
        condition.matrix = SparseTensor(row=row, col=col, value=dense[row, col],
                                         sparse_sizes=(24, 24), is_sorted=True)
        condition.features = features
    else:
        condition.matrix = dense.to_sparse_csr()
        condition.transpose = dense.T.contiguous().to_sparse_csr()
        condition.first_image = dense @ features
        labelled = dense.index_select(0, rows)
        condition.label_matrix = labelled.to_sparse_csr()
        condition.label_transpose = labelled.T.contiguous().to_sparse_csr()

    def optimizer(model):
        return torch.optim.Adam([
            dict(params=model.reg_modules.parameters(), weight_decay=0.),
            dict(params=model.nonreg_modules.parameters(), weight_decay=0.),
        ], lr=.01)

    for zero in (False, True):
        with torch.device(device):
            actual = DoubleBNGCN(features.shape[1], 7 if flickr else 40,
                                 recipe['layers'], recipe['dropout'], flickr).double()
        named = dict(actual.named_parameters())
        with torch.no_grad():
            for index, key in enumerate(selected):
                named[key].copy_(torch.linspace(.7, 1.3, 256, device=device, dtype=torch.float64)
                                 if key.endswith('weight') else
                                 torch.linspace(-.2, .3, 256, device=device, dtype=torch.float64)
                                 * (1 if index < 2 else -1))
                if zero:
                    named[key][0] = 0
        reference = copy.deepcopy(actual)
        actual_optimizer, reference_optimizer = optimizer(actual), optimizer(reference)
        engine = FrozenMeanP2(actual, actual_optimizer) if flickr else LMCStructural(actual, actual_optimizer)
        for group in actual_optimizer.param_groups:
            group['lr'] = recipe['lr']
        actual_optimizer.defaults['lr'] = recipe['lr']
        head_names = tuple(key for key in named if key.startswith(f"convs.{recipe['layers'] - 1}."))
        for index in range(2):
            _block(actual, reference, {'Adam': actual_optimizer}, {'Adam': reference_optimizer},
                   selected, head_names, unused, condition, prefix, head, core.conditional_loss,
                   engine.block if flickr else engine.update_block, device, engine, fp32=False, zero=zero)
            clocks = engine.clocks() if flickr else engine.verify_clocks()
            assert all(value == (0 if key in unused else 2 * (index + 1))
                       for key, value in clocks.items())


def check(name, device):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Core tensor checks require the supplied CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    with torch.cuda.device(device), torch.random.fork_rng(devices=[device]):
        torch.cuda.manual_seed(733)
        if name == 'arxiv-IGLU':
            _iglu(device)
        elif name in ('arxiv-LMC', 'Flickr-LMC'):
            _lmc(name, device)
        else:
            raise ValueError(name)
