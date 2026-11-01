import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graph_adapters import affine_case
from reference import basis, check_block, close


def normalization(device, channels):
    from empirical.saint.core import geometry
    hidden = torch.randn(7, channels, device=device, dtype=torch.float64)
    hidden[0].zero_()
    hidden[1].zero_()
    hidden[1, 1] = 1e-12
    hidden[2].zero_()
    hidden[2, 1] = .5e-12
    hidden.requires_grad_(True)
    normalized, branch = geometry.normalize_capture(hidden)
    gate = (torch.arange(channels, device=device) % 3 != 0).to(hidden.dtype)
    native = F.normalize(hidden, p=2, dim=1, eps=1e-12)
    close(normalized, native, atol=0, rtol=0)
    boundary = native * gate
    cotangent = torch.randn_like(hidden)
    gradient = torch.autograd.grad((boundary * cotangent).sum(), hidden)[0]
    expected = (hidden.detach() * gradient).sum(0)
    actual = geometry.channel_measurement(normalized, normalized * gate, branch, cotangent)
    close(actual, expected, atol=2e-10, rtol=1e-9)
    assert not branch[0] and branch[1] and not branch[2]


def bounded_decode(device, reddit):
    from empirical.saint.core import reddit_production_graph_v3, yelp_production_graph_v3
    channels = 128 if reddit else 512
    decode = reddit_production_graph_v3._tensor_decode_half if reddit else yelp_production_graph_v3._tensor_decode
    gamma = torch.randn(channels, device=device, dtype=torch.float64)
    beta = torch.randn_like(gamma)
    gamma[0] = beta[0] = 0
    selected = {'gamma': gamma, 'beta': beta}
    matrix = basis(selected, [('gamma', 0, 0, 1), ('beta', 0, 0, 1)])
    for case in ('ordinary', 'limited', 'zero', 'at_cap', 'just_outside', 'outside'):
        gradients = [torch.full((channels, 2 * channels), .0003, device=device, dtype=torch.float64),
                     torch.full((channels,), .02, device=device, dtype=torch.float64),
                     torch.randn_like(gamma) * .03, torch.randn_like(beta) * .03]
        if case in ('at_cap', 'just_outside'):
            for gradient in gradients:
                gradient.zero_()
            gradients[0][0, 0] = 5. if case == 'at_cap' else torch.nextafter(gamma.new_tensor(5.), gamma.new_tensor(6.))
        if case == 'outside':
            gradients[2][1] = 6.
        half = torch.randn_like(gamma) * (100. if case == 'limited' else .001)
        if case == 'zero':
            half.zero_()
        coefficient = gamma.new_tensor(.7)
        raw = matrix @ (torch.linalg.pinv(matrix.T @ matrix, hermitian=True) @ half)
        innovation = raw * coefficient
        mean = torch.cat(gradients[2:])
        full_square = sum(value.square().sum() for value in gradients)
        lower, upper = gamma.new_zeros(()), gamma.new_ones(())
        for _ in range(80):
            middle = (lower + upper) * .5
            first = full_square - mean.square().sum() + (mean - middle * innovation).square().sum()
            second = full_square - mean.square().sum() + (mean + middle * innovation).square().sum()
            feasible = torch.maximum(first, second) <= 25.
            lower = torch.where(feasible, middle, lower)
            upper = torch.where(feasible, upper, middle)
        expected_tau = torch.where(innovation.square().sum() == 0, gamma.new_ones(()),
                                   torch.where(full_square >= 25., gamma.new_zeros(()), lower))
        inputs = half if reddit else (torch.zeros_like(half), half * 2)
        actual = decode(tuple(selected.values()), tuple(gradients), (2, 3), inputs, coefficient, None)
        close(actual.tau, expected_tau, atol=2e-9, rtol=1e-8)
        for position, sign in enumerate((-1., 1.)):
            expected = (mean + sign * expected_tau * innovation).reshape(2, channels)
            close(torch.stack(actual.pair[position]), expected, atol=2e-9, rtol=1e-8)
        if case == 'limited':
            assert 0 < actual.tau < 1
        if case in ('at_cap', 'just_outside', 'outside'):
            assert actual.tau == 0
        assert torch.isfinite(torch.stack(actual.pair[0])).all()


def check(name, device):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'empirical/vendor/graphsaint'))
    from empirical.saint import model as model_module
    from empirical.saint.model import GraphSAINT
    from empirical.saint.train import engine_class, reddit_engine_class
    from empirical.saint.optim import saint_native_fused_adam_v1
    from empirical.saint.schedules import terminal_beta07, reddit_terminal_groups
    from empirical.saint.sparse import reddit_bmm_route, yelp_canonical_public_v4
    reddit = name == 'Reddit2-SAINT'
    normalization(device, 128 if reddit else 512)
    bounded_decode(device, reddit)
    device = torch.device(device)
    model_module.args_global.gpu = device.index
    n = 12
    nodes = np.arange(n, dtype=np.int64)
    features = torch.randn(n, 602 if reddit else 300, device=device).cpu().numpy()
    labels = (F.one_hot(torch.arange(n, device=device) % 41, 41).double() if reddit else
              torch.randint(0, 2, (n, 100), device=device).double()).cpu().numpy()
    matrix = torch.eye(n, device=device).to_sparse().coalesce()
    weights = torch.full((n,), .01, device=device)
    layout = [(f'conv_layers.{3 if reddit else 2}.params.{index}', 0, 0, 1) for index in (2, 1)]
    head_names = {'classifier.f_lin.0.weight', 'classifier.params.0', 'classifier.params.1', 'classifier.params.2'}
    inactive = {'classifier.params.1', 'classifier.params.2'}
    head_names -= inactive
    route = reddit_bmm_route if reddit else yelp_canonical_public_v4.ROUTE

    def build():
        parameters = dict(lr=.01, dropout=.1, weight_decay=0.)
        architecture = dict(dim=128 if reddit else 512, aggr='concat', loss='softmax' if reddit else 'sigmoid',
                            arch='1-0-1-0' if reddit else '1-1-0', act='relu', bias='norm')
        model = GraphSAINT(41 if reddit else 100, architecture, parameters, features, labels).to(device)
        saint_native_fused_adam_v1.install(model)
        lr, dropout = (.01, .125) if reddit else (.015, .05)
        for group in model.optimizer.param_groups:
            group.update(lr=lr, **({'eps': 1e-6} if reddit else {}))
        model.optimizer.defaults.update(lr=lr, **({'eps': 1e-6} if reddit else {}))
        model.dropout, model.lr = dropout, lr
        for layer in [*model.aggregators, model.classifier]:
            layer.f_dropout.p = dropout
        if reddit:
            model.optimizer._phase2_declared_rate = dict(transitions=[])
            reddit_terminal_groups.install(model, model.optimizer)
            engine = reddit_engine_class()(model, 'lateA')
        else:
            terminal_beta07.BASE_OPTIONS = dict(terminal_beta07.BASE_OPTIONS, fused=True, foreach=False)
            terminal_beta07.install(model)
            engine = engine_class()(model, 'lateA')
        return engine, model.optimizer

    def prefix(engine):
        engine.randomness.begin()
        return engine._structured_prefix(nodes, matrix)

    def objective(engine, boundary):
        model = engine.model
        logits = model.classifier.f_lin[0](boundary) + model.classifier.f_bias[0]
        target = model.label_full_cat[nodes] if reddit else model.label_full[nodes]
        return model._loss(logits, target, weights)

    def clip_head(engine):
        torch.nn.utils.clip_grad_norm_(list(engine.head.values()), 5.)

    def validate(reference, gradients, recovered):
        for names in reference.groups.values():
            for raw in gradients:
                norm = sum(raw[name].double().square().sum() for name in names).sqrt()
                assert norm < 5
        assert max(value.abs().max() for value in recovered.values()) > 2e-6
        assert engine._L_coefficient == 1 and engine._graph_decoder._output.tau == 1

    for zero in (False, True):
        engine, optimizer = build()
        affine_case(engine.model, [row[0] for row in layout], zero)
        try:
            with route.strict_determinism(torch), route.install(engine.model):
                for _ in range(1 if zero else 2):
                    reference = check_block(engine, optimizer, build, prefix, objective,
                                            lambda current: current.block(nodes, matrix, weights),
                                            head_names, layout, clip_head=clip_head, validate=validate,
                                            atol=2e-7, rtol=3e-4)
                    reference._release_structural()
                    reference.close()
        finally:
            engine.close()
