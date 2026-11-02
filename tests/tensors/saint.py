import copy
import ctypes
import math
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch


def measurement(reddit, device):
    from empirical.saint.core import geometry, reddit_structural
    from empirical.saint.head import mu_execution, reddit_half_pointwise_v1
    channels = 128 if reddit else 512
    z = torch.nn.functional.normalize(torch.randn(32, channels, device=device), dim=1)
    boundary = torch.randn_like(z)
    branch = torch.rand(32, 1, device=device) > .5
    first, second = torch.randn_like(z), torch.randn_like(z)
    if reddit:
        operation = reddit_half_pointwise_v1.Measurement()

        def original(z, boundary, branch, first, second):
            return reddit_structural.channel_half_difference(geometry, z, boundary, branch, first, second)

        actual = operation.measure(z, boundary, branch, first, second, original)
        expected = original(z, boundary, branch, first, second)
    else:
        operation = mu_execution.Measurement(Path(mu_execution.__file__).parent / 'libmu_elementwise_sm80.so')
        actual = operation.measure(z, boundary, branch, first, geometry.channel_measurement)
        expected = geometry.channel_measurement(z, boundary, branch, first)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert operation.fast_calls == 1 and actual.dtype == torch.float64
    operation.close()


def compare_adam(optimizer):
    groups = []
    pairs = []
    for group in optimizer.param_groups:
        parameters = [torch.nn.Parameter(p.detach().clone()) for p in group['params']]
        groups.append(dict(copy.deepcopy({k: v for k, v in group.items() if k != 'params'}), params=parameters))
        pairs.extend(zip(group['params'], parameters))
    reference = torch.optim.Adam(groups, **copy.deepcopy(optimizer.defaults))
    calls = [0]

    def before(current, args, kwargs):
        for live, expected in zip(current.param_groups, reference.param_groups):
            expected['lr'] = live['lr']
        for live, expected in pairs:
            expected.grad = live.grad.detach().clone() if live.grad is not None else None

    def after(current, args, kwargs):
        reference.step()
        for live, expected in pairs:
            torch.testing.assert_close(live, expected, rtol=0, atol=0)
            actual_state = current.state.get(live, {})
            expected_state = reference.state.get(expected, {})
            assert actual_state.keys() == expected_state.keys()
            for key in actual_state:
                torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
        calls[0] += 1

    handles = optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)
    return handles, calls


def average(model, reddit):
    from empirical.saint.delivery.averaging import CausalAverage
    cadence, after, budget = (20, 800, 1100) if reddit else (216, 10800, 16200)
    operation = CausalAverage(model, after_updates=after, cadence=cadence, window=5)
    live = tuple(p.detach().clone() for p in model.parameters())
    synchronize = lambda: torch.cuda.synchronize(next(model.parameters()).device)
    try:
        for updates in range(cadence, budget + 1, cadence):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(updates)
            with operation.installed(updates, synchronize):
                expected_steps = list(range(after, updates + 1, cadence))[-5:]
                assert [h for h, _ in operation.history] == expected_steps
                expected = sum(expected_steps) / len(expected_steps) if expected_steps else updates
                for parameter in model.parameters():
                    torch.testing.assert_close(parameter, torch.full_like(parameter, expected), rtol=0, atol=0)
            for parameter in model.parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, updates), rtol=0, atol=0)
        assert len(operation.history) == 5 and not operation.is_installed
    finally:
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), live):
                parameter.copy_(original)


def check(name, device, method='reprise'):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'empirical/vendor/graphsaint'))
    from graphsaint.minibatch import Minibatch
    from graphsaint.globals import args_global
    from empirical.saint import model as saint_model
    from empirical.saint.model import GraphSAINT
    from empirical.saint.data import adj_norm
    from empirical.saint.train import engine_class, reddit_engine_class
    from empirical.saint.optim import saint_native_fused_adam_v1
    from empirical.saint.schedules import terminal_beta07, reddit_terminal_groups
    from empirical.saint.delivery.evaluation import saint_evaluate
    from empirical.saint.delivery import yelp_low_peak_eval_identity_input_v3 as evaluation
    from empirical.saint.sparse import reddit_bmm_route, yelp_canonical_public_v4
    from empirical.saint.input import reddit_unit_csr_norm
    reddit = name == 'Reddit2-SAINT'
    if method == 'reprise':
        measurement(reddit, device)
    device = torch.device(device)
    saint_model.args_global.gpu = args_global.gpu = device.index
    full = sp.eye(80, format='csr', dtype=np.int32)
    training = sp.diags(np.r_[np.ones(60), np.zeros(20)]).astype(np.int32).tocsr()
    roles = dict(tr=np.arange(60), va=np.arange(60, 70), te=np.arange(70, 80))
    parameters = dict(lr=.01, weight_decay=0., dropout=.1, sample_coverage=30 if reddit else 1)
    if reddit:
        parameters['dataset'] = 'reddit'
    normalized = reddit_unit_csr_norm.apply(adj_norm, full)[0] if reddit else adj_norm(full)
    minibatch = Minibatch(normalized, training, roles, parameters)
    rng = np.random.default_rng(7)
    features = rng.standard_normal((80, 602 if reddit else 300))
    labels = np.eye(41)[np.arange(80) % 41] if reddit else rng.integers(0, 2, (80, 100)).astype(np.float64)
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
        if method == 'direct':
            from empirical.saint.core.direct import Direct
            engine = Direct(model)
        else:
            engine = reddit_engine_class()(model, 'lateA')
        route = reddit_bmm_route
    else:
        terminal_beta07.BASE_OPTIONS = dict(terminal_beta07.BASE_OPTIONS, fused=True, foreach=False)
        terminal_beta07.install(model)
        if method == 'direct':
            from empirical.saint.core.direct import Direct
            from empirical.saint.sparse.yelp_train_strict_bmm import ROUTE
            engine = Direct(model)
            route = ROUTE
        else:
            engine = engine_class()(model, 'lateA')
            route = yelp_canonical_public_v4.ROUTE
    handles, calls = compare_adam(model.optimizer)
    rates = []
    if method == 'direct':
        def inspect_step(optimizer, args, kwargs):
            rates.append([group['lr'] for group in optimizer.param_groups])
            for group in engine.clip_groups:
                norm = sum(p.grad.double().square().sum() for p in group if p.grad is not None).sqrt()
                assert float(norm) <= 5.00001
        handles += (model.optimizer.register_step_pre_hook(inspect_step),)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    ctypes.CDLL(None).srand(ctypes.c_uint(2026092901 if reddit else 123))
    minibatch.set_sampler(dict(sampler='rw', num_root=2 if reddit else 4, depth=2))
    minibatch.shuffle()
    if not reddit and method == 'reprise':
        yelp_canonical_public_v4.metadata.install(minibatch)
    try:
        with route.strict_determinism(torch), route.install(model):
            for index in range(4):
                if index == 2:
                    engine.visits = 800 if reddit else 8640
                    if reddit:
                        for group in model.optimizer.param_groups:
                            group['lr'] = .0025
                        model.optimizer.defaults['lr'] = model.lr = .0025
                        model.optimizer._phase2_declared_rate['transitions'].append(dict(after_completed_updates=800))
                nodes, matrix, weights = minibatch.one_batch('train')
                assert isinstance(nodes, np.ndarray) if reddit else torch.is_tensor(nodes)
                losses = engine.block(nodes, matrix, weights)
                assert all(torch.isfinite(loss) for loss in losses)
        expected_calls = 4 if method == 'direct' else 12
        assert calls[0] == expected_calls and engine.optimizer_calls == expected_calls
        if method == 'direct':
            if reddit:
                rate = .0025 * (1 + math.cos(math.pi * .5 / 200))
                assert rates[-2] == [rate, rate, rate * .25]
                assert rates[-1] == [rate, rate, rate]
            else:
                rate = .0075 * (2 / 3 * (1 + math.cos(math.pi * .5 / 3780)))
                assert rates[-2:] == [[rate, rate], [rate, rate]]
                assert float(model._r69_late_positive_bce_scalar) == 1.25
        inactive = {'classifier.params.1', 'classifier.params.2'}
        for key, parameter in model.named_parameters():
            assert torch.isfinite(parameter).all(), key
            assert int(model.optimizer.state.get(parameter, {}).get('step', 0)) == (0 if key in inactive else 4 if method == 'direct' else 8), key
            if key in inactive:
                torch.testing.assert_close(parameter, before[key], rtol=0, atol=0)
        assert any(not torch.equal(parameter, before[key]) for key, parameter in model.named_parameters())
        metrics = saint_evaluate(model, minibatch) if reddit else evaluation.evaluate(
            saint_evaluate, model, minibatch, thresholds=(.5, .45))
        assert np.isfinite(metrics['valid_score']) and np.isfinite(metrics['test_score'])
        average(model, reddit)
    finally:
        if not reddit and method == 'reprise':
            yelp_canonical_public_v4.metadata.close(minibatch)
        for handle in handles:
            handle.remove()
        engine.close()
    from saint_training import check as training_check
    training_check(name, device, method)
