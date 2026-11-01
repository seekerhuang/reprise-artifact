import ctypes
from pathlib import Path
import random
import sys

import numpy as np
import torch

from main.common import Output, Timer, seed, sync
from main.initialization import load_initial


def engine_class():
    from empirical.saint.core import training
    from empirical.saint.core import threeapi
    from empirical.saint.core import adapter
    from empirical.saint.core import production_adapter
    from empirical.saint.core import saint_onepass_dropout_v1 as onepass
    from empirical.saint.head import yelp_serial_blas_workspace_binding_v1 as head
    from empirical.saint.optim import yelp_native_adam_shared_staging_binding_v1 as adam
    from empirical.saint.sparse import yelp_canonical_public_v4 as backend
    from empirical.saint.schedules import late_cosine
    from empirical.saint.schedules import positive_phase
    from empirical.saint.schedules import pulse_schedule
    current = production_adapter.adapter_view(adapter, 'yelp')
    current = head.adapter_view(current)
    current = adam.adapter_view(current)
    return backend.adapt(late_cosine.adapt_S(positive_phase.adapt(pulse_schedule.adapt(
        current.adapt(threeapi.adapt(onepass.adapt(training.Training)), 'ordered')))))



def reddit_engine_class():
    from types import SimpleNamespace
    from empirical.saint.core import training, threeapi, production_adapter, geometry, reddit_structural
    from empirical.saint.core import saint_onepass_dropout_v1 as onepass
    from empirical.saint.head import reddit_half_pointwise_v1 as half
    from empirical.saint.optim import reddit_native_adam_graph_v1 as adam
    from empirical.saint.schedules import reddit_head1_rate, reddit_late_cosine
    from empirical.saint.sparse import reddit_bmm_route as backend

    def adapt(parent, rule):
        if rule != 'ordered':
            raise ValueError(rule)
        return reddit_structural.adapt(parent)

    current = SimpleNamespace(adapt=adapt, specification=lambda rule: {},
                              geometry=geometry, structural=reddit_structural)
    current = production_adapter.adapter_view(current, 'reddit')
    current = half.adapter_view(current)
    current = adam.adapter_view(current)
    return backend.adapt(reddit_head1_rate.adapt(reddit_late_cosine.adapt(
        current.adapt(threeapi.adapt(onepass.adapt(training.Training)), 'ordered'))))

def run(args, recipe):
    if recipe['dataset'] == 'reddit2':
        return run_reddit(args, recipe)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/graphsaint'))
    from graphsaint.minibatch import Minibatch
    from empirical.saint import model as saint_model
    from empirical.saint.model import GraphSAINT
    from empirical.saint.data import adj_norm, load_yelp
    from empirical.saint.schedules import terminal_beta07
    from empirical.saint.optim import saint_native_fused_adam_v1
    from empirical.saint.delivery.evaluation import saint_evaluate
    from empirical.saint.sparse import yelp_canonical_public_v4 as backend
    from empirical.saint.delivery import yelp_low_peak_eval_identity_input_v3 as evaluation
    from empirical.saint.delivery.averaging import CausalAverage
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    saint_model.args_global.gpu = device.index
    timer = Timer(device)
    with timer:
        seed(123)
        ctypes.CDLL(None).srand(ctypes.c_uint(123))
        full, training, features, labels, roles = load_yelp(args.data_dir)
        full, training = full.astype(np.int32), training.astype(np.int32)
        normalized = adj_norm(full)
        parameters = dict(lr=0.01, weight_decay=0.0, dropout=0.1, sample_coverage=50)
        architecture = dict(dim=512, aggr='concat', loss='sigmoid', arch='1-1-0', act='relu', bias='norm')
        minibatch = Minibatch(normalized, training, roles, parameters)
        model = GraphSAINT(100, architecture, parameters, features, labels).to(device)
        unused_minibatch = Minibatch(normalized, training, roles, parameters, cpu_eval=True)
        unused_model = GraphSAINT(100, architecture, parameters, features, labels, cpu_eval=True)
        load_initial(model, args)
        saint_native_fused_adam_v1.install(model)
        for group in model.optimizer.param_groups:
            group.update(lr=recipe['lr'], weight_decay=0.0)
        model.optimizer.defaults.update(lr=recipe['lr'], weight_decay=0.0)
        model.dropout, model.lr = recipe['dropout'], recipe['lr']
        for layer in [*model.aggregators, model.classifier]:
            layer.f_dropout.p = recipe['dropout']
        terminal_beta07.BASE_OPTIONS = dict(terminal_beta07.BASE_OPTIONS, fused=True, foreach=False)
        terminal_beta07.install(model)
        engine = engine_class()(model, 'lateA')
        average = CausalAverage(model)
        ctypes.CDLL(None).srand(ctypes.c_uint(123))
        minibatch.set_sampler(dict(sampler='rw', num_root=1250, depth=2, end=75))
        if len(minibatch.subgraphs_remaining_nodes) < recipe['updates'] // 2:
            raise RuntimeError('The native warm pool does not cover the training budget')
        seed(2026092901)
        ctypes.CDLL(None).srand(ctypes.c_uint(2026092901))
        del unused_model, unused_minibatch
    output = Output(args.out)
    try:
        for _ in range(recipe['epochs']):
            with timer:
                minibatch.shuffle()
                backend.metadata.install(minibatch)
                try:
                    route = backend.ROUTE
                    with route.strict_determinism(torch), route.install(model):
                        while not minibatch.end():
                            if not minibatch.subgraphs_remaining_nodes:
                                raise RuntimeError('The native warm pool is exhausted')
                            nodes, matrix, weights = minibatch.one_batch('train')
                            engine.block(nodes, matrix, weights)
                            minibatch.batch_num += 1
                finally:
                    backend.metadata.close(minibatch)
            with average.installed(engine.visits, lambda: sync(device)):
                metrics = evaluation.evaluate(saint_evaluate, model, minibatch, thresholds=(0.5, 0.45))
                output.observe(model, engine.visits, timer.seconds + average.required_seconds,
                               metrics['valid_score'], metrics['test_score_045'])
    finally:
        engine.close()
        output.close()


def run_reddit(args, recipe):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/graphsaint'))
    from graphsaint.minibatch import Minibatch
    from empirical.saint import model as saint_model
    from empirical.saint.model import GraphSAINT
    from empirical.saint.data import adj_norm, load_reddit
    from empirical.saint.input import reddit_unit_csr_norm, native_import
    from empirical.saint.optim import saint_native_fused_adam_v1
    from empirical.saint.schedules import reddit_terminal_groups
    from empirical.saint.sparse import reddit_bmm_route as backend
    from empirical.saint.delivery.averaging import CausalAverage
    from empirical.saint.delivery.evaluation import saint_evaluate
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    saint_model.args_global.gpu = device.index
    timer = Timer(device)
    with timer:
        seed(recipe['sampler_seed'])
        ctypes.CDLL(None).srand(ctypes.c_uint(recipe['sampler_seed']))
        (full, training, features, labels, roles), _ = native_import.load(load_reddit, args.data_dir)
        full, training = full.astype(np.int32), training.astype(np.int32)
        normalized, _ = reddit_unit_csr_norm.apply(adj_norm, full)
        parameters = dict(lr=0.01, weight_decay=0.0, dropout=0.1,
                          sample_coverage=recipe['sample_coverage'], dataset='reddit')
        architecture = dict(dim=recipe['hidden'], aggr='concat', loss=recipe['loss'],
                            arch=recipe['architecture'], act='relu', bias='norm')
        minibatch = Minibatch(normalized, training, roles, parameters)
        model = GraphSAINT(41, architecture, parameters, features, labels).to(device)
        load_initial(model, args)
        saint_native_fused_adam_v1.install(model)
        for group in model.optimizer.param_groups:
            group.update(lr=recipe['lr'], eps=recipe['eps'], weight_decay=0.0)
        model.optimizer.defaults.update(lr=recipe['lr'], eps=recipe['eps'], weight_decay=0.0)
        model.dropout, model.lr = recipe['dropout'], recipe['lr']
        for layer in [*model.aggregators, model.classifier]:
            layer.f_dropout.p = recipe['dropout']
        model.optimizer._phase2_declared_rate = dict(initial_lr=recipe['lr'], transitions=[], current_lr=recipe['lr'])
        reddit_terminal_groups.install(model, model.optimizer)
        engine = reddit_engine_class()(model, 'lateA')
        average = CausalAverage(model, after_updates=recipe['average_start'],
                                cadence=recipe['eval_every'], window=recipe['average_window'])
        ctypes.CDLL(None).srand(ctypes.c_uint(recipe['sampler_seed']))
        minibatch.set_sampler(dict(sampler='rw', num_root=recipe['root_count'], depth=recipe['walk_depth']))
        if len(minibatch.subgraphs_remaining_nodes) < recipe['updates'] // 2:
            raise RuntimeError('The native warm pool does not cover the training budget')
        seed(recipe['training_seed'])
        ctypes.CDLL(None).srand(ctypes.c_uint(recipe['training_seed']))
    output = Output(args.out)
    try:
        for _ in range(recipe['epochs']):
            with timer:
                with backend.strict_determinism(torch), backend.install(model):
                    for step, rate in recipe['milestones']:
                        if engine.visits == step:
                            before = [group['lr'] for group in model.optimizer.param_groups]
                            for group in model.optimizer.param_groups:
                                group['lr'] = rate
                            model.optimizer.defaults['lr'] = model.lr = rate
                            phase = model.optimizer._phase2_declared_rate
                            phase['current_lr'] = rate
                            phase['transitions'].append(dict(after_completed_updates=step, before_group_lrs=before,
                                                             after_group_lrs=[rate] * len(before)))
                    minibatch.shuffle()
                    while not minibatch.end():
                        if not minibatch.subgraphs_remaining_nodes:
                            raise RuntimeError('The native warm pool is exhausted')
                        nodes, matrix, weights = minibatch.one_batch('train')
                        engine.block(nodes, matrix, weights)
                        minibatch.batch_num += 1
            with average.installed(engine.visits, lambda: sync(device)):
                metrics = saint_evaluate(model, minibatch)
                output.observe(model, engine.visits, timer.seconds + average.required_seconds,
                               metrics['valid_score'], metrics['test_score'])
    finally:
        engine.close()
        output.close()
