from contextlib import contextmanager
import time

import torch


class Recording:
    def __init__(self, device, *, memory=False):
        self.device = torch.device(device)
        self.record_memory = memory
        self.started = time.perf_counter()
        self.costs = {}
        self.identity = None
        self.close_callbacks = []
        if self.device.type == 'cuda':
            started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            self.costs['CUDA_context_initialization_seconds'] = time.perf_counter() - started

    def identify(self, args, recipe, model, optimizer):
        from main.identity import describe
        with self.phase('identity_record_seconds'):
            self.identity = describe(args, recipe, model, optimizer)

    def add(self, name, seconds, group=None):
        values = self.costs if group is None else self.costs.setdefault(group, {})
        values[name] = values.get(name, 0.0) + seconds

    @contextmanager
    def phase(self, name, *, group=None, synchronize=True):
        if synchronize and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.add(name, time.perf_counter() - started, group)

    def memory(self):
        if not self.record_memory or self.device.type != 'cuda':
            return {}
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        return dict(whole_process_peak_allocated_bytes=allocated,
                    whole_process_peak_reserved_bytes=reserved,
                    whole_process_peak_allocated_GiB=allocated / 2 ** 30,
                    whole_process_peak_reserved_GiB=reserved / 2 ** 30)

    def snapshot(self):
        costs = {key: dict(value) if isinstance(value, dict) else value for key, value in self.costs.items()}
        if 'history_refresh_seconds' in costs:
            costs['training_without_history_refresh_seconds'] = costs['training_outer_seconds'] - costs['history_refresh_seconds']
            costs['required_history_refresh_seconds'] = (costs['setup'].get('initial_history_fill_seconds', 0.0)
                                                          + costs['history_refresh_seconds']
                                                          - costs['current_evaluation_refresh_excluded_seconds']
                                                          - costs['prior_history_outputs_excluded_seconds'])
        return dict(costs=costs,
                    elapsed_wall_seconds=time.perf_counter() - self.started, **self.memory())
