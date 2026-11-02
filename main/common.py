import copy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device, recording=None, name=None, group=None):
        self.device = device
        self.seconds = 0.0
        self.recording, self.name, self.group = recording, name, group

    def __enter__(self):
        sync(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args):
        sync(self.device)
        elapsed = time.perf_counter() - self.started
        self.seconds += elapsed
        if self.recording is not None:
            self.recording.add(self.name, elapsed, self.group)


def score(logits, labels, threshold=0.5):
    if labels.ndim == 1:
        return int((logits.argmax(-1) == labels).sum()) / len(labels)
    predicted = logits.sigmoid() > threshold
    target = labels > 0
    true_positive = (predicted & target).sum().double()
    denominator = predicted.sum() + target.sum()
    return float(2 * true_positive / denominator.clamp_min(1))


class Output:
    def __init__(self, directory, recording=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / 'trajectory.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=['updates', 'seconds', 'valid', 'test'])
        self.writer.writeheader()
        self.best = float('-inf')
        self.recording = recording
        self.best_row = self.last_row = None
        self.best_delivery_seconds = 0.0
        self.full_seconds = None
        self.result = dict(status='incomplete')
        self.details = (self.directory / 'trajectory.jsonl').open('w') if recording is not None else None
        if recording is not None and recording.identity is not None:
            (self.directory / 'run.json').write_text(json.dumps(recording.identity, indent=2, allow_nan=False) + '\n')

    def observe(self, model, updates, seconds, valid, test):
        if self.recording is not None:
            seconds += self.recording.costs.get('CUDA_context_initialization_seconds', 0.0)
            seconds -= self.recording.costs.get('evaluation_setup_excluded_seconds', 0.0)
        row = dict(updates=updates, seconds=seconds, valid=valid, test=test)
        self.last_row = row
        self.writer.writerow(row)
        self.stream.flush()
        if valid > self.best:
            self.best = valid
            self.best_row = row
            if self.recording is not None:
                self.best_delivery_seconds = self.recording.costs.get('single_delivery_preparation_seconds', 0.0)
            started = time.perf_counter()
            torch.save(dict(model=copy.deepcopy(model.state_dict()), **row), self.directory / 'best.pt')
            if self.recording is not None:
                self.recording.add('selected_save_seconds', time.perf_counter() - started)
        if self.details is not None:
            started = time.perf_counter()
            self.details.write(json.dumps(dict(**row, **self.recording.snapshot()), allow_nan=False) + '\n')
            self.details.flush()
            self.recording.add('metadata_write_seconds', time.perf_counter() - started)
        print(f'{updates} {seconds:.6f} {valid:.6f} {test:.6f}', flush=True)

    def finish(self, model, updates, evaluate, *, expected_updates, optimizer=None,
               clocks, bn_clocks, tolerances=(0.0, 0.0), counts=None, readback=None):
        from main.verification import verify
        self.result['verification'] = verify(model, optimizer if optimizer is not None else model.optimizer,
                                              updates, expected_updates, clocks, bn_clocks)
        with self.recording.phase('endpoint_evaluation_seconds'):
            with torch.no_grad():
                model.eval()
                valid, test = evaluate(model)
        endpoint = dict(updates=updates, seconds=self.last_row['seconds'], valid=valid, test=test)
        with self.recording.phase('endpoint_save_seconds'):
            torch.save(dict(model=copy.deepcopy(model.state_dict()), **endpoint), self.directory / 'endpoint.pt')
        readbacks = {}
        try:
            for name in ('best', 'endpoint'):
                with self.recording.phase('independent_readback_seconds'):
                    checkpoint = torch.load(self.directory / (name + '.pt'), map_location='cpu', weights_only=False)
                    model.load_state_dict(checkpoint['model'], strict=True)
                    model.eval()
                    with torch.no_grad():
                        valid, test = evaluate(model) if readback is None else readback(model, checkpoint)
                    errors = (abs(valid - checkpoint['valid']), abs(test - checkpoint['test']))
                    passed = all(error <= tolerance for error, tolerance in zip(errors, tolerances))
                    readbacks[name] = dict(valid=valid, test=test, errors=errors,
                                           tolerances=tolerances, passed=passed)
                    if counts is not None:
                        actual = (valid, test)
                        expected = (checkpoint['valid'], checkpoint['test'])
                        count_errors = tuple(abs(round(a * n) - round(b * n)) for a, b, n in zip(actual, expected, counts))
                        passed = (all(abs(value * n - round(value * n)) <= 1e-6
                                      for values in (actual, expected) for value, n in zip(values, counts))
                                  and all(error <= round(tolerance * n)
                                          for error, tolerance, n in zip(count_errors, tolerances, counts)))
                        readbacks[name].update(correct_count_errors=count_errors, passed=passed)
                    if not passed:
                        raise RuntimeError(f'{name} checkpoint readback: {readbacks[name]}')
        finally:
            model.load_state_dict(torch.load(self.directory / 'endpoint.pt', map_location='cpu',
                                             weights_only=False)['model'], strict=True)
            self.result.update(endpoint=endpoint, readbacks=readbacks)
        self.result['status'] = 'complete'

    def close(self):
        self.stream.close()
        if self.details is not None:
            for callback in reversed(self.recording.close_callbacks):
                callback()
            self.details.close()
            result = dict(**self.recording.snapshot(), **self.result, best=self.best_row, last=self.last_row,
                          full_run_seconds=((self.last_row['seconds'] if self.full_seconds is None else self.full_seconds)
                                            + self.recording.costs.get('training_close_seconds', 0.0)
                                            if self.result['status'] == 'complete' else None))
            (self.directory / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
