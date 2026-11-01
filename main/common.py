import copy
import csv
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
    def __init__(self, device):
        self.device = device
        self.seconds = 0.0

    def __enter__(self):
        sync(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args):
        sync(self.device)
        self.seconds += time.perf_counter() - self.started


def score(logits, labels, threshold=0.5):
    if labels.ndim == 1:
        return int((logits.argmax(-1) == labels).sum()) / len(labels)
    predicted = logits.sigmoid() > threshold
    target = labels > 0
    true_positive = (predicted & target).sum().double()
    denominator = predicted.sum() + target.sum()
    return float(2 * true_positive / denominator.clamp_min(1))


class Output:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / 'trajectory.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=['updates', 'seconds', 'valid', 'test'])
        self.writer.writeheader()
        self.best = float('-inf')

    def observe(self, model, updates, seconds, valid, test):
        row = dict(updates=updates, seconds=seconds, valid=valid, test=test)
        self.writer.writerow(row)
        self.stream.flush()
        if valid > self.best:
            self.best = valid
            torch.save(dict(model=copy.deepcopy(model.state_dict()), **row), self.directory / 'best.pt')
        print(f'{updates} {seconds:.6f} {valid:.6f} {test:.6f}', flush=True)

    def close(self):
        self.stream.close()
