import csv
import json
import time
from contextlib import contextmanager

import torch

from main.common import score


class Output:
    def __init__(self, args, recipe):
        self.args, self.recipe = args, recipe
        self.device = torch.device(args.device)
        self.costs = {}
        self.best = float('-inf')
        self.selected = None
        args.out.mkdir(parents=True, exist_ok=False)
        self.stream = (args.out / 'trajectory.csv').open('w', newline='')
        self.writer = csv.writer(self.stream)
        self.writer.writerow(['epoch', 'updates', 'seconds', 'valid', 'test'])
        identity = dict(config=args.config, method=args.method, initialization=args.initialization,
                        parameters=recipe, torch=torch.__version__, cuda=torch.version.cuda)
        if self.device.type == 'cuda':
            started = time.perf_counter()
            properties = torch.cuda.get_device_properties(self.device)
            torch.cuda.synchronize(self.device)
            self.costs['device_initialization'] = time.perf_counter() - started
            identity.update(gpu=properties.name, gpu_uuid=str(properties.uuid))
        (args.out / 'run.json').write_text(json.dumps(identity, indent=2) + '\n')

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.costs[name] = self.costs.get(name, 0.0) + time.perf_counter() - start

    @property
    def seconds(self):
        return sum(value for key, value in self.costs.items()
                   if key not in ('evaluation', 'checkpoint', 'readback'))

    def save(self, name, model, **values):
        with self.phase('checkpoint'):
            torch.save(dict(model={key: value.detach().cpu().clone()
                                   for key, value in model.state_dict().items()}, **values),
                       self.args.out / (name + '.pt'))

    def observe(self, model, epoch, updates, valid, test):
        row = dict(epoch=epoch, updates=updates, seconds=self.seconds,
                   valid=float(valid), test=float(test))
        self.writer.writerow(row.values())
        self.stream.flush()
        if valid > self.best:
            self.best = float(valid)
            self.selected = row
            self.save('best', model, selection=row)
        self.last = row

    def finish(self, model, evaluate, optimizer=None, *, work=None):
        self.save('endpoint', model, selection=self.last,
                  optimizer=None if optimizer is None else optimizer.state_dict())
        with self.phase('readback'):
            saved = torch.load(self.args.out / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'], strict=True)
            actual = evaluate(model)
            if any(abs(float(actual[i]) - self.selected[key]) > 1e-5
                   for i, key in enumerate(('valid', 'test'))):
                raise RuntimeError('Selected checkpoint evaluation differs')
        result = dict(status='complete', selected=self.selected, endpoint=self.last,
                      full_run_seconds=self.last['seconds'], costs=self.costs)
        if work is not None:
            result['work'] = work
        (self.args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        self.stream.close()


def metrics(logits, data):
    valid = score(logits[data['val_mask']], data['y'][data['val_mask']], 0.5)
    test = score(logits[data['test_mask']], data['y'][data['test_mask']], 0.5)
    return valid, test
