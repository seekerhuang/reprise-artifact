import csv
import json
import math
from pathlib import Path
import tempfile

import torch

from main.recipes import DIRECT_CONFIGS


def check_output(directory, method):
    directory = Path(directory)
    result = json.loads((directory / 'results.json').read_text())
    assert result['status'] == 'complete'
    assert all(item['passed'] for item in result['readbacks'].values())
    assert result['verification']['updates'] == result['last']['updates']
    assert result['full_run_seconds'] >= result['best']['seconds'] > 0
    with (directory / 'trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    details = [json.loads(line) for line in (directory / 'trajectory.jsonl').read_text().splitlines()]
    assert len(rows) == len(details) and details[-1]['updates'] == result['last']['updates']
    assert all(later['seconds'] >= earlier['seconds'] for earlier, later in zip(details, details[1:]))
    assert result['costs']['setup_seconds'] > 0 and result['costs']['training_outer_seconds'] > 0
    assert result['costs']['independent_readback_seconds'] > 0
    for row in details:
        costs = row['costs']
        seconds = costs['setup_seconds'] + costs['training_outer_seconds']
        seconds += costs.get('CUDA_context_initialization_seconds', 0.0)
        seconds += costs.get('single_delivery_preparation_seconds', 0.0)
        if 'single_delivery_preparation_seconds' not in costs:
            seconds += costs.get('averaging_seconds', 0.0)
        seconds -= costs.get('current_evaluation_refresh_excluded_seconds', 0.0)
        seconds -= costs.get('prior_history_outputs_excluded_seconds', 0.0)
        seconds -= costs.get('evaluation_setup_excluded_seconds', 0.0)
        assert math.isclose(row['seconds'], seconds, rel_tol=1e-12, abs_tol=1e-12)
    identity = json.loads((directory / 'run.json').read_text())
    memory = 'whole_process_peak_allocated_bytes'
    record_memory = identity['config'] in DIRECT_CONFIGS
    assert (memory in result) == record_memory
    if record_memory:
        assert result[memory] >= max(row[memory] for row in details) > 0
        assert result['whole_process_peak_allocated_GiB'] == result[memory] / 2 ** 30
        assert result['whole_process_peak_reserved_bytes'] >= result[memory]
    assert identity['method'] == method and identity['optimizer'] and identity['inputs']
    if identity['config'] == 'PPI-GCN':
        best = next(row for row in details if row['updates'] == result['best']['updates'])
        expected = result['last']['seconds'] - details[-1]['costs']['single_delivery_preparation_seconds']
        expected += best['costs']['single_delivery_preparation_seconds']
        assert math.isclose(result['full_run_seconds'], expected, rel_tol=1e-12, abs_tol=1e-12)


def check(device):
    from main.common import Output
    from main.recording import Recording
    from main.verification import expected_clocks, verify
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(4, 3, device=device)).square().sum().backward()
    optimizer.step()
    clocks = expected_clocks(model, 1)
    for updates, expected, incorrect in ((0, 1, clocks), (1, 1, expected_clocks(model, 2))):
        try:
            verify(model, optimizer, updates, expected, incorrect, {})
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incomplete updates or wrong Adam clocks accepted')
    rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    recording = Recording(device, memory=True)

    @torch.no_grad()
    def evaluate(current):
        score = float(current(torch.ones(4, 3, device=device)).sum())
        return score, score

    with tempfile.TemporaryDirectory(prefix='reprise-recording-') as temporary:
        output = Output(Path(temporary) / 'output', recording)
        try:
            valid, test = evaluate(model)
            output.observe(model, 1, .01, valid, test)
            path = output.directory / 'best.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            state['model']['bias'].add_(1)
            torch.save(state, path)
            try:
                output.finish(model, 1, evaluate, expected_updates=1, optimizer=optimizer,
                              clocks=clocks, bn_clocks={})
            except RuntimeError as error:
                assert 'checkpoint readback' in str(error)
            else:
                raise AssertionError('Corrupted checkpoint accepted')
        finally:
            output.close()
        result = json.loads((output.directory / 'results.json').read_text())
        assert result['status'] == 'incomplete' and not result['readbacks']['best']['passed']
        assert result['whole_process_peak_allocated_bytes'] == torch.cuda.max_memory_allocated(device)
    assert torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state(device))


if __name__ == '__main__':
    check('cuda:0')
    print('PASS recording', flush=True)
