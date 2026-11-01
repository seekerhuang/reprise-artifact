import copy
import inspect

import torch


def close(actual, expected, *, atol=3e-5, rtol=4e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def basis(parameters, layout):
    groups = {}
    for name, group, axis, sign in layout:
        groups[group] = parameters[name].shape[axis]
    offsets, width = {}, 0
    for group, size in groups.items():
        offsets[group], width = width, width + size
    count = sum(parameters[name].numel() for name, _, _, _ in layout)
    matrix = next(iter(parameters.values())).new_zeros((count, width), dtype=torch.float64)
    start = 0
    for name, group, axis, sign in layout:
        value = parameters[name].detach().double()
        shape = [1] * value.ndim
        shape[axis] = value.shape[axis]
        columns = torch.arange(value.shape[axis], device=value.device).reshape(shape).expand_as(value).reshape(-1)
        rows = torch.arange(value.numel(), device=value.device) + start
        matrix[rows, columns + offsets[group]] = value.reshape(-1) * sign
        start += value.numel()
    return matrix


def projection(parameters, layout, first, second):
    matrix = basis(parameters, layout)
    names = [name for name, _, _, _ in layout]
    flat = lambda values: torch.cat([values[name].detach().double().reshape(-1) for name in names])
    g0, g1 = flat(first), flat(second)
    measured = matrix.T @ g0, matrix.T @ g1
    innovation = matrix @ (torch.linalg.pinv(matrix.T @ matrix, hermitian=True) @ ((measured[1] - measured[0]) * .5))
    recovered = {}
    start = 0
    for name in names:
        parameter = parameters[name]
        recovered[name] = innovation[start:start + parameter.numel()].reshape_as(parameter).to(parameter.dtype)
        start += parameter.numel()
    close(matrix.T @ (innovation - (g1 - g0) * .5), torch.zeros_like(measured[0]), atol=2e-8, rtol=1e-6)
    return recovered, measured


def local_measurements(engine):
    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            values = frame.f_locals
            if values.get('self') is engine and frame.f_code.co_name in ('block', 'update_block'):
                for name in ('measurements', 'mu'):
                    found = values.get(name)
                    if isinstance(found, (tuple, list)) and len(found) == 2 and all(torch.is_tensor(x) for x in found):
                        return tuple(x.detach().clone().reshape(-1) for x in found)
                for names in (('a0', 'a1'), ('mu0', 'mu1')):
                    if all(name in values for name in names):
                        return tuple(values[name].detach().clone().reshape(-1) for name in names)
                if torch.is_tensor(values.get('difference_half')):
                    return values['difference_half'].detach().clone().reshape(-1)
            frame = frame.f_back
    finally:
        del frame
    return None


def check_block(engine, optimizer, build, prefix, objective, invoke, head_names, layout,
                *, measure=None, clip_head=None, validate=None, atol=3e-5, rtol=4e-4):
    model = engine.model
    named = dict(model.named_parameters())
    body = {name: parameter for name, parameter in named.items() if name not in head_names}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(next(model.parameters()).device)
    reference, native = build()
    reference.model.load_state_dict(model.state_dict(), strict=True)
    native.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for actual_history, reference_history in zip(getattr(model, 'histories', ()), getattr(reference.model, 'histories', ())):
        reference_history.emb.copy_(actual_history.emb)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, next(model.parameters()).device)
    records, measurements = [], []
    counts = {name: 0 for name in body}
    mean_gradients = {}
    handles = []
    for name, parameter in body.items():
        def backward_hook(gradient, name=name):
            counts[name] += 1
            mean_gradients[name] = gradient.detach().clone()
        handles.append(parameter.register_hook(backward_hook))

    def before(current, args, kwargs):
        if any(parameter.grad is not None for parameter in body.values()):
            records.append({name: parameter.grad.detach().clone() for name, parameter in body.items() if parameter.grad is not None})
            value = measure() if measure is not None else local_measurements(engine)
            if value is not None and not measurements:
                measurements.append(value)

    handles.append(optimizer.register_step_pre_hook(before))
    try:
        invoke(engine)
    finally:
        for handle in handles:
            handle.remove()
    outgoing_cpu, outgoing_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(next(model.parameters()).device)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, next(model.parameters()).device)
    reference.model.train()
    reference_named = dict(reference.model.named_parameters())
    selected = {name: reference_named[name] for name, _, _, _ in layout}
    try:
        boundary = prefix(reference)
        gradients = []
        for query in range(2):
            loss = objective(reference, boundary)
            values = torch.autograd.grad(loss, tuple(reference_named.values()), retain_graph=True, allow_unused=True)
            gradients.append({name: value.detach().clone() for name, value in zip(reference_named, values) if value is not None})
            if query == 0:
                native.zero_grad(set_to_none=True)
                for name in head_names:
                    reference_named[name].grad = gradients[-1][name].clone()
                if clip_head is not None:
                    clip_head(reference)
                torch.optim.Adam.step(native)
        active = {name for name in body if name in gradients[0]}
        assert len(records) == 2 and records[0].keys() == active and records[1].keys() == active
        assert all(counts[name] == int(name in active) for name in body), counts
        recovered, expected_measurements = projection(selected, layout, gradients[0], gradients[1])
        if validate is not None:
            validate(reference, gradients, recovered)
        assert measurements, 'Actual adapter measurements were not observed'
        actual_measurements = measurements[0]
        if torch.is_tensor(actual_measurements):
            close(actual_measurements, (expected_measurements[1] - expected_measurements[0]) * .5, atol=atol, rtol=rtol)
        else:
            for actual, expected in zip(actual_measurements, expected_measurements):
                close(actual, expected, atol=atol, rtol=rtol)
        for name in active:
            mean = (gradients[0][name] + gradients[1][name]) * .5
            innovation = recovered.get(name, torch.zeros_like(mean))
            close(mean_gradients[name], mean, atol=atol, rtol=rtol)
            close((records[0][name] + records[1][name]) * .5, mean, atol=atol, rtol=rtol)
            close(records[0][name], mean - innovation, atol=atol, rtol=rtol)
            close(records[1][name], mean + innovation, atol=atol, rtol=rtol)
        assert torch.equal(outgoing_cpu, torch.get_rng_state())
        assert torch.equal(outgoing_cuda, torch.cuda.get_rng_state(next(model.parameters()).device))
    finally:
        torch.set_rng_state(outgoing_cpu)
        torch.cuda.set_rng_state(outgoing_cuda, next(model.parameters()).device)
    return reference
