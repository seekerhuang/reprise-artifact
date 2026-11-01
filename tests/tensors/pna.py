import torch


def mean_vjp(device):
    from empirical.pna.training import _UnitSeedMeanVJP
    first = torch.randn(9, 6, device=device, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)

    def ports(value):
        return value.sin(), value.square(), value.relu(), value.exp(), value

    original, candidate = ports(first), ports(second)
    cotangents = tuple(torch.randn_like(value) for value in original)
    torch.autograd.backward(original, cotangents)
    _UnitSeedMeanVJP.apply(*candidate, *cotangents).backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def average(device):
    from empirical.pna.averaging import Window, selected_named_parameters
    from main.recipes import RECIPES
    from prepare_initial import factory
    build = factory(RECIPES['Flickr-PNA'])
    model, clone = build().to(device), build().to(device)
    operation = Window(model)
    selected = dict(selected_named_parameters(model))
    original = {key: value.detach().clone() for key, value in model.state_dict().items() if key not in selected}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    for updates in range(4, 1601, 4):
        with torch.no_grad():
            for parameter in selected.values():
                parameter.fill_(updates)
        operation.capture(model, updates)
        clone.load_state_dict(model.state_dict(), strict=True)
        operation.install(clone)
        expected_steps = list(range(800, updates + 1, 4))[-3:]
        assert [h for h, _ in operation.window] == expected_steps
        expected = sum(expected_steps) / len(expected_steps) if expected_steps else updates
        for key, value in clone.state_dict().items():
            if key in selected:
                torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        for parameter in selected.values():
            torch.testing.assert_close(parameter, torch.full_like(parameter, updates), rtol=0, atol=0)
    assert operation.snapshots == 201 and operation.averaged_deliveries == 201
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))


def check(name, device):
    from training import check as training_check
    mean_vjp(device)
    average(device)
    training_check(name, device)
