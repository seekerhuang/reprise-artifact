def verify(model, optimizer, updates, expected_updates, clocks, bn_clocks):
    if updates != expected_updates:
        raise RuntimeError(f'Incomplete updates: {updates} != {expected_updates}')
    optimizers = list(optimizer.values()) if isinstance(optimizer, dict) else [optimizer]
    actual = {name: max(int(opt.state.get(parameter, {}).get('step', 0)) for opt in optimizers)
              for name, parameter in model.named_parameters()}
    batches = {name: int(value) for name, value in model.named_buffers()
               if name.endswith('num_batches_tracked')}
    if actual != clocks:
        raise RuntimeError(f'Adam clocks: {actual}, expected {clocks}')
    for name, parameter in model.named_parameters():
        if clocks[name] == 0 and (parameter.grad is not None or any(opt.state.get(parameter, {}) for opt in optimizers)):
            raise RuntimeError(f'Unused parameter acquired optimizer state: {name}')
    if batches != bn_clocks:
        raise RuntimeError(f'BN clocks: {batches}, expected {bn_clocks}')
    return dict(updates=updates, parameter_clocks=actual, BN_num_batches_tracked=batches)


def expected_clocks(model, updates, unused=()):
    return {name: 0 if name in unused else updates for name, _ in model.named_parameters()}


def expected_bn(model, calls, unused=()):
    return {name: 0 if name in unused else calls for name, _ in model.named_buffers()
            if name.endswith('num_batches_tracked')}
