def expected_clocks(model, engine, updates, direct):
    parameters = dict(model.named_parameters())
    unused = {'classifier.params.1', 'classifier.params.2'}
    count = 24 if model.sigmoid_loss else 28
    if len(parameters) != count or not unused <= parameters.keys() or list(model.named_buffers()):
        raise RuntimeError('Original SAINT parameter support or empty buffer domain changed')
    calls = updates if direct else 3 * updates // 2
    if engine.visits != updates or engine.optimizer_calls != calls or getattr(engine, 'active_cache', False):
        raise RuntimeError('Original complete SAINT update/API budget or empty cache boundary violated')
    for name in unused:
        parameter = parameters[name]
        if model.optimizer.state.get(parameter, {}) or parameter.grad is not None:
            raise RuntimeError('Original unused SAINT parameter acquired optimizer state or gradient')
    return {name: 0 if name in unused else updates for name in parameters}
