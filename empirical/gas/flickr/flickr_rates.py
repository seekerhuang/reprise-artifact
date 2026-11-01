def rate_at(completed_contexts):
    if not isinstance(completed_contexts, int) or completed_contexts < 0:
        raise ValueError('Context clock must be a nonnegative integer')
    if completed_contexts < 64:
        return 0.0125
    return 0.01 if completed_contexts < 400 else 0.005 if completed_contexts < 600 else 0.0025
