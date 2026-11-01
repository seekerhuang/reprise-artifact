class OnePassDropout:
    def __init__(self, model):
        self.active = False
        self.states = {}
        self.draws = self.replays = self.peak_rng_bytes = 0
        self.originals = []
        modules = [layer.f_dropout for layer in model.aggregators] + [model.classifier.f_dropout]
        if len({id(module) for module in modules}) != len(modules):
            raise ValueError('Distinct original dropout sites required')
        for index, module in enumerate(modules):
            if module.inplace:
                raise ValueError('Original non-inplace dropout required')
            original = module.forward
            self.originals.append((module, original))
            module.forward = self._wrapper(index, module, original)

    def _wrapper(self, index, module, original):
        def forward(value):
            if self.active and module.training:
                if index in self.states:
                    raise RuntimeError('One-pass elision is invalid for a repeated dropout site')
                self.states[index] = True
                self.draws += 1
            return original(value)
        return forward

    def begin(self):
        if self.active or self.states:
            raise RuntimeError('Unreleased dropout scope')
        self.active = True

    def end(self):
        self.states.clear()
        self.active = False

    def close(self):
        self.end()
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()

def adapt(parent):
    class OnePassTraining(parent):
        def __init__(self, model, method):
            if method != 'lateA':
                raise ValueError('Own single-prefix, late-affine M221 only')
            super().__init__(model, method)
            if self.randomness.active or self.randomness.states:
                raise RuntimeError('Only replace an unused wrapper at fresh construction')
            self.randomness.close()
            self.randomness = OnePassDropout(model)

        def work(self):
            return dict(super().work(),
                onepass_dropout_rng_snapshot_elision=True,
                rng_scope='Original dropout forwards and stream; no RNG snapshots or replay')
    return OnePassTraining
