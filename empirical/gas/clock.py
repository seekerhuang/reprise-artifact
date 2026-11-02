import ast
import inspect
import textwrap
import time


class Clock:
    def __init__(self, model):
        self.model = model
        self.output_seconds = self.last_output_seconds = self.shared_setup_seconds = 0.0
        self._active = False
        self._pool_restore = []
        self._class = type(model)
        self._owned = 'mini_inference' in self._class.__dict__
        self._old = self._class.__dict__.get('mini_inference')
        original = inspect.unwrap(self._class.mini_inference)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        function = tree.body[0]
        boundaries = [i for i, node in enumerate(function.body)
                      if isinstance(node, ast.For) and 'self.histories[-1].emb' in ast.unparse(node)]
        if len(boundaries) != 1 or not isinstance(function.body[-1], ast.Return):
            raise RuntimeError('Unknown native history/output boundary')
        function.body.insert(boundaries[0], ast.parse('_begin_output(self)').body[0])
        function.body.insert(-1, ast.parse('_end_output(self)').body[0])
        namespace = dict(original.__globals__, _begin_output=self.begin_output, _end_output=self.end_output)
        exec(compile(ast.fix_missing_locations(tree), __file__ + '::native-output-boundary', 'exec'), namespace)
        self._installed = namespace[original.__name__]
        self._class.mini_inference = self._installed

    def begin_output(self, model):
        if model is not self.model:
            return
        if self._active:
            raise RuntimeError('Nested native output measurement')
        self._started = time.perf_counter()
        self._active = True
        self._shared = 0.0
        pool = model.pool
        for name, field in (('_pull_stream', '_pull_streams'),
                            ('_cpu_buffer', '_cpu_buffers'), ('_cuda_buffer', '_cuda_buffers')):
            if not hasattr(pool, name):
                continue
            original = getattr(pool, name)
            owned, old = name in vars(pool), vars(pool).get(name)

            def counted(index, _original=original, _field=field, _pool=pool):
                if getattr(_pool, _field)[index] is not None:
                    return _original(index)
                started = time.perf_counter()
                value = _original(index)
                self._shared += time.perf_counter() - started
                return value

            self._pool_restore.append((pool, name, owned, old))
            setattr(pool, name, counted)

    def end_output(self, model):
        if model is not self.model:
            return
        if not self._active:
            raise RuntimeError('Native output measurement did not begin')
        for pool, name, owned, old in reversed(self._pool_restore):
            if owned:
                setattr(pool, name, old)
            else:
                delattr(pool, name)
        self._pool_restore.clear()
        self.last_output_seconds = time.perf_counter() - self._started - self._shared
        self.output_seconds += self.last_output_seconds
        self.shared_setup_seconds += self._shared
        self._active = False

    def prior_output(self, current_refresh):
        return self.output_seconds - (self.last_output_seconds if current_refresh else 0.0)

    def close(self):
        if self._active or self._pool_restore:
            raise RuntimeError('Unfinished native output measurement')
        if self._class.mini_inference is not self._installed:
            raise RuntimeError('Native inference binding changed')
        if self._owned:
            self._class.mini_inference = self._old
        else:
            delattr(self._class, 'mini_inference')
