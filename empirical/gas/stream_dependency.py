import inspect
import types

class Installation:

    def __init__(self, pool):
        import torch
        self.pool = pool
        self.state = dict(enabled=pool is not None, pushes=0, restored=False, scope='This own raw PPI pool only; all prefix, history refresh and final-output pushes', ordering='retire old slot; retain source; push stream waits for current compute producer; original write_async', added_device_wide_synchronizations=0, old_run_actual_race_proven=False)
        if pool is None:
            return
        if 'async_push' in vars(pool):
            raise RuntimeError('Refuse an already-overridden pool')
        native = inspect.unwrap(pool.async_push)
        writer = native.__globals__['write_async']
        if not callable(writer):
            raise RuntimeError('Original native writer unavailable')
        state = self.state

        @torch.no_grad()
        def async_push(this, src, offset, count, dst):
            this._push_index = (this._push_index + 1) % this.pool_size
            this.synchronize_push(this._push_index)
            this._push_cache[this._push_index] = src
            stream = this._push_stream(this._push_index)
            stream.wait_stream(torch.cuda.current_stream(src.device))
            with torch.cuda.stream(stream):
                writer(src, offset, count, dst)
            state['pushes'] += 1
        self.installed = types.MethodType(async_push, pool)
        pool.async_push = self.installed

    def restore(self):
        if self.pool is not None:
            if vars(self.pool).get('async_push') is not self.installed:
                raise RuntimeError('Own pool installation unexpectedly replaced')
            delattr(self.pool, 'async_push')
        self.state['restored'] = True
