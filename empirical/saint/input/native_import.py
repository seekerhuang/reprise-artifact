import importlib
import sys
import threading

def _run(loader, args, kwargs, importer):
    record = dict(version='R75-native-import-overlap-no-profiler-v3', worker_joined=False)
    failure = []

    def worker_body():
        try:
            module = importer('torch._dynamo')
            record['import_returned_module'] = getattr(module, '__name__', None)
        except BaseException as exc:
            failure.append((exc, exc.__traceback__))
    worker = threading.Thread(target=worker_body, name='paid-native-Dynamo-import', daemon=False)
    worker.start()
    load_error = None
    try:
        result = loader(*args, **kwargs)
    except BaseException as exc:
        load_error = (exc, exc.__traceback__)
    finally:
        worker.join()
        record['worker_joined'] = not worker.is_alive()
        record['dynamo_loaded_after'] = 'torch._dynamo' in sys.modules
    if load_error is not None:
        (exc, traceback) = load_error
        if failure:
            exc.add_note('The joined Dynamo worker also failed: ' + repr(failure[0][0]))
        raise exc.with_traceback(traceback)
    if failure:
        (exc, traceback) = failure[0]
        raise exc.with_traceback(traceback)
    return (result, record)

def load(loader, *args, **kwargs):
    if 'torch' not in sys.modules:
        raise RuntimeError('Original author import/Torch initialization must already precede the paid load interval')
    (result, record) = _run(loader, args, kwargs, importlib.import_module)
    if not record['worker_joined'] or not record['dynamo_loaded_after'] or record['import_returned_module'] != 'torch._dynamo':
        raise RuntimeError('Complete native Dynamo import and joined worker required before any model construction')
    return (result, record)
