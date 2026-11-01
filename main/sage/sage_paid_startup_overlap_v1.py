VERSION = 'SAGE-paid-native-Dynamo-import-and-original-array-loader-overlap-v1'

def load_training_arrays(original_loader, directory, split_type, device):
    import importlib
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        imported = pool.submit(importlib.import_module, 'torch._dynamo')
        arrays = original_loader(directory, split_type, device)
        imported.result()
    return arrays
