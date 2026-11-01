import copy

def training_view_without_X(data):
    x = data.x
    if x.ndim != 2 or x.device.type != 'cpu' or x.requires_grad:
        raise ValueError('This execution variant requires fixed CPU input features')
    (nodes, channels) = x.shape
    if channels == 0 or data.num_nodes != nodes:
        raise ValueError('Original full node features required')
    version = x._version
    view = copy.copy(data)
    view.x = x.new_empty((nodes, 0))
    if view is data or view._store is data._store or data.x is not x:
        raise RuntimeError('Training view must not mutate the original PyG store')
    if data.x._version != version or data.x.shape != (nodes, channels):
        raise RuntimeError('Full original feature owner changed')
    for (key, value) in data:
        if key != 'x' and view[key] is not value:
            raise RuntimeError('Non-feature input identity changed: ' + key)
    if view.num_nodes != nodes:
        raise RuntimeError('Zero-width features changed node coordinates')
    return (view, dict(source_shape=[nodes, channels], training_shape=[nodes, 0], full_input_retained_for='PX setup, native full refresh and validation/test evaluation', source_X_unchanged=True, non_feature_objects_shared_unchanged=True, training_feature_values=0, scope='Own cached-first M221 only; all topology, supervision, sampling and state updates retained'))
