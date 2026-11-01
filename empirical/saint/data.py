import numpy as np
import scipy.sparse as sp

def parse_layer_yml(arch_gcn, dim_input):
    num_layers = len(arch_gcn['arch'].split('-'))
    bias_layer = [arch_gcn['bias']] * num_layers
    act_layer = [arch_gcn['act']] * num_layers
    aggr_layer = [arch_gcn['aggr']] * num_layers
    dims_layer = [arch_gcn['dim']] * num_layers
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    return ([dim_input] + dims_layer, order_layer, act_layer, bias_layer, aggr_layer)

def adj_norm(adj, deg=None, sort_indices=True):
    diag_shape = (adj.shape[0], adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1 / D, 0), shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


def load_yelp(directory):
    import json
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from empirical.saint.input.binary_labels import Decoder
    directory = Path(directory) / 'raw/yelp/raw'
    full = sp.load_npz(directory / 'adj_full.npz').astype(np.bool_)
    training = sp.load_npz(directory / 'adj_train.npz').astype(np.bool_)
    with (directory / 'role.json').open() as stream:
        roles = json.load(stream)
    features = np.load(directory / 'feats.npy')
    present = np.flatnonzero(np.diff(training.indptr) > 0).astype(np.int32, copy=False)
    rows = np.array(list(set(present)))
    scaler = StandardScaler().fit(features[rows])
    features = scaler.transform(features)
    labels, _ = Decoder().attempt((directory / 'class_map.json').read_bytes(), len(features), 100)
    if labels is None:
        with (directory / 'class_map.json').open() as stream:
            mapping = json.load(stream)
        labels = np.zeros((len(features), 100))
        for key, value in mapping.items():
            labels[int(key)] = value
    return full, training, features, labels, roles


def load_reddit(directory):
    import json
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    directory = Path(directory) / 'raw/reddit2/raw'
    full = sp.load_npz(directory / 'adj_full.npz').astype(np.bool_)
    training = sp.load_npz(directory / 'adj_train.npz').astype(np.bool_)
    with (directory / 'role.json').open() as stream:
        roles = json.load(stream)
    features = np.load(directory / 'feats.npy')
    present = np.flatnonzero(np.diff(training.indptr) > 0).astype(np.int32, copy=False)
    rows = np.array(list(set(present)))
    scaler = StandardScaler().fit(features[rows])
    features = scaler.transform(features)
    with (directory / 'class_map.json').open() as stream:
        mapping = {int(key): value for key, value in json.load(stream).items()}
    offset = min(mapping.values())
    labels = np.zeros((len(features), max(mapping.values()) - offset + 1))
    for node, label in mapping.items():
        labels[node, label - offset] = 1
    return full, training, features, labels, roles
