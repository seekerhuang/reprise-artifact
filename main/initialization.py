import numpy as np
import torch


SEEDS = (None, 2026102701, 2026102702)
PREFERRED = {'top': 123, 'products_gcn': 2028, 'iglu': 2026092901,
             'lmc': 2027, 'history': 123, 'ppi': 123, 'saint': 123}


def construct(factory, args, recipe, *, device=None):
    family = recipe['family']
    preferred = PREFERRED.get(family, 123)
    if family == 'sage':
        preferred = 123 if recipe['dataset'] == 'arxiv' else 2026092901
    if args.config == 'Reddit2-SAINT':
        preferred = 1
    if args.config == 'Yelp-GCN':
        preferred = 2027
    torch.manual_seed(preferred)
    if family in ('history', 'ppi', 'pna'):
        draws = 2
    elif family == 'lmc':
        if recipe['dataset'] == 'arxiv':
            torch.randperm(80)
        draws = 6
    elif family == 'top':
        draws = 1
    else:
        draws = 0
    for _ in range(draws):
        torch.empty((), dtype=torch.int64).random_()
    model = factory()
    if device is not None:
        model = model.to(device)
    if family == 'top':
        for _ in range(4):
            model.reset_parameters()
            for _ in range(5):
                torch.empty((), dtype=torch.int64).random_()
        torch.randn(2176, 2176, device=device)
        torch.randperm(84675)
        torch.randperm(84668)
    training_rng = (torch.get_rng_state(),
                    torch.cuda.get_rng_state(device) if device is not None and torch.device(device).type == 'cuda' else None)
    alternate = SEEDS[args.initialization - 1]
    if alternate is not None:
        devices = [torch.device(device)] if device is not None and torch.device(device).type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(alternate)
            alternate_model = factory()
            if family not in ('top', 'sage', 'saint'):
                alternate_model.reset_parameters()
            model.load_state_dict(alternate_model.state_dict(), strict=True)
        del alternate_model
    return model, training_rng


def restore(rng, device):
    torch.set_rng_state(rng[0])
    if rng[1] is not None:
        torch.cuda.set_rng_state(rng[1], device)


def load_initial(model, args):
    path = args.data_dir / 'initial' / args.config / (str(args.initialization) + '.pt')
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    return state['rng']


def iglu_tensors(seed):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        values = {}
        for name, shape in [('Q', (128, 256)), ('R', (256, 256)), ('w', (256, 40))]:
            limit = np.sqrt(6 / sum(shape))
            values[name] = tf.Variable(tf.random_uniform(shape, minval=-limit, maxval=limit, dtype=tf.float32), name=name)
        for name, shape, value in [('b1', (256,), 0), ('b2', (256,), 0), ('b3', (40,), 0),
                                   ('offset', (1, 256), 0), ('scale', (1, 256), 1),
                                   ('Layer2/offset', (1, 256), 0), ('Layer2/scale', (1, 256), 1)]:
            values[name] = tf.Variable(tf.fill(shape, float(value)), name=name)
        init = tf.global_variables_initializer()
    options = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=graph, config=options) as session:
        session.run(init)
        arrays = session.run(values)
    mapping = {'Q': 'R', 'R': 'R_1', 'w': 'w', 'b1': 'b1', 'b2': 'b2', 'b3': 'b3',
               'offset': 'offset', 'scale': 'scale', 'Layer2/offset': 'Layer2/offset', 'Layer2/scale': 'Layer2/scale'}
    return {mapping[name]: value for name, value in arrays.items()}
