from pathlib import Path
import os

import torch


def describe(args, recipe, model, optimizer):
    root = Path(args.data_dir).resolve()
    initial = root / 'initial' / args.config / (str(args.initialization) + '.pt')
    family, dataset = recipe['family'], recipe['dataset']
    if family == 'saint':
        files = [root / 'raw' / dataset / 'raw' / name
                 for name in ('adj_full.npz', 'adj_train.npz', 'feats.npy', 'role.json', 'class_map.json')]
    else:
        files = [root / (dataset + '.pt')]
    if family == 'lmc':
        files += [root / (dataset + suffix) for suffix in ('-lmc-partition.pt', '-lmc-sequence.pt')]
    elif family in ('history', 'ppi', 'pna') and dataset != 'yelp':
        files.append(root / (dataset + '-partition.pt'))
    names = {id(parameter): name for name, parameter in model.named_parameters()}

    def value(item):
        if torch.is_tensor(item):
            return item.detach().cpu().tolist()
        if isinstance(item, (tuple, list)):
            return [value(part) for part in item]
        return item

    optimizers = optimizer if isinstance(optimizer, dict) else {'model': optimizer}
    groups = {name: [dict(parameters=[names[id(parameter)] for parameter in group['params']],
                          **{key: value(item) for key, item in group.items() if key != 'params'})
                     for group in opt.param_groups] for name, opt in optimizers.items()}
    device = torch.device(args.device)
    environment = dict(pytorch=torch.__version__, cuda=torch.version.cuda,
                       cpu_affinity=sorted(os.sched_getaffinity(0)),
                       torch_threads=torch.get_num_threads(),
                       torch_interop_threads=torch.get_num_interop_threads(),
                       omp_threads=os.environ.get('OMP_NUM_THREADS'),
                       mkl_threads=os.environ.get('MKL_NUM_THREADS'),
                       openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    if device.type == 'cuda':
        gpu = torch.cuda.get_device_properties(device)
        environment.update(gpu=gpu.name, gpu_uuid=str(gpu.uuid))
    seeds = dict(python=123, numpy=123, torch_rng_file=str(initial))
    if family == 'saint':
        seeds = dict(training=recipe.get('training_seed', 2026092901),
                     sampling=recipe.get('sampler_seed', 123))
    elif family == 'sage':
        seeds['dgl'] = 123
    elif family == 'iglu':
        seeds['batch_order'] = recipe['training_seed']
    return dict(config=args.config, method=getattr(args, 'method', 'reprise'),
                initialization=args.initialization, initial_weights=str(initial),
                seeds=seeds, recipe=recipe, optimizer=groups,
                precision=dict(parameter_dtypes=sorted({str(parameter.dtype) for parameter in model.parameters()}),
                               training_tf32=family in ('sage', 'iglu'), evaluation_tf32=family == 'iglu'),
                environment=environment, inputs=[str(path) for path in files])
