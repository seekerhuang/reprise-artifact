import argparse
from pathlib import Path

import torch

from empirical.data.datasets import DATASETS, download, partition


PARTS = {'arxiv': 80, 'reddit2': 200, 'flickr': 24, 'ppi': 20}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        path = args.data_dir / (name + '.pt')
        if path.exists():
            data = torch.load(path, map_location='cpu', weights_only=True)
        else:
            data = download(name, args.data_dir)
            torch.save(data, path)
        if name in PARTS:
            target = args.data_dir / (name + '-partition.pt')
            if name != 'arxiv' and not target.exists():
                source = data['train'] if name == 'ppi' else data
                torch.save(partition(source, PARTS[name]), target)
            if name in ('arxiv', 'flickr'):
                target = args.data_dir / (name + '-lmc-partition.pt')
                if not target.exists():
                    source = (partition(data, PARTS[name], symmetric=True) if name == 'arxiv'
                              else torch.load(args.data_dir / 'flickr-partition.pt', weights_only=True))
                    counts = source['ptr'][1:] - source['ptr'][:-1]
                    cluster = torch.empty(len(data['x']), dtype=torch.long)
                    cluster[source['permutation']] = torch.repeat_interleave(torch.arange(PARTS[name]), counts)
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(2027 if name == 'arxiv' else 123)
                        cluster = torch.randperm(PARTS[name])[cluster]
                    cluster, permutation = cluster.sort()
                    ptr = torch.ops.torch_sparse.ind2ptr(cluster, PARTS[name])
                    torch.save(dict(permutation=permutation, ptr=ptr), target)
                saved = torch.load(target, weights_only=True)
                from empirical.lmc.lmc_sequence import generate
                torch.save(generate(name, saved['ptr']), args.data_dir / (name + '-lmc-sequence.pt'))
        if name == 'reddit2':
            from torch_geometric.data import download_google_url
            directory = args.data_dir / 'raw/reddit2/raw'
            directory.mkdir(parents=True, exist_ok=True)
            files = {'adj_full.npz': '1sncK996BM5lpuDf75lDFqCiDZyErc1c2',
                     'feats.npy': '1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3',
                     'class_map.json': '1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u',
                     'role.json': '1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A',
                     'adj_train.npz': '1tRRfrUJAc3ZtPZLJ91fWv5KlKTYygaf7'}
            for filename, file_id in files.items():
                if not (directory / filename).exists():
                    download_google_url(file_id, str(directory), filename)
        print(name, flush=True)


if __name__ == '__main__':
    main()
