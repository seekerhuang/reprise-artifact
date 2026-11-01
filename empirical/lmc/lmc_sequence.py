import torch
from torch.utils.data import DataLoader

from main.models import DoubleBNGCN


def generate(dataset, ptr):
    arxiv = dataset == 'arxiv'
    batch_parts = 40 if arxiv else 12
    epochs = 300 if arxiv else 400
    keys = [0, int(ptr[batch_parts])]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        if arxiv:
            torch.randperm(80)
        prepared = list(DataLoader(keys, batch_size=1, shuffle=True, num_workers=0))
        evaluation = DataLoader(range(4), batch_size=1, shuffle=False, num_workers=0)
        training = DataLoader(prepared, batch_size=1, shuffle=True, num_workers=0)
        list(evaluation)
        list(evaluation)
        list(training)
        DoubleBNGCN(128 if arxiv else 500, 40 if arxiv else 7,
                    3 if arxiv else 2, 0.5 if arxiv else 0.3, not arxiv)
        list(evaluation)
        sequence = []
        for _ in range(epochs):
            sequence.extend(int(value) for value in training)
            list(evaluation)
    return torch.tensor(sequence)
