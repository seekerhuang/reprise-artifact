import torch

def weights_from_train_labels(training_labels, classes, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('predeclared inverse-frequency exponent must be in [0,1]')
    counts = torch.bincount(training_labels, minlength=classes)
    if len(counts) != classes or not training_labels.numel():
        raise ValueError('Unexpected supervised class domain')
    weight = counts.clamp_min(1).to(torch.float32).pow(-alpha)
    weight *= counts.sum() / (counts * weight).sum()
    return (weight, counts)
