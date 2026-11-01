def split_scores(logits, labels, indices, multilabel, thresholds=(0.5,)):
    import torch
    import torch.nn.functional as F
    ids = torch.as_tensor(indices, dtype=torch.long, device=logits.device)
    prediction = logits.index_select(0, ids)
    target = labels.index_select(0, ids)
    if multilabel:
        target = target.to(prediction.dtype)
        truth = target > 0.5
        probabilities = prediction.sigmoid()
        scores = {}
        for threshold in thresholds:
            positive = probabilities > threshold
            tp = int((positive & truth).sum())
            fp = int((positive & ~truth).sum())
            fn = int((~positive & truth).sum())
            denom = 2 * tp + fp + fn
            scores[str(threshold)] = dict(score=2 * tp / denom if denom else 0.0, TP=tp, FP=fp, FN=fn)
        return dict(loss=float(F.binary_cross_entropy_with_logits(prediction, target)), scores=scores, nodes=ids.numel(), classes=prediction.shape[1])
    target = target.argmax(1) if target.ndim == 2 else target.long()
    correct = int((prediction.argmax(1) == target).sum())
    return dict(loss=float(F.cross_entropy(prediction, target)), score=correct / ids.numel(), correct=correct, nodes=ids.numel(), classes=prediction.shape[1])
