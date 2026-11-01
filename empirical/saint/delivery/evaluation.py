from empirical.saint.delivery.policy import split_scores

def quality_from_logits(logits, labels, train_ids, valid_ids, test_ids, multilabel, thresholds=(0.5,)):
    train = split_scores(logits, labels, train_ids, multilabel, (0.5,))
    valid = split_scores(logits, labels, valid_ids, multilabel, thresholds)
    test = split_scores(logits, labels, test_ids, multilabel, thresholds)
    result = dict(train_fit=train['loss'], valid_loss=valid['loss'], test_loss=test['loss'], train_metrics=train, valid_metrics=valid, test_metrics=test, validation_metrics_computed=True)
    for (name, score) in (('valid', valid), ('test', test)):
        result[name + '_score'] = score['scores']['0.5']['score'] if multilabel else score['score']
        if multilabel:
            for (value, suffix) in ((0.45, '045'), (0.55, '055')):
                if value in thresholds:
                    result[name + '_score_' + suffix] = score['scores'][str(value)]['score']
    return result

def saint_evaluate(model, minibatch, thresholds=(0.5,)):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        (nodes, matrix, _unused_full_loss_weights) = minibatch.one_batch('test')
        (_, hidden) = model.conv_layers((matrix, model.feat_full[nodes]))
        logits = model.classifier((None, F.normalize(hidden, p=2, dim=1)))[1]
        return quality_from_logits(logits, model.label_full, minibatch.node_train, minibatch.node_val, minibatch.node_test, model.sigmoid_loss, thresholds)
