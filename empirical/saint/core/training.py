from types import SimpleNamespace
from empirical.saint.core import base_training as base
from empirical.saint.core.saint_onepass_dropout_v1 import OnePassDropout as BlockDropout

METHODS = ("lateA",)

class Training(base.Training):

    def __init__(self, model, method):
        import torch
        if method not in METHODS:
            raise ValueError(method)
        if model.order_layer not in ([1, 1, 0], [1, 0, 1, 0]) or any((layer.bias != 'norm' for layer in model.aggregators)):
            raise ValueError('Locked original SAINT feature architectures required')
        if list(model.named_buffers()):
            raise ValueError('These feature-axis normalization recipes have no running BN buffers')
        if model.classifier.order != 0 or model.classifier.bias != 'bias' or model.classifier.aggr != 'mean':
            raise ValueError('Original final affine classifier required')
        (self.model, self.method) = (model, method)
        named = dict(model.named_parameters())
        pids = {id(p) for layer in model.aggregators[:-1] for p in layer.parameters()}
        lids = {id(p) for p in model.aggregators[-1].parameters()}
        cids = {id(model.classifier.f_lin[0].weight), id(model.classifier.f_bias[0])}
        unused_ids = {id(p) for p in model.classifier.params[1:]}
        if any((a & b for (a, b) in ((pids, lids), (pids, cids), (lids, cids), (pids | lids | cids, unused_ids)))):
            raise ValueError('Clip groups must be disjoint actual parameter identities')
        self.groups = {label: {k: p for (k, p) in named.items() if id(p) in ids} for (label, ids) in (('P', pids), ('L', lids), ('C', cids))}
        self.unused = {k: p for (k, p) in named.items() if id(p) in unused_ids}
        expected_p = sum((4 * (layer.order + 1) for layer in model.aggregators[:-1]))
        expected_l = 4 * (model.aggregators[-1].order + 1)
        if [len(v) for v in self.groups.values()] != [expected_p, expected_l, 2] or len(self.unused) != 2:
            raise ValueError('Actual module parameter support differs from original norm/affine recipe')
        all_ids = pids | lids | cids | unused_ids
        if all_ids != {id(p) for p in named.values()} or all_ids != {id(p) for g in model.optimizer.param_groups for p in g['params']}:
            raise ValueError('Model/Adam unique parameter references differ')
        (self.body_labels, self.head_labels) = (('P', 'L'), ('C',))
        self.body = {k: p for label in self.body_labels for (k, p) in self.groups[label].items()}
        self.head = dict(self.groups['C'])
        self.active_parameters = {k: p for (k, p) in named.items() if k not in self.unused}
        self.randomness = BlockDropout(model)
        self.blocks = self.visits = self.optimizer_calls = 0
        self.prefix_forwards = self.body_vjps = self.head_queries = 0
        self.local_forwards = self.local_vjps = 0
        self.loss_node_presentations = self.prefix_node_presentations = 0
        self.cache_peak_payload_bytes = 0
        self.active_cache = False
        self.closed = False
        self.clip_calls = {label: 0 for label in self.groups}
        self.clip_active = None
