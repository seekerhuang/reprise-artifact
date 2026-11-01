from types import SimpleNamespace

import torch
import torch.nn.functional as F

from reference import check_block


def affine_case(model, names, zero):
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            parameter = named[name]
            if parameter.ndim == 1:
                scale = name.endswith('.weight') or name.endswith('.params.2')
                parameter.copy_(torch.linspace(.7 if scale else -.2, 1.3 if scale else .3,
                                               parameter.numel(), device=parameter.device))
            if zero:
                parameter[0].zero_()


def gcn(name, device):
    from main.native.dgl_adapter import configure
    configure()
    from main.models import TOPGCN, GCN
    from main.top.arxiv_coupled_structural_r64_v1 import CoupledStructural
    from main.products_gcn.products_equal_body_small_head_graph_v1 import EqualBodySmallSecondStructural
    from main.top.top_graph_operator import ExactPropagation
    top = name == 'arxiv-TOP'
    n, inputs, classes = (19, 128, 40) if top else (24, 100, 47)
    dense = torch.eye(n, device=device)
    ids = torch.arange(n, device=device)
    dense[ids, (ids + 1) % n] = .7
    dense[ids, (ids + 4) % n] = .2
    dense = dense + dense.T
    degree = dense.sum(1).rsqrt()
    matrix = (degree[:, None] * dense * degree[None, :]).to_sparse_csr()
    features = torch.randn(n, inputs, device=device)
    labels = ids % classes
    training = ids[::2]
    layout = ([('bns.0.weight', 0, 0, 1), ('bns.0.bias', 0, 0, 1),
               ('convs.1.weight', 0, 0, -1), ('bns.1.weight', 0, 0, 1), ('bns.1.bias', 0, 0, 1)]
              if top else [('bns.1.weight', 0, 0, 1), ('bns.1.bias', 0, 0, 1)])
    head_names = {'convs.2.weight', 'convs.2.bias'} if top else {'convs.2.lin.weight', 'convs.2.bias'}

    def build():
        model = (TOPGCN(inputs, 256, classes, .3) if top else GCN(inputs, 256, classes, 3, .7)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.005 if top else .0035,
                                     **({} if top else {'eps': 1e-6}))
        engine = (CoupledStructural(model, matrix, features, training_ids=training) if top else
                  EqualBodySmallSecondStructural(model, optimizer, matrix, features, training, labels))
        return engine, optimizer

    def objective(engine, boundary):
        head = engine.model.convs[-1]
        if top:
            logits = ExactPropagation.apply(boundary @ head.weight, engine.matrix, engine.transpose) + head.bias
            return F.cross_entropy(logits[training], labels[training])
        return F.cross_entropy(F.linear(boundary, head.lin.weight, head.bias), labels[training])

    for zero in (False, True):
        engine, optimizer = build()
        affine_case(engine.model, [row[0] for row in layout], zero)
        for _ in range(1 if zero else 2):
            engine.model.train()
            check_block(engine, optimizer, build, lambda e: e.prefix(), objective,
                        lambda e: e.update_block(optimizer, labels[training], 'A') if top else e.update_block(),
                        head_names, layout, atol=6e-5, rtol=8e-4)
        if not top:
            engine.recovery_graph.close()


def sage(name, device):
    from main.native.dgl_adapter import configure
    configure()
    import dgl
    from main.sage.sage_model import HistorySAGE
    from main.sage.fresh_arxiv_structural import FreshArxivStructural
    from main.sage.sage_exact_head_graph_v1 import ExactHeadSAGE
    fresh = name == 'arxiv-Fresh'
    n, inputs, classes, roots = (31, 128, 40, 7) if fresh else (640, 100, 47, 615)
    nodes = torch.arange(n, device=device)
    src = torch.stack([nodes, (nodes + 1) % n, (nodes + 7) % n, (nodes + 13) % n, (nodes + 18) % n], 1).reshape(-1)
    dst = nodes.repeat_interleave(5)
    graph = dgl.graph((src, dst), num_nodes=n).int().formats('csc')
    features = torch.randn(n, inputs, device=device)
    labels = nodes % classes
    first_image = features[src.long()].reshape(n, 5, inputs).mean(1)
    weights = None if fresh else torch.linspace(.7, 1.7, classes, device=device)
    head_names = {'convs.2.lin_l.weight', 'convs.2.lin_l.bias', 'convs.2.lin_r.weight'}
    layout = [('bns.1.weight', 0, 0, 1), ('bns.1.bias', 0, 0, 1)]

    def build():
        model = HistorySAGE(inputs, 256, classes, 3, .1 if fresh else .35,
                            'CSR', {}, SimpleNamespace(method='grad')).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001 if fresh else .0011,
                                     betas=(.9 if fresh else .8, .999), **({} if fresh else {'fused': True}))
        if fresh:
            engine = FreshArxivStructural(model, optimizer, dgl, graph, features, first_image, labels,
                                           sampler='neighbor', fanouts=(10, 10))
        else:
            engine = ExactHeadSAGE(model, optimizer, dgl, graph, features, labels,
                                    first_fanout=10, fanouts=(10, 10), sampler='labor')
            engine.sampler.set_seed(123)
        engine.loss_weights = weights
        return engine, optimizer

    for zero in (False, True):
        engine, optimizer = build()
        affine_case(engine.model, [row[0] for row in layout], zero)
        for _ in range(1 if zero else 2):
            engine.model.train()
            engine.set_context(nodes[:roots].to(graph.idtype))

            def reference():
                current, native = build()
                current.context, current.labels = engine.context, engine.labels
                return current, native

            check_block(engine, optimizer, reference, lambda e: e.prefix(),
                        lambda e, ports: F.cross_entropy(e.head(ports), e.labels, weight=weights),
                        lambda e: e.update_block(2), head_names, layout, atol=8e-5, rtol=2e-3)
        if not fresh:
            engine._exact_head.close()


def check(name, device):
    (sage if name in ('arxiv-Fresh', 'products-SAGE') else gcn)(name, device)
