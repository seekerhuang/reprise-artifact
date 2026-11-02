import math

import dgl
import torch

from baselines.common import Output, metrics
from baselines.data import dgl_graph, initial, read
from baselines.pfnc.model import CommonSAGE, PFNCStep


def run(args, recipe):
    device = torch.device(args.device)
    output = Output(args, recipe)
    with output.phase('input'):
        data = read(args, 'products')
        graph = dgl_graph(data, self_loops=True)
        graph.ndata['feat'], graph.ndata['label'] = graph.ndata['features'], graph.ndata['labels']
        roots = data['train_mask'].nonzero().flatten().to(device)
    with output.phase('model_and_cache'):
        saved = initial(args)
        model = CommonSAGE(data['x'].shape[1], recipe['hidden'], int(data['y'].max()) + 1,
                           dropout=recipe['dropout']).to(device)
        model.load_fresh_state(saved['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
        engine = PFNCStep(model, optimizer, len(data['x']), math.floor(recipe['cache_fraction'] * len(data['x'])),
                          beta=recipe['beta'], alpha=recipe['alpha'],
                          queue_size=min(recipe['queue_size'], math.ceil(len(roots) / recipe['batch_size'])))
        from main.initialization import restore
        restore(saved['rng'], device)
    with output.phase('sampler'):
        dgl.seed(recipe['seed'])
        sampler = dgl.dataloading.NeighborSampler(recipe['fanouts'], prefetch_node_feats=['feat'],
                                                 prefetch_labels=['label'])
        loader = dgl.dataloading.DataLoader(graph, roots, sampler, device=device,
                                            batch_size=recipe['batch_size'], shuffle=True, drop_last=False,
                                            num_workers=0, use_uva=device.type == 'cuda')

    @torch.no_grad()
    def evaluate(candidate):
        torch.backends.cuda.matmul.allow_tf32 = False
        candidate.eval()
        logits = candidate.full_forward(graph.to(device), data['x'].to(device)).cpu()
        return metrics(logits, data)

    for epoch in range(recipe['epochs']):
        with output.phase('training'):
            torch.backends.cuda.matmul.allow_tf32 = True
            model.train()
            for _, roots, blocks in loader:
                engine.step(blocks, blocks[0].srcdata['feat'], roots, blocks[-1].dstdata['label'])
        with output.phase('evaluation'):
            valid, test = evaluate(model)
        output.observe(model, epoch + 1, engine.updates, valid, test)
    if engine.updates != recipe['epochs'] * len(loader):
        raise RuntimeError('PFNC update count differs')
    output.finish(model, evaluate, optimizer, work=engine.stats())
