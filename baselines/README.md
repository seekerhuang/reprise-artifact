# Baselines

We use the authors' official implementations where code is available; where code is unavailable, we reproduce and adapt the required components following the details in the original papers.

REST is reconstructed on GAS. PFNC–Products uses our reconstructed embedding cache and the released PFN gradient component.

After the setup in the [main README](../README.md), run from the repository root:

```bash
python -m baselines.prepare --data-dir /path/to/data
python -m baselines.run --method rest --config products-GCN --data-dir /path/to/data --initialization 1 --out /path/to/output
python -m baselines.run --method rest --config Reddit2-GCN --data-dir /path/to/data --initialization 1 --out /path/to/output
python -m baselines.run --method pfnc --config products-SAGE --data-dir /path/to/data --initialization 1 --out /path/to/output
```

`--initialization` accepts `1`, `2`, or `3`. Use a new output directory for each run.

| Reconstruction | Implementation | Upstream |
|---|---|---|
| REST–Products, REST–Reddit2 | [rest/](rest/) | [REST](https://github.com/RXPHD/REST), [GAS](https://github.com/rusty1s/pyg_autoscale/tree/be65e922bd613861bc2c4d31ed8439ef53a0e581) |
| PFNC–Products | [pfnc/](pfnc/) | [PFNC](https://github.com/qzhou77/PFNC-artifact/tree/54b1f52685b7aa566e41167565edc94f82b3ed89) |

The following files contain models; training implementations remain in the upstream repositories.

| Comparison | Model | Model code | Upstream |
|---|---|---|---|
| LABOR–Products | Three-layer SAGE, hidden 256, BatchNorm, root branch | [model.py](labor/products/model.py) | [LABOR](https://github.com/dmlc/dgl/tree/f971e25a4dff33d5d219dccf523e32c62360ffd2/examples/pytorch/labor) |
| LABOR–Flickr | Two-layer GCN, hidden 256, first-layer BatchNorm, input and hidden dropout | [model.py](labor/flickr/model.py) | [LABOR](https://github.com/dmlc/dgl/tree/f971e25a4dff33d5d219dccf523e32c62360ffd2/examples/pytorch/labor) |
| LABOR–Yelp | Two-layer GCN, hidden 512, hidden dropout | [model.py](labor/yelp/model.py) | [LABOR](https://github.com/dmlc/dgl/tree/f971e25a4dff33d5d219dccf523e32c62360ffd2/examples/pytorch/labor) |
| GraphSAINT–Flickr | Two-layer GCN, hidden 256, first-layer BatchNorm, input and hidden dropout | [model.py](graphsaint/flickr/model.py) | [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT/tree/c9b1e340d7b951465ac4a9251eef93832e68b003) |
| GraphSAINT–PPI | Two-layer GCN, hidden 1024, BatchNorm, residuals, input/output affine layers | [model.py](graphsaint/ppi/model.py) | [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT/tree/c9b1e340d7b951465ac4a9251eef93832e68b003) |
| VR-GCN–Reddit2 | Two-layer GCN, hidden 256, bias, hidden dropout, no LayerNorm or concatenation | [model.py](vrgcn/model.py) | [VR-GCN](https://github.com/thu-ml/stochastic_gcn/tree/6ca520621b501032b526a7de166b86910260957f) |
| Cluster-GCN–Reddit2 | Two-layer GCN, hidden 256, bias, hidden dropout, no LayerNorm or concatenation | [model.py](clustergcn/model.py) | [Cluster-GCN](https://github.com/google-research/google-research/tree/0c1bbe5/cluster_gcn) |

GCN models use symmetric normalization with one self-loop per node; GraphSAINT's corresponding operator is in [normalization.py](graphsaint/normalization.py).

Other comparisons use the released implementations with the paper's model and configuration settings:

| Method | Official repository |
|---|---|
| TOP | [MIRALab-USTC/TOP](https://github.com/MIRALab-USTC/TOP) |
| IGLU | [sdeepaknarayanan/iglu](https://github.com/sdeepaknarayanan/iglu) |
| LMC | [MIRALab-USTC/GNN-LMC](https://github.com/MIRALab-USTC/GNN-LMC) |
| FreshGNN | [xxcclong/history-cache](https://github.com/xxcclong/history-cache) |
| GAS | [rusty1s/pyg_autoscale](https://github.com/rusty1s/pyg_autoscale) |
| GraphFM | [divelab/DIG](https://github.com/divelab/DIG/tree/21476b079c9226f38915dcd082b5c2ee0cddaac8/examples/lsgraph/GraphFMOB) |
| PFNC | [qzhou77/PFNC-artifact](https://github.com/qzhou77/PFNC-artifact) |
| LABOR | [dmlc/dgl](https://github.com/dmlc/dgl/tree/f971e25a4dff33d5d219dccf523e32c62360ffd2/examples/pytorch/labor) |
| GraphSAINT | [GraphSAINT/GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) |

Products–GAS, Products–GraphFM, and Products–TOP use a three-layer GCN with hidden size 256 and ordinary BatchNorm; the Products–TOP configuration is reconstructed from the released implementation and paper settings.
