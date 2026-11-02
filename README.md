# REPRISE

Linux x86-64, Python 3.11, CUDA 12.8 toolkit, C/C++ compiler with OpenMP, and an NVIDIA A800 80 GB GPU.

```bash
python -m venv .venv
source .venv/bin/activate
export CUDA_HOME=/path/to/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
pip install -r requirements.txt
python build_extensions.py
python prepare_data.py --data-dir /path/to/data
python prepare_initial.py --data-dir /path/to/data
```

Run from this directory:

```bash
python run.py --config arxiv-TOP --data-dir /path/to/data --initialization 1 --out /path/to/output
```

`--initialization` accepts `1`, `2`, or `3`. Use a new output directory for each run.

Runs use four allowed CPU cores by default; `--cpu-cores` selects specific cores.

Available configurations:

```text
arxiv-TOP       arxiv-IGLU      arxiv-LMC       arxiv-Fresh
products-GCN    products-SAGE
Reddit2-GCN     Reddit2-SAINT
Flickr-GCN      Flickr-LMC     Flickr-PNA
Yelp-GCN        Yelp-SAINT
PPI-GCN
```

`prepare_data.py --datasets arxiv products` prepares selected datasets. `prepare_initial.py --configs arxiv-TOP arxiv-IGLU` prepares selected configurations.

Run tensor tests with synthetic inputs:

```bash
python tests/tensors/run.py
python tests/tensors/run.py --configs Reddit2-SAINT Yelp-SAINT
```

Run independent gradient checks for Algorithm 1 and Equations (7)–(8):

```bash
python tests/core/run.py
python tests/core/run.py --configs Flickr-PNA Reddit2-SAINT
```

See [baselines/README.md](baselines/README.md) for baseline implementations, models, and upstream repositories.

Directory structure:

```text
main/
  top/                 arxiv-TOP
  iglu/                arxiv-IGLU
  products_gcn/        products-GCN
  sage/                arxiv-Fresh, products-SAGE
  native/              DGL adapter
empirical/
  data/                Dataset loading
  gas/
    reddit/            Reddit2-GCN
    flickr/            Flickr-GCN
    yelp/              Yelp-GCN
    ppi/               PPI-GCN
  lmc/
    arxiv/             arxiv-LMC
    flickr/            Flickr-LMC
  pna/                 Flickr-PNA
  saint/
    core/              REPRISE training
    head/              Head execution
    optim/             Optimizer execution
    input/             Input preparation
    sparse/            Sparse operators
    delivery/          Evaluation and parameter averaging
    schedules/         Training schedules
  vendor/
    gas/               Required model and history components
    graphsaint/        Required sampling components
baselines/
  rest/                Reconstructed REST
  pfnc/                Reconstructed PFNC–Products
  labor/               LABOR models
  graphsaint/          GraphSAINT models
  vrgcn/               VR-GCN model
  clustergcn/          Cluster-GCN model
  vendor/pfnc/         Required PFN gradient component
tests/
  core/
    run.py             Configuration-to-adapter map and entry point
    reference.py       Independent gradient and projection references
    graph_adapters.py  TOP, products-GCN, SAGE
    history_adapters.py  GAS and PNA
    iglu_lmc.py        IGLU and LMC
    saint.py           GraphSAINT
  tensors/             Tensor checks for all configurations
appendix/
  appendix-proofs.pdf  Proofs for the paper
```
