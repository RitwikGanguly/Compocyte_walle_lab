# Compocyte v2 — CPU / GPU run guide

How to set up the `Compocyte v2_cpu` and `Compocyte v2_gpu` environments from
the requirements files in this repo, then run pre-training (train from
scratch or load a pretrained model) and inference.

File map:

| File | Use on |
|---|---|
| `requirements-compocyte-cpu.txt` | CPU-only machines, or any machine where you want deterministic CPU training (`torch==2.11.0`, runs on CPU when CUDA 13 is not supported by the driver) |
| `requirements-compocyte-gpu.txt` | NVIDIA-GPU machines, driver supporting CUDA 11.8+ (`torch==2.5.1+cu118`, `torch.cuda.is_available()` must be `True`) |

Both files pin the same Compocyte dependency set from `pyproject.toml`
(`anndata`, `scanpy`, `scikit-learn`, `pooch`, `torch`, `catboost`,
`balanced-loss`, `cytopus`, `dask`, `leidenalg`, `pygraphviz`, `SEACells`).
Only the `torch` build (and its version-locked helpers) differs. No
requirements files existed before (only `docs/requirements.txt` for Sphinx),
so these two are new.

## 1. Prerequisites

- Python 3.12 (both reference envs use 3.12.14; `pyproject.toml` needs `>=3.12`).
- `conda` / `miniconda3` (recommended; the benchmark scripts use `conda run -n <env>`).
- System build tools for `pygraphviz`:
  ```bash
  conda install -c conda-forge gcc gxx graphviz pygraphviz
  # or on Debian/Ubuntu:
  # sudo apt-get install -y gcc g++ graphviz graphviz-dev
  ```
- For GPU only: NVIDIA driver supporting CUDA 11.8+ (check with `nvidia-smi`;
  reference machine: RTX A6000, driver 525.105.17).

## 2. Create the environment and install

### CPU (`Compocyte v2_cpu`)

```bash
git clone https://github.com/WALL-E-Lab/Compocyte.git
cd Compocyte

conda create -n compocyte-cpu python=3.12 -y
conda activate compocyte-cpu
conda install -c conda-forge gcc gxx graphviz pygraphviz

pip install -r requirements-compocyte-cpu.txt
pip install -e .   # installs THIS repo as the `Compocyte` package
```

Verify (expect `cuda_available=False`; CPU run is the intended mode here):

```bash
conda run -n compocyte-cpu python -c \
  "import Compocyte, torch; \
   print('compocyte OK'); \
   print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
```

### GPU (`Compocyte v2_gpu`)

```bash
git clone https://github.com/WALL-E-Lab/Compocyte.git
cd Compocyte

conda create -n compocyte-gpu python=3.12 -y
conda activate compocyte-gpu
conda install -c conda-forge gcc gxx graphviz pygraphviz

pip install -r requirements-compocyte-gpu.txt
pip install -e .   # installs THIS repo as the `Compocyte` package
```

Verify (expect `cuda_available=True`; the benchmark script asserts this):

```bash
conda run -n compocyte-gpu python -c \
  "import Compocyte, torch; \
   assert torch.cuda.is_available(), 'CUDA not visible - check driver/nvidia-smi'; \
   print('compocyte OK'); \
   print('torch', torch.__version__, 'cuda_available=True')"
```

## 3. Data format (both CPU and GPU)

- Input is an `AnnData` object with **raw counts** in `.X` (fallback: `.raw.X`
  or `.layers["raw"]`). `HierarchicalClassifier.load_adata()` raises
  `ValueError('No raw counts found ...')` if no counts are found; pass
  `ignore_counts=True` only for non-RNA / pre-reduced input.
- With default `default_input_data='normlog'`, counts are auto-normalized to
  10,000 per cell + `log1p` on load.
- A quick built-in test set (861 cells x 1011 genes, counts in `.X`, labels in
  `.obs['labels']`):
  ```python
  import Compocyte
  adata = Compocyte.data.sample_data()
  hierarchy = Compocyte.data.sample_hierarchy()
  print(adata, hierarchy)
  ```

## 4. Inference with a pretrained model (no training)

Pretrained PBMC / TIL classifiers download once from Zenodo via `pooch`
(~PBMC: `zenodo.org/records/19708295`, TIL: `zenodo.org/records/19707910`):

```python
import Compocyte

# Pick one:
hc = Compocyte.pretrained.pbmc_pretrained()   # root node 'Blood'
# hc = Compocyte.pretrained.til_pretrained()  # root node 'blood' (lowercase!)

adata = Compocyte.data.sample_data()  # replace with your own AnnData
hc.load_adata(adata)

# CPU:
hc.predict_all_child_nodes(hc.root_node)
# GPU (same call, just pin the device; batch_size bounds peak memory):
# hc.predict_all_child_nodes(hc.root_node, device='cuda', batch_size=8192)

print(hc.adata.obs.filter(like='_pred').head())
```

Notes:

- Prediction columns are written per level as `<Level_i>_pred` in
  `.adata.obs`. Use `device='cuda'` / `device='cpu'` explicitly if you want to
  override auto-detection (auto order: CUDA > MPS > CPU, see
  `DenseTorch.resolve_device`).
- `threshold=-1` (default) always assigns the top label; pass e.g.
  `threshold=0.9` to leave low-confidence cells empty (`''`).
- `hc.save()` / `hc.load()` persist / restore timestamped snapshots under
  `save_path` (see section 6).

## 5. Pre-training: train your own classifier from scratch

### 5.1 Minimal example (works on CPU and GPU)

```python
import Compocyte
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.tools import infer_levels

adata = Compocyte.data.sample_data()        # your labeled AnnData here
hierarchy = Compocyte.data.sample_hierarchy()

# Expand leaf labels into per-level columns; returns (obs_df, obs_names).
_, obs_names = infer_levels(
    hierarchy=hierarchy,
    labels='labels',        # adata.obs column with leaf labels
    root_node='T_cell',     # must be the hierarchy root key
    adata=adata,
)

hc = HierarchicalClassifier(
    save_path='./my_classifier',
    adata=adata,
    root_node='T_cell',
    dict_of_cell_relations=hierarchy,
    obs_names=obs_names,
    num_threads=1,
)

hc.train_all_child_nodes()          # sequential; simplest and lowest RAM
hc.save()                           # snapshot to ./my_classifier
```

To train with the fast sklearn backend at every node (as in the PBMC
tutorial `colab_tutorials/01_training_PBMC_classifier.ipynb`), add before
training:

```python
from Compocyte.core.models.log_reg import LogisticRegression
for node in hc.graph.nodes:
    hc.set_classifier_type(node, LogisticRegression)
```

Default local classifier is otherwise `DenseTorch` (2-layer `[64, 64]`
network, `dropout=0.4`).

### 5.2 CPU settings (mirrors `train_compocyte_cpu.py`)

```python
hc.train_all_child_nodes(parallelize=True, processes=4)
# equivalent low-level call used by the benchmark, with explicit fit kwargs:
# nodes = [n for n in hc.graph.nodes if len(list(hc.graph.successors(n))) >= 1]
# hc._train_nodes_in_pool(nodes, 4, fit_kwargs={
#     'epochs': 100, 'batch_size': 64,
#     'num_threads': 1, 'selection_max_cells': 100_000})
```

- `batch_size=64`, `epochs=100`, `selection_max_cells=100_000` are the
  benchmark defaults. `selection_max_cells` caps the feature-selection subset;
  per-node training itself caps at `max_cells` (default 1,000,000).
- `processes=4` with `num_threads=1` avoids BLAS oversubscription
  (`processes` is divided by `num_threads` internally).
- Memory is adaptively budgeted via env vars (see
  `fit_methods.train_resource_budget`):
  `COMPOCYTE_MAX_THREADS` (default 16), `COMPOCYTE_MAX_RAM_FRACTION`
  (default 0.5). Giant nodes automatically fall back to sequential training.
- The CPU torch build may print `CUDA initialization: The NVIDIA driver ...
  too old` once; it is harmless — training stays on CPU.

### 5.3 GPU settings (mirrors `train_compocyte_gpu.py`)

```python
import torch
assert torch.cuda.is_available()

hc.train_all_child_nodes(parallelize=True, processes=4)
# benchmark-equivalent fit kwargs: epochs=100, batch_size=1024,
# num_threads=1, selection_max_cells=100_000
```

- `batch_size=1024` (vs 64 on CPU) — the GPU default. Small node matrices
  (<= ~1 GB dense) are moved to the device once; larger ones stream via
  pinned host memory with fork-safe CPU loader workers.
- **Fork rule (hard failure if violated):** never touch CUDA in the parent
  process before the training pool is created — no `torch.cuda.*` calls,
  no `model.to('cuda')` before `train_all_child_nodes(parallelize=True, ...)`.
  Violations die with `RuntimeError: Cannot re-initialize CUDA in forked
  subprocess`. The benchmark guards this explicitly.
- Inference on GPU: `hc.predict_all_child_nodes(hc.root_node, device='cuda')`.

## 6. Save / load / reuse a trained classifier

```python
hc.save()  # creates ./my_classifier/hierarchical_classifiers/<timestamp>/ ...

from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
hc2 = HierarchicalClassifier(save_path='./my_classifier')
hc2.load()                    # restores latest snapshot (models + node content)
hc2.load_adata(new_adata)     # new cells; genes auto-matched, missing genes zero-filled
hc2.predict_all_child_nodes(hc2.root_node)
```

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: DenseTorch requires the 'torch' extra` | You installed base deps only; `pip install -r requirements-compocyte-{cpu,gpu}.txt` includes torch. |
| `ValueError: No raw counts found ...` | Put raw counts in `.X` (or `.raw.X` / `.layers["raw"]`), or construct with `ignore_counts=True` for non-count input. |
| `Cannot re-initialize CUDA in forked subprocess` (GPU) | Remove any `torch.cuda.*` call before training; let the pool own CUDA init. |
| `torch.cuda.is_available()` False in `compocyte-gpu` env | Driver too old / no GPU visible; check `nvidia-smi`, need driver >= ~520 for cu118. |
| `pygraphviz` build fails | Install system graphviz first (section 1), then `conda install -c conda-forge pygraphviz`. |
| Out of memory on huge atlases | Lower `batch_size`, lower `selection_max_cells` / `max_cells`, reduce `processes`, or set `COMPOCYTE_MAX_RAM_FRACTION=0.3`. |

## 8. Reference commands used in this repo

- CPU single-subset run: `conda run -n compo python train_compocyte_cpu.py --subset <name> --epochs 100 --out-dir results/cpu` (`comparison/resource_comparison/`).
- GPU single-subset run: `conda run -n compo-gpu python train_compocyte_gpu.py --subset <name> --epochs 100 --out-dir results/gpu`.
- Full benchmark harnesses: `comparison/resource_comparison/run_benchmark.sh` (CPU), `gpu_benchmark.sh` (GPU), `run_benchmark_5fold.sh` (5-fold CPU).
