#!/usr/bin/env python3
"""Benchmark GPU CellTypist training/inference time and memory.

Usage:
    python train_celltypist_gpu.py --subset oelen_e1_500k

CellTypist's GPU support relies on two RAPIDS packages:
  * `cuml`            - trains the logistic-regression classifier on the GPU
  * `rapids_singlecell` - runs the over-clustering step of majority voting on
                          the GPU (optional; without it CellTypist falls back
                          to scanpy's CPU neighbour graph, and only the
                          over-clustering is affected)

Both live in the dedicated `ct-gpu` conda environment, because the `scrna`
environment's scientific stack is pip-installed and its torch pins
nvidia-*-cu12==12.8 -- adding conda's RAPIDS packages there would downgrade
torch's CUDA libraries and break scANVI / scPoli / treeArches.
See `setup_celltypist_gpu_env.sh` for the exact creation commands.

Run this script with the `ct-gpu` environment, e.g.:
    conda run -n ct-gpu python train_celltypist_gpu.py --subset oelen_e1_500k
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import celltypist
import numpy as np
import scanpy as sc

from shared import PeakMemorySampler, load_subset, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "gpu")


def _patch_celltypist_gpu_overcluster():
    """Work around a celltypist 1.7.1 bug in the GPU majority-voting path.

    ``Classifier._construct_neighbor_graph`` moves the AnnData to the GPU and
    then calls *scanpy's* ``sc.pp.filter_genes`` on the GPU-backed object:

        if use_GPU:
            fsc.get.anndata_to_GPU(adata)     # X becomes a cupy array
        if 'highly_variable' not in adata.var:
            sc.pp.filter_genes(adata, min_cells=5)   # <-- scanpy on GPU data
            fsc.pp.highly_variable_genes(adata, n_top_genes=min([2500, adata.n_vars]))

    which raises::

        TypeError: Implicit conversion to a NumPy array is not allowed.
        Please use `.get()` to construct a NumPy array explicitly.

    The fix is to run the cheap gene filtering / HVG selection on the CPU
    *before* the GPU transfer, using exactly the same scanpy calls the CPU
    code path uses (so the CPU and GPU runs stay numerically comparable), and
    then let rapids-singlecell do the expensive part -- scaling, PCA, the KNN
    graph and Leiden -- on the GPU.
    """
    from celltypist.classifier import Classifier

    original = Classifier._construct_neighbor_graph

    def patched(adata, use_GPU=False):
        if (
            use_GPU
            and "highly_variable" not in adata.var
            and getattr(adata, "raw", None) is None
        ):
            sc.pp.filter_genes(adata, min_cells=5)
            sc.pp.highly_variable_genes(
                adata, n_top_genes=min([2500, adata.n_vars])
            )
        return original(adata, use_GPU)

    Classifier._construct_neighbor_graph = staticmethod(patched)


def _require_gpu_deps():
    """Check the RAPIDS packages CellTypist needs for GPU execution.

    ``cuml`` is mandatory: ``celltypist.train(..., use_GPU=True)`` returns
    None when it is missing, which would crash the run later on.

    ``rapids_singlecell`` is optional: it only accelerates the
    over-clustering step of majority voting.  Without it CellTypist falls back
    to scanpy's CPU neighbour graph, and the logistic-regression training is
    still performed on the GPU.
    """
    if importlib.util.find_spec("cuml") is None:
        raise RuntimeError(
            "CellTypist GPU benchmark requires the RAPIDS package 'cuml', "
            "which is not installed in this environment.\n"
            "Create the dedicated 'ct-gpu' environment, for example:\n"
            "    conda create -n ct-gpu -y -c rapidsai -c nvidia -c conda-forge "
            "cuml=24.08 cudf=24.08 cugraph=24.08 'cupy>=12.0' rmm=24.08 "
            "python=3.10 cuda-version=12.5 'nccl>=2.9.9,<3.0a0' "
            "'nvtx>=0.2.1,<0.3' scanpy=1.10.3 anndata=0.10.8 numpy=1.26.4 "
            "scipy=1.13.1 scikit-learn=1.6.1 natsort scikit-misc statsmodels "
            "scikit-image h5py loompy pyyaml requests tqdm matplotlib "
            "python-igraph leidenalg openpyxl click pip\n"
            "    conda run -n ct-gpu pip install --no-deps celltypist==1.7.1 "
            "rapids-singlecell==0.10.10"
        )
    if importlib.util.find_spec("rapids_singlecell") is None:
        print(
            "WARNING: 'rapids_singlecell' is not installed. The logistic "
            "regression will still be trained on the GPU, but the "
            "over-clustering used by majority voting falls back to CPU.",
            file=sys.stderr,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    _require_gpu_deps()
    _patch_celltypist_gpu_overcluster()

    os.makedirs(args.out_dir, exist_ok=True)
    train, test = load_subset(args.subset, seed=args.seed, test_size=args.test_size)

    train.obs["label"] = train.obs["annotation"].astype(str)
    test.obs["label"] = test.obs["annotation"].astype(str)

    sc.pp.highly_variable_genes(train, inplace=True, flavor="seurat_v3", n_top_genes=5000)
    train = train[:, train.var["highly_variable"]].copy()
    # Same HVG set for the test split
    test = test[:, train.var_names].copy()

    sc.pp.normalize_total(train, target_sum=10000)
    sc.pp.log1p(train)
    sc.pp.normalize_total(test, target_sum=10000)
    sc.pp.log1p(test)

    # Training phase
    train_sampler = PeakMemorySampler(gpu=True)
    train_sampler.start()
    t0 = time.perf_counter()
    model = celltypist.train(
        train,
        labels="label",
        feature_selection=True,
        with_mean=False,
        use_GPU=True,
    )
    if model is None:
        raise RuntimeError("celltypist.train returned None; cuml may not be usable.")
    t1 = time.perf_counter()
    train_peak = train_sampler.stop()
    gpu_train_peak = train_sampler.gpu_peak_bytes

    # Inference phase
    # celltypist 1.7.1 + scipy 1.13.1 produces a COO matrix from sparse arithmetic,
    # which does not support item assignment. Passing a dense AnnData works around this.
    test_dense = sc.AnnData(
        X=test.X.toarray() if hasattr(test.X, "toarray") else np.asarray(test.X),
        obs=test.obs.copy(),
        var=test.var.copy(),
    )
    inf_sampler = PeakMemorySampler(gpu=True)
    inf_sampler.start()
    t2 = time.perf_counter()
    predictions = celltypist.annotate(
        test_dense, model=model, majority_voting=True, use_GPU=True
    )
    t3 = time.perf_counter()
    inference_peak = inf_sampler.stop()
    gpu_inf_peak = inf_sampler.gpu_peak_bytes

    result = {
        "method": "CellTypist",
        "subset": args.subset,
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(train.n_vars),
        "device_used": "cuda",
        "train_s": t1 - t0,
        "inference_s": t3 - t2,
        "rss_train_peak_bytes": int(train_peak),
        "rss_inference_peak_bytes": int(inference_peak),
        "rss_total_peak_bytes": int(max(train_peak, inference_peak)),
        "gpu_train_peak_bytes": int(gpu_train_peak),
        "gpu_inference_peak_bytes": int(gpu_inf_peak),
        "gpu_peak_bytes": int(max(gpu_train_peak, gpu_inf_peak)),
        "gpu_peak_mb": float(max(gpu_train_peak, gpu_inf_peak) / 1024 ** 2),
    }
    path = write_json(result, args.out_dir)
    print(json.dumps(result, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    sys.exit(main())
