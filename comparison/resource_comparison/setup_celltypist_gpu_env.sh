#!/usr/bin/env bash
# Create the dedicated `ct-gpu` conda environment needed for the CellTypist
# GPU benchmark.
#
# WHY A SEPARATE ENV?
#   The `scrna` env used by the other SOTA methods has its entire scientific
#   stack pip-installed (numpy 1.26.4, pandas 1.5.3, scipy 1.13.1,
#   scikit-learn 1.6.1, scanpy 1.10.3, torch 2.8.0) and its torch pins the
#   `nvidia-*-cu12==12.8` wheels.  Installing RAPIDS into that env would
#   (a) place conda-managed numpy/pandas into the same site-packages directory
#       as the pip ones, and (b) downgrade torch's CUDA libraries to 12.5,
#   which would break scANVI / scPoli / treeArches.  `ct-gpu` is therefore
#   fully isolated.
#
#   The `ct-gpu` env keeps numpy 1.26.4, scipy 1.13.1, scikit-learn 1.6.1 and
#   scanpy 1.10.3 -- i.e. the same numerical stack as the CPU benchmark -- so
#   the CPU vs GPU comparison is not confounded by library version drift.
#
#   Driver note: this node has driver 525.105.17 (CUDA 12.0).  The RAPIDS
#   24.08 CUDA 12.x wheels are compatible through CUDA minor-version
#   compatibility (>= 525.60.13).
#
# Usage:
#     cd comparison/resource_comparison
#     ./setup_celltypist_gpu_env.sh
#
# Afterwards, `gpu_benchmark.sh` automatically picks the env up.

set -euo pipefail

CONDA_BASE="${HOME}/miniconda3"
ENV_NAME="ct-gpu"

source_conda() {
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
        exit 1
    fi
}

source_conda

if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists -- skipping creation."
else
    echo "===== Creating conda env '$ENV_NAME' with RAPIDS 24.08 ====="
    conda create -n "$ENV_NAME" -y \
        -c rapidsai -c nvidia -c conda-forge \
        "cuml=24.08" "cudf=24.08" "cugraph=24.08" "cupy>=12.0" "rmm=24.08" \
        "python=3.10" "cuda-version=12.5" "nccl>=2.9.9,<3.0a0" "nvtx>=0.2.1,<0.3" \
        "scanpy=1.10.3" "anndata=0.10.8" "numpy=1.26.4" "scipy=1.13.1" "scikit-learn=1.6.1" \
        natsort scikit-misc statsmodels scikit-image h5py loompy pyyaml requests tqdm matplotlib \
        python-igraph leidenalg openpyxl click pip
fi

echo ""
echo "===== Installing celltypist and rapids-singlecell (pip, --no-deps) ====="
# --no-deps keeps pip from upgrading numpy to 2.x, which would break the
# conda-managed RAPIDS stack (cuml/cudf 24.08 require numpy < 2).
conda run -n "$ENV_NAME" pip install --no-deps \
    "celltypist==1.7.1" "rapids-singlecell==0.10.10"

echo ""
echo "===== Verifying ====="
conda run -n "$ENV_NAME" python - <<'PY'
import numpy, pandas, scipy, sklearn, scanpy, celltypist
import cuml, cupy, cudf, rapids_singlecell

print("python packages:")
for m in (numpy, pandas, scipy, sklearn, scanpy, celltypist):
    print(f"  {m.__name__:16s} {getattr(m, '__version__', '?')}")
print("RAPIDS packages:")
for m in (cuml, cupy, cudf, rapids_singlecell):
    print(f"  {m.__name__:16s} {getattr(m, '__version__', '?')}")

# Real GPU round-trip: cuML logistic regression on synthetic data.
import numpy as np
from cuml import LogisticRegression as cuLogisticRegression

X = np.random.RandomState(0).rand(20000, 500).astype("float32")
y = (X[:, 0] + X[:, 1] > 1.0).astype("int32")
clf = cuLogisticRegression(solver="qn", max_iter=50)
clf.fit(X, y)
acc = float((clf.predict(X) == y).mean())
print(f"cuML GPU logistic regression accuracy: {acc:.4f}")
assert acc > 0.9, "cuML GPU fit produced a bad model"
print("OK")
PY

echo ""
echo "===== Environment '$ENV_NAME' ready ====="
