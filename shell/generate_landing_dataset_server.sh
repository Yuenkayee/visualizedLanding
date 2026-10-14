#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Small numerical kernels must not each launch 16 BLAS/OpenCV threads inside
# the planning/render workers. Cycles receives its own explicit thread budget.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python3}" -m visual_training.data.generate_server_dataset "$@"
