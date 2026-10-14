#!/usr/bin/env bash
# Internal tmux worker wrapper. pipefail preserves Python failures through tee.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
"$1" -u -m visual_training.server_train --run-job "$2" 2>&1 | tee -a "$3"
