#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
LANDING_PYTHON="${PYTHON_BIN:-python}"
LANDING_LOG_DIR="outputs/logs/dataset_prepare/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LANDING_LOG_DIR"
{
  "$LANDING_PYTHON" shell/utility_dependencies.py prepare
  "$LANDING_PYTHON" -m visual_training.data.prepare_four_weather "$@"
} 2>&1 | tee "$LANDING_LOG_DIR/prepare.log"
printf 'Preparation log: %s\n' "$LANDING_LOG_DIR/prepare.log"
