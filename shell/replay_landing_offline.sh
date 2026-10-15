#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
LANDING_PYTHON="${PYTHON_BIN:-python}"
LANDING_LOG_DIR="outputs/logs/replay/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LANDING_LOG_DIR"
{
  "$LANDING_PYTHON" shell/utility_dependencies.py replay
  "$LANDING_PYTHON" -m simulation.offline.replay_dataset "$@"
} 2>&1 | tee "$LANDING_LOG_DIR/replay.log"
printf 'Replay log: %s\n' "$LANDING_LOG_DIR/replay.log"
