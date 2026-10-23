#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
LANDING_PYTHON=""
for LANDING_CANDIDATE in "${PYTHON_BIN:-}" "$PWD/.venv/bin/python" python3 python; do
  if [[ -n "$LANDING_CANDIDATE" ]] && command -v "$LANDING_CANDIDATE" >/dev/null 2>&1 &&
      "$LANDING_CANDIDATE" -c 'import sys; sys.exit(not ((3,11) <= sys.version_info[:2] < (3,13)))' >/dev/null 2>&1; then
    LANDING_PYTHON="$LANDING_CANDIDATE"
    break
  fi
done
if [[ -z "$LANDING_PYTHON" ]]; then
  printf '%s\n' 'Python 3.11/3.12 required. Set PYTHON_BIN to the simulation interpreter.' >&2
  exit 2
fi
for LANDING_ARG in "$@"; do
  if [[ "$LANDING_ARG" == --help || "$LANDING_ARG" == -h ]]; then
    "$LANDING_PYTHON" -m main.replay_sensors --help
    exit 0
  fi
done
LANDING_LOG_DIR="outputs/logs/sensor_replay/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LANDING_LOG_DIR"
{
  printf 'Replay log: %s/replay.log\n' "$LANDING_LOG_DIR"
  "$LANDING_PYTHON" shell/utility_dependencies.py replay
  "$LANDING_PYTHON" -m main.replay_sensors "$@"
} 2>&1 | tee "$LANDING_LOG_DIR/replay.log"
