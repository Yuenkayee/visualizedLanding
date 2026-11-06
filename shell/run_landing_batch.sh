#!/usr/bin/env bash
# Batch execution never installs dependencies concurrently in worker processes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LANDING_BATCH_MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$LANDING_BATCH_MODE" == check ]]; then
  exec bash shell/run_landing_closed_loop.sh check "$@"
fi
if [[ "$LANDING_BATCH_MODE" == help || "$LANDING_BATCH_MODE" == --help ]]; then
  printf '%s\n' \
    'check [--check] [--device cuda] [--render-device OPTIX]: install/check existing validated dependencies.' \
    'plan/run/analyze: main.batch CLI; run --help for all flags.' \
    'run --pilot --weather clear --seeds-per-weather 1 --duration 2 --workers 4 --output outputs/batch/smoke' \
    'run --workers 4 --output outputs/batch/formal' \
    'run --resume --workers 4 --output outputs/batch/formal' \
    'Optional --matlab-sessions Landing1,Landing2,... (one dedicated session per worker).' \
    'Without --matlab-sessions, independent worker-owned MATLAB sessions start automatically.'
  exit 0
fi
case "$LANDING_BATCH_MODE" in plan|run|analyze) ;; *) printf 'Unknown mode: %s\n' "$LANDING_BATCH_MODE" >&2; exit 2;; esac
source shell/download_mirrors.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
LANDING_BATCH_PYTHON="${PYTHON_BIN:-}"
if [[ -z "$LANDING_BATCH_PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then LANDING_BATCH_PYTHON="$PWD/.venv/bin/python"; else LANDING_BATCH_PYTHON=python3; fi
fi
if [[ "$LANDING_BATCH_MODE" == plan || "$LANDING_BATCH_MODE" == analyze ]]; then
  exec "$LANDING_BATCH_PYTHON" -m main.batch "$LANDING_BATCH_MODE" "$@"
fi
LANDING_BATCH_REPORT="outputs/logs/batch/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LANDING_BATCH_REPORT"
"$LANDING_BATCH_PYTHON" shell/closed_loop_dependencies.py --check --report-dir "$LANDING_BATCH_REPORT"
IFS= read -r LANDING_BATCH_PYTHON < "$LANDING_BATCH_REPORT/python.txt"
IFS= read -r MATLAB_ROOT < "$LANDING_BATCH_REPORT/matlab-root.txt"
export MATLAB_ROOT PATH="$MATLAB_ROOT/bin:$PATH"
if [[ "$(uname -s)" == Linux ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$MATLAB_ROOT/bin/glnxa64:"*) ;;
    *) export LD_LIBRARY_PATH="$MATLAB_ROOT/bin/glnxa64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  if [[ -f "$LANDING_BATCH_REPORT/ld-preload.txt" ]]; then
    IFS= read -r LANDING_BATCH_CPP < "$LANDING_BATCH_REPORT/ld-preload.txt"
    if [[ -n "$LANDING_BATCH_CPP" ]]; then export LD_PRELOAD="$LANDING_BATCH_CPP"; fi
  fi
fi
# Import the exact failing scipy.stats route in a fresh process before dispatch.
"$LANDING_BATCH_PYTHON" -I -c 'import numpy, scipy, torch, yaml, matplotlib, cv2, matlab.engine; from scipy.stats import chi2; print("NumPy",numpy.__version__,numpy.__file__,"SciPy",scipy.__version__,scipy.__file__,"chi2",chi2.ppf(.95,3))'
exec "$LANDING_BATCH_PYTHON" -m main.batch run "$@"
