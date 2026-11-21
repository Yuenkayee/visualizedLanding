#!/usr/bin/env bash
# Batch execution never installs dependencies concurrently in worker processes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LANDING_BATCH_MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$LANDING_BATCH_MODE" == check ]]; then
  exec bash shell/run_landing_closed_loop.sh check --batch-policy main/batch_closed_loop.yaml "$@"
fi
if [[ "$LANDING_BATCH_MODE" == help || "$LANDING_BATCH_MODE" == --help ]]; then
  printf '%s\n' \
    'check [--check] [--matlab-sessions NAME,...]: dependencies, Chinese fonts and two disposable model probes.' \
    'plan/run/analyze/status: main.batch CLI; run --help for all flags.' \
    'run --c3-only: 80 C3 sorties, four weathers and both specified sea conditions, at most 80 s each.' \
    'Default full matrix: 320 sorties; revised thesis analysis generates exactly 15 PNGs.' \
    'run --pilot --weather clear --seeds-per-weather 1 --duration 2 --workers 4 --output outputs/batch/smoke' \
    'run --workers 4 --output outputs/batch/formal' \
    'run --resume --workers 4 --output outputs/batch/formal' \
    'Optional --matlab-sessions Landing1,Landing2,... (one dedicated session per worker).' \
    'Without --matlab-sessions, independent worker-owned MATLAB sessions start automatically.'
  exit 0
fi
case "$LANDING_BATCH_MODE" in configured-start|configured-check|configured|clear-c3|sea56-c3|plan|run|analyze|status) ;; *) printf 'Unknown mode: %s\n' "$LANDING_BATCH_MODE" >&2; exit 2;; esac
source shell/download_mirrors.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
LANDING_BATCH_PYTHON="${PYTHON_BIN:-}"
if [[ -z "$LANDING_BATCH_PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then LANDING_BATCH_PYTHON="$PWD/.venv/bin/python"; else LANDING_BATCH_PYTHON=python3; fi
fi
if [[ "$LANDING_BATCH_MODE" == plan || "$LANDING_BATCH_MODE" == analyze || "$LANDING_BATCH_MODE" == status ]]; then
  exec "$LANDING_BATCH_PYTHON" -m main.batch "$LANDING_BATCH_MODE" "$@"
fi
LANDING_BATCH_REPORT="outputs/logs/batch/$(date +%Y%m%d_%H%M%S)_$$"
if [[ "$LANDING_BATCH_MODE" == configured || "$LANDING_BATCH_MODE" == configured-start ]]; then LANDING_BATCH_REPORT="$1/dependencies"; fi
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
case "$LANDING_BATCH_MODE" in
  configured|configured-start|configured-check)
    # Run under the same interpreter and native library environment as the workers.
    LANDING_C3_SESSIONS=$("$LANDING_BATCH_PYTHON" -m main.batch.session_licenses)
    export LANDING_C3_SESSIONS
    if [[ "$LANDING_BATCH_MODE" == configured-check ]]; then exit 0; fi
    if [[ "$LANDING_BATCH_MODE" == configured-start ]]; then
      out=$1
      nohup "$LANDING_BATCH_PYTHON" -m main.batch.clear_c3 "$out" configured >"$out/batch.log" 2>&1 </dev/null &
      pid=$!
      printf '%s\n' "$out" > "outputs/batch/.latest_configured_batch.$$"
      mv "outputs/batch/.latest_configured_batch.$$" outputs/batch/.latest_configured_batch
      printf 'PID: %s\n结果目录: %s\n日志: %s/batch.log\n' "$pid" "$out" "$out"
      exit 0
    fi ;;
esac
if [[ "$LANDING_BATCH_MODE" == clear-c3 || "$LANDING_BATCH_MODE" == sea56-c3 || "$LANDING_BATCH_MODE" == configured ]]; then
  exec "$LANDING_BATCH_PYTHON" -m main.batch.clear_c3 "$@" "$LANDING_BATCH_MODE"
fi
exec "$LANDING_BATCH_PYTHON" -m main.batch run "$@"
