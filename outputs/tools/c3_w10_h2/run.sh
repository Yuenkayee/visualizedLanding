#!/usr/bin/env bash
set -euo pipefail
LANDING_C3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LANDING_C3_DIR/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
LANDING_C3_PYTHON="${PYTHON_BIN:-python3}"
if [[ "${1:-}" == run ]]; then
  LANDING_C3_REPORT="outputs/logs/c3_w10_h2/check_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$LANDING_C3_REPORT"
  "$LANDING_C3_PYTHON" shell/closed_loop_dependencies.py --check --report-dir "$LANDING_C3_REPORT"
  IFS= read -r LANDING_C3_PYTHON < "$LANDING_C3_REPORT/python.txt"
  IFS= read -r MATLAB_ROOT < "$LANDING_C3_REPORT/matlab-root.txt"
  export MATLAB_ROOT PATH="$MATLAB_ROOT/bin:$PATH"
  if [[ "$(uname -s)" == Linux ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$MATLAB_ROOT/bin/glnxa64:"*) ;;
      *) export LD_LIBRARY_PATH="$MATLAB_ROOT/bin/glnxa64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    if [[ -f "$LANDING_C3_REPORT/ld-preload.txt" ]]; then
      IFS= read -r LANDING_C3_PRELOAD < "$LANDING_C3_REPORT/ld-preload.txt"
      if [[ -n "$LANDING_C3_PRELOAD" ]]; then export LD_PRELOAD="$LANDING_C3_PRELOAD"; fi
    fi
  fi
fi
exec "$LANDING_C3_PYTHON" "$LANDING_C3_DIR/run.py" "$@"
