#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ -n "${MATLAB_ROOT:-}" ]]; then export PATH="$MATLAB_ROOT/bin:$PATH"; fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  LANDING_PYTHON="$PYTHON_BIN"
elif [[ "$(uname -s)" == Darwin && -x .venv/bin/python ]]; then
  LANDING_PYTHON="$PWD/.venv/bin/python"
else
  LANDING_PYTHON=python
fi
LANDING_MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$LANDING_MODE" in
  check|configure|inspect|smoke|run|analyze) ;;
  *) printf 'Usage: %s {check|configure|inspect|smoke|run|analyze} [arguments]\n' "$0" >&2; exit 2 ;;
esac
LANDING_LOG_DIR="outputs/logs/closed_loop/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LANDING_LOG_DIR"
# Forward only environment-related arguments to the bootstrap.
LANDING_ENV_ARGS=()
if [[ "$LANDING_MODE" == smoke || "$LANDING_MODE" == configure || "$LANDING_MODE" == inspect ]]; then
  LANDING_ENV_ARGS+=(--renderer cpu)
fi
LANDING_ARGUMENTS=("$@")
for ((i=0; i<${#LANDING_ARGUMENTS[@]}; i++)); do
  case "${LANDING_ARGUMENTS[$i]}" in
    --config|--renderer|--render-python|--device|--render-device)
      if ((i+1 >= ${#LANDING_ARGUMENTS[@]})); then printf 'Missing argument value\n' >&2; exit 2; fi
      LANDING_ENV_ARGS+=("${LANDING_ARGUMENTS[$i]}" "${LANDING_ARGUMENTS[$((i+1))]}"); i=$((i+1)) ;;
  esac
done
{
  if [[ "$LANDING_MODE" == analyze ]]; then
    "$LANDING_PYTHON" shell/utility_dependencies.py replay
    "$LANDING_PYTHON" -m main.analyze "$@"
  elif [[ "$LANDING_MODE" == check ]]; then
    "$LANDING_PYTHON" -m main.check_environment --report "$LANDING_LOG_DIR/environment.json" "$@"
  else
    "$LANDING_PYTHON" -m main.check_environment --report "$LANDING_LOG_DIR/environment.json" "${LANDING_ENV_ARGS[@]}"
    "$LANDING_PYTHON" -m main.run "$LANDING_MODE" "$@"
  fi
} 2>&1 | tee "$LANDING_LOG_DIR/console.log"
printf 'Console log: %s\n' "$LANDING_LOG_DIR/console.log"
