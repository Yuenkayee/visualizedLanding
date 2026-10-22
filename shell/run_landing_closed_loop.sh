#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source shell/download_mirrors.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}" PIP_RETRIES="${PIP_RETRIES:-5}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}" UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
if [[ -n "${MATLAB_ROOT:-}" ]]; then export PATH="$MATLAB_ROOT/bin:$PATH"; fi
LANDING_MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$LANDING_MODE" in
  check|configure|inspect|smoke|run|analyze) ;;
  *) printf 'Usage: %s {check|configure|inspect|smoke|run|analyze} [arguments]\n' "$0" >&2; exit 2 ;;
esac
for LANDING_ARG in "$@"; do
  if [[ "$LANDING_ARG" == --help || "$LANDING_ARG" == -h ]]; then
    printf '%s\n' 'check: install missing Ubuntu/Python/MATLAB/renderer dependencies and verify licenses/GPU.' \
      'check --check: inspect only. check --login: install, open interactive MATLAB login, then verify.' \
      'Modes: check, inspect, smoke (interface fixture), run (actual aircraft), configure, analyze.' \
      'Options: --config FILE --device {cpu,cuda,mps} --render-device {CPU,CUDA,OPTIX}' \
      'Run/smoke: --duration SECONDS --checkpoint FILE --output DIR --weather NAME' \
      'Environment: MATLAB_ROOT, PYTHON_BIN, RENDER_PYTHON, UV_PYTHON_INSTALL_MIRROR.' \
      'MATLAB_LICENSE_MODE={online,onlinelicensing,file,auto} applies to login/check/run.' \
      'MATLAB_SHARED_SESSION=NAME: check/run connect to an existing named MATLAB session.' \
      'LANDING_DOWNLOAD_PROFILE=china: TUNA PyPI/Ubuntu/Python source + NJU CPython binaries.' \
      'Default run/check/inspect config: main/navigation_observer.yaml. See main/README.md.'
    exit 0
  fi
done
LANDING_LOG_DIR="outputs/logs/closed_loop/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LANDING_LOG_DIR"
LANDING_BOOTSTRAP_ARGS=(--report-dir "$LANDING_LOG_DIR")
LANDING_ARGUMENTS=()
LANDING_CHECK_ONLY=0
LANDING_HAS_CONFIG=0
for LANDING_ARG in "$@"; do
  case "$LANDING_ARG" in
    --login)
      if [[ "$LANDING_MODE" != check ]]; then printf '%s\n' '--login is only valid for check' >&2; exit 2; fi
      LANDING_BOOTSTRAP_ARGS+=(--login) ;;
    --check)
      if [[ "$LANDING_MODE" != check ]]; then printf '%s\n' '--check is only valid for check' >&2; exit 2; fi
      LANDING_CHECK_ONLY=1
      LANDING_BOOTSTRAP_ARGS+=(--check)
      LANDING_ARGUMENTS+=(--check) ;;
    --config|--config=*) LANDING_HAS_CONFIG=1; LANDING_ARGUMENTS+=("$LANDING_ARG") ;;
    *) LANDING_ARGUMENTS+=("$LANDING_ARG") ;;
  esac
done
if [[ "$LANDING_MODE" == analyze ]]; then
  LANDING_BOOTSTRAP_ARGS+=(--analysis-only)
elif [[ "$LANDING_HAS_CONFIG" == 0 ]]; then
  case "$LANDING_MODE" in
    check|run|inspect) LANDING_ARGUMENTS+=(--config main/navigation_observer.yaml) ;;
    smoke|configure) LANDING_ARGUMENTS+=(--config main/closed_loop.yaml) ;;
  esac
fi
# Ubuntu's system Python 3.10 is sufficient to bootstrap the actual runtimes.
LANDING_BOOTSTRAP_PYTHON=""
for LANDING_CANDIDATE in "${PYTHON_BIN:-}" "$PWD/.venv/bin/python" python3 python python3.12 python3.11; do
  if [[ -n "$LANDING_CANDIDATE" ]] && command -v "$LANDING_CANDIDATE" >/dev/null 2>&1 &&
    "$LANDING_CANDIDATE" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    LANDING_BOOTSTRAP_PYTHON="$LANDING_CANDIDATE"; break
  fi
done
# Forward environment flags only; all run-specific overrides still go to main.run.
LANDING_ENV_ARGS=()
if [[ "$LANDING_MODE" == smoke || "$LANDING_MODE" == configure || "$LANDING_MODE" == inspect ]]; then
  LANDING_ENV_ARGS+=(--renderer cpu)
fi
for ((i=0; i<${#LANDING_ARGUMENTS[@]}; i++)); do
  case "${LANDING_ARGUMENTS[$i]}" in
    --config|--renderer|--render-python|--device|--render-device|--checkpoint)
      if ((i+1 >= ${#LANDING_ARGUMENTS[@]})); then printf 'Missing argument value\n' >&2; exit 2; fi
      LANDING_ENV_ARGS+=("${LANDING_ARGUMENTS[$i]}" "${LANDING_ARGUMENTS[$((i+1))]}"); i=$((i+1)) ;;
    --config=*|--renderer=*|--render-python=*|--device=*|--render-device=*|--checkpoint=*)
      LANDING_ENV_ARGS+=("${LANDING_ARGUMENTS[$i]}") ;;
  esac
done
{
  printf 'Console log: %s\n' "$LANDING_LOG_DIR/console.log"
  printf '[deploy] Download profile: %s\n' "${LANDING_DOWNLOAD_PROFILE:-default}"
  if [[ -z "$LANDING_BOOTSTRAP_PYTHON" ]]; then
    if [[ "$(uname -s)" != Linux || "$LANDING_CHECK_ONLY" == 1 ]]; then
      printf 'Python >=3.10 required to bootstrap; install python3 first.\n' >&2; exit 2
    fi
    LANDING_SUDO=()
    if [[ "$(id -u)" != 0 ]]; then LANDING_SUDO=(sudo); fi
    "${LANDING_SUDO[@]}" apt-get update
    "${LANDING_SUDO[@]}" apt-get install -y python3 python3-venv
    LANDING_BOOTSTRAP_PYTHON=python3
  fi
  "$LANDING_BOOTSTRAP_PYTHON" shell/closed_loop_dependencies.py "${LANDING_BOOTSTRAP_ARGS[@]}"
  IFS= read -r LANDING_PYTHON < "$LANDING_LOG_DIR/python.txt"
  IFS= read -r MATLAB_ROOT < "$LANDING_LOG_DIR/matlab-root.txt"
  export MATLAB_ROOT
  export PATH="$MATLAB_ROOT/bin:$PATH"
  # The Engine wheel builder searches LD_LIBRARY_PATH, not MATLAB_ROOT/PATH.
  # Export before Python starts so pip's isolated build and the runtime loader
  # both see a non-default MATLAB installation. Do not add sys/os libraries.
  if [[ "$(uname -s)" == Linux && "$LANDING_MODE" != analyze ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$MATLAB_ROOT/bin/glnxa64:"*) ;;
      *) export LD_LIBRARY_PATH="$MATLAB_ROOT/bin/glnxa64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    # The bootstrap validates a replacement only if Python/Conda carries an
    # older C++ runtime. Apply it before BOTH check and simulation processes.
    if [[ -f "$LANDING_LOG_DIR/ld-preload.txt" ]]; then
      IFS= read -r LANDING_CPP_PRELOAD < "$LANDING_LOG_DIR/ld-preload.txt"
      if [[ -n "$LANDING_CPP_PRELOAD" ]]; then export LD_PRELOAD="$LANDING_CPP_PRELOAD"; fi
    fi
  fi
  "$LANDING_PYTHON" -c 'import sys; print("Python:", sys.executable, sys.version.split()[0], flush=True)'
  if [[ "$LANDING_MODE" == analyze ]]; then
    "$LANDING_PYTHON" shell/utility_dependencies.py replay
    "$LANDING_PYTHON" -m main.analyze "${LANDING_ARGUMENTS[@]}"
  elif [[ "$LANDING_MODE" == check ]]; then
    "$LANDING_PYTHON" -m main.check_environment --report "$LANDING_LOG_DIR/environment.json" "${LANDING_ARGUMENTS[@]}"
  else
    "$LANDING_PYTHON" -m main.check_environment --report "$LANDING_LOG_DIR/environment.json" "${LANDING_ENV_ARGS[@]}"
    "$LANDING_PYTHON" -m main.run "$LANDING_MODE" "${LANDING_ARGUMENTS[@]}"
  fi
} 2>&1 | tee "$LANDING_LOG_DIR/console.log"
printf 'Console log: %s\n' "$LANDING_LOG_DIR/console.log"
