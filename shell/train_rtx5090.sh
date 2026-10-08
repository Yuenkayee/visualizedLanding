#!/usr/bin/env bash
# Run installation and training as one logged job, detached in tmux by default.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/shell/python_environment.sh"
SESSION=landing_train
FOREGROUND=0
LOG=''
PYTHON_ARG=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session|--log|--python)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      if [[ "$1" == --session ]]; then SESSION="$2"
      elif [[ "$1" == --python ]]; then PYTHON_ARG="$2"
      else LOG="$2"; fi
      shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    --help)
      echo 'Usage: bash shell/train_rtx5090.sh [--python PATH] [--session NAME] [--foreground] [--log PATH] [-- TRAIN_ARGS...]'
      echo 'Uses the current Python environment (no .venv creation). Installs/checks dependencies before training in tmux.'
      echo 'Example: bash shell/train_rtx5090.sh -- --data-root /data/multi_camera --batch-size 8'
      exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [[ ! "$SESSION" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'Session name must contain only letters, digits, underscore or dash.' >&2; exit 2
fi
if [[ "$FOREGROUND" == 0 ]]; then
  command -v tmux >/dev/null 2>&1 || { echo 'Install tmux: sudo apt install tmux (or use --foreground).' >&2; exit 2; }
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "Session already exists; attach with: tmux attach -t $SESSION" >&2; exit 2
  fi
fi
resolve_landing_python "$PYTHON_ARG"
PY="$LANDING_PYTHON"
cd "$ROOT"
if [[ -z "$LOG" ]]; then
  mkdir -p "$ROOT/outputs/logs"
  LOG="$(mktemp "$ROOT/outputs/logs/rtx5090_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
  mv -- "$LOG" "$LOG.log"
  LOG="$LOG.log"
else
  [[ "$LOG" == /* ]] || LOG="$ROOT/$LOG"
  mkdir -p "$(dirname -- "$LOG")"
fi
if [[ "$FOREGROUND" == 0 ]]; then
  # tmux >=3.2 (Ubuntu 22.04) accepts separate command arguments without eval.
  # Refresh installer/GPU settings even when tmux was started in an older SSH
  # session. Explicitly remove unset variables (an empty CUDA_VISIBLE_DEVICES
  # would hide all GPUs).
  ENV_COMMAND=(env)
  ENV_VALUES=("PATH=$PATH")
  for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy \
      PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_CERT PIP_CONFIG_FILE PIP_REQUIRE_VIRTUALENV \
      PIP_USER PIP_TARGET PIP_PREFIX LANDING_PYTHON CUDA_VISIBLE_DEVICES \
      CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL VIRTUAL_ENV PYTHONHOME PYTHONPATH \
      PYTHONNOUSERSITE PYTHONUSERBASE LD_LIBRARY_PATH SSL_CERT_FILE REQUESTS_CA_BUNDLE; do
    if value="$(printenv "$name")"; then
      ENV_VALUES+=("$name=$value")
    else
      ENV_COMMAND+=(-u "$name")
    fi
  done
  tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "${ENV_COMMAND[@]}" "${ENV_VALUES[@]}" \
    bash "$ROOT/shell/train_rtx5090.sh" --foreground --python "$PY" --log "$LOG" -- "$@"
  echo "Background job started. Attach: tmux attach -t $SESSION"
  echo 'Detach without stopping: Ctrl+b, then d'
  printf 'Log: %s\nExit status after completion: %s.exit_code\n' "$LOG" "$LOG"
  exit 0
fi
printf 'Python: %s\nLog: %s\n' "$PY" "$LOG"
# Explicitly stop on failed setup, even inside a pipeline/conditional.
if (
  bash "$ROOT/shell/check_dependencies.sh" --rtx5090 --python "$PY" || exit "$?"
  "$PY" -u -m visual_training.train \
    --config "$ROOT/configs/multi_camera_training_rtx5090.yaml" "$@"
) 2>&1 | tee -a "$LOG"; then
  result=0
else
  result=$?
fi
printf '%s\n' "$result" > "$LOG.exit_code"
printf 'Job finished with exit code %s. Log: %s\n' "$result" "$LOG"
exit "$result"
