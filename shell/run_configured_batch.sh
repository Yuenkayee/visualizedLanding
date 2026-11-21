#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/batch
case "${1:-start}" in
  logs) read -r out < "$root/.latest_configured_batch"; exec tail -n 40 -f "$out/batch.log" ;;
  check) exec bash shell/run_landing_batch.sh configured-check ;;
  start) ;;
  *) echo 'Usage: bash shell/run_configured_batch.sh [start|check|logs]' >&2; exit 2 ;;
esac
out=$("${PYTHON_BIN:-python3}" -m main.batch.user_config configs/closed_loop_batch.json "$root")
exec bash shell/run_landing_batch.sh configured-start "$out"
