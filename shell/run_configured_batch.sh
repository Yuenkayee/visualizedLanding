#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=outputs/batch
case "${1:-start}" in
  logs) read -r out < "$root/.latest_configured_batch"; exec tail -n 40 -f "$out/batch.log" ;;
  start) ;;
  *) echo 'Usage: bash shell/run_configured_batch.sh [start|logs]' >&2; exit 2 ;;
esac
out=$("${PYTHON_BIN:-python3}" -m main.batch.user_config configs/closed_loop_batch.json "$root")
nohup bash shell/run_landing_batch.sh configured "$out" >"$out/batch.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$out" > "$root/.latest_configured_batch.$$"
mv "$root/.latest_configured_batch.$$" "$root/.latest_configured_batch"
printf 'PID: %s\n结果目录: %s\n日志: %s/batch.log\n' "$pid" "$out" "$out"
