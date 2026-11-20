#!/usr/bin/env bash
# One fresh output directory per launch. Shared MATLAB sessions must be idle.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/batch
out=$(mktemp -d "outputs/batch/c3_four_weather_eiH_window_w12h20_w15h30_t80_n80_w6_$(date +%Y%m%d_%H%M%S)_XXXXXX")
nohup bash shell/run_landing_batch.sh sea56-c3 "$out" >"$out/batch.log" 2>&1 </dev/null &
printf 'PID: %s\n结果目录: %s\n查看进度: tail -f %q/batch.log\n' "$!" "$out" "$out"
