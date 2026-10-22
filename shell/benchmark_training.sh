#!/usr/bin/env bash
# Reuse the current-environment, dependency-checking, logged tmux launcher.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash shell/benchmark_training.sh [--python PATH] [--session NAME] [--foreground] [--log PATH] [-- BENCHMARK_ARGS...]'
  echo 'Default tmux session: landing_benchmark. Uses the active Python; no .venv creation.'
  echo 'Example: bash shell/benchmark_training.sh -- --checkpoint visual_training/checkpoints/multi_camera/best.pt --output outputs/benchmarks/server'
  echo 'Python options: python -m visual_training.benchmark --help'
  exit 0
fi
exec bash "$ROOT/shell/train_rtx5090.sh" --benchmark "$@"
