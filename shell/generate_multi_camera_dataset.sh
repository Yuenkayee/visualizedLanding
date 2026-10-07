#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for arg in "$@"; do
  if [[ "$arg" == '--plan-only' || "$arg" == cpu ]]; then
    exec .venv/bin/python -m visual_training.data.generate_multi_camera_dataset "$@"
  fi
done
if command -v blender >/dev/null 2>&1; then BLENDER="$(command -v blender)";
elif [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then BLENDER=/Applications/Blender.app/Contents/MacOS/Blender;
else exec .venv/bin/python -m visual_training.data.generate_multi_camera_dataset "$@"; fi
export LANDING_SITE_PACKAGES="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
export LANDING_DATASET_MODULE=visual_training.data.generate_multi_camera_dataset
exec "$BLENDER" --background --python visual_training/data/blender_landing_entry.py -- "$@"
