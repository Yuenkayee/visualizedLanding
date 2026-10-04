#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if command -v blender >/dev/null 2>&1; then BLENDER="$(command -v blender)";
elif [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then BLENDER=/Applications/Blender.app/Contents/MacOS/Blender;
else exec .venv/bin/python -m visual_modeling.blender.render_sequence "$@"; fi
export LANDING_SITE_PACKAGES="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
exec "$BLENDER" --background --python visual_modeling/blender/render_sequence.py -- "$@"
