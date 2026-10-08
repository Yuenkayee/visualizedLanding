#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/shell/python_environment.sh"
PYTHON_OVERRIDE=""
ARGS=()
while (($#)); do
  case "$1" in
    --python)
      if (($# < 2)); then echo '--python needs a path' >&2; exit 2; fi
      PYTHON_OVERRIDE="$2"; shift 2 ;;
    --python=*) PYTHON_OVERRIDE="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
# Resolve before cd, so relative executable paths refer to the caller's directory.
resolve_landing_python "$PYTHON_OVERRIDE"
cd "$ROOT"
echo "Launcher Python: $LANDING_PYTHON"
exec "$LANDING_PYTHON" -u -m visual_training.data.blender_runtime ${ARGS[@]+"${ARGS[@]}"}
