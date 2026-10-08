#!/usr/bin/env bash
# Preserve the caller's active environment; do not resolve executable symlinks.
resolve_landing_python() {
  local candidate="${1:-${LANDING_PYTHON:-}}"
  if [[ -z "$candidate" ]]; then
    if command -v python >/dev/null 2>&1; then candidate=python
    elif command -v python3 >/dev/null 2>&1; then candidate=python3
    else
      echo 'Python not found. Activate your existing environment or pass --python /path/to/python.' >&2
      return 2
    fi
  fi
  if ! LANDING_PYTHON="$("$candidate" -c 'import os, sys; print(os.path.abspath(sys.executable))')"; then
    echo "Cannot run Python: $candidate" >&2
    return 2
  fi
  if [[ ! -x "$LANDING_PYTHON" ]]; then
    echo "Invalid Python executable: $LANDING_PYTHON" >&2
    return 2
  fi
  export LANDING_PYTHON
}
