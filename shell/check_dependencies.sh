#!/usr/bin/env bash
# Install into the repository venv; never alter the system Python.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CHECK=0
BLENDER=1
MATLAB=auto
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --without-blender) BLENDER=0 ;;
    --with-matlab) MATLAB=yes ;;
    --without-matlab) MATLAB=no ;;
    --help) echo 'Usage: shell/check_dependencies.sh [--check] [--without-blender] [--with-matlab|--without-matlab]'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
# Locate licensed MATLAB before trying to build its Engine package.
MATLAB_ROOT="${MATLAB_ROOT:-}"
if [[ -z "$MATLAB_ROOT" ]] && command -v matlab >/dev/null 2>&1; then
  MATLAB_ROOT="$(matlab -batch 'disp(matlabroot)' 2>/dev/null | tail -1)"
fi
if [[ -z "$MATLAB_ROOT" && -d /Applications/MATLAB_R2024b.app ]]; then MATLAB_ROOT=/Applications/MATLAB_R2024b.app; fi
if [[ "$MATLAB" == auto ]]; then
  if [[ -n "$MATLAB_ROOT" ]]; then MATLAB=yes; else MATLAB=no; fi
fi
if [[ "$MATLAB" == yes && ! -x "$MATLAB_ROOT/bin/matlab" ]]; then
  echo 'MATLAB R2024b not found. Install/activate the licensed application and set MATLAB_ROOT.' >&2
  exit 2
fi
if [[ -n "$MATLAB_ROOT" ]]; then export MATLAB_ROOT; export PATH="$MATLAB_ROOT/bin:$PATH"; fi
if [[ ! -x "$PY" ]]; then
  if [[ "$CHECK" == 1 ]]; then echo 'Missing .venv (run without --check to install).'; exit 2; fi
  mkdir -p "$ROOT/.tools"
  if command -v uv >/dev/null 2>&1; then UV="$(command -v uv)"; else
    command -v curl >/dev/null 2>&1 || { echo 'curl is required'; exit 2; }
    case "$(uname -s):$(uname -m)" in
      Darwin:arm64) TARGET=aarch64-apple-darwin ;;
      Darwin:x86_64) TARGET=x86_64-apple-darwin ;;
      Linux:x86_64) TARGET=x86_64-unknown-linux-gnu ;;
      Linux:aarch64) TARGET=aarch64-unknown-linux-gnu ;;
      *) echo 'Unsupported OS/architecture; install Python 3.11 and create .venv manually.'; exit 2 ;;
    esac
    curl -fL --retry 3 "https://github.com/astral-sh/uv/releases/download/0.9.5/uv-$TARGET.tar.gz" -o "$ROOT/.tools/uv.tar.gz"
    tar -xzf "$ROOT/.tools/uv.tar.gz" -C "$ROOT/.tools"
    UV="$ROOT/.tools/uv-$TARGET/uv"
  fi
  UV_CACHE_DIR="$ROOT/.tools/uv-cache" UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python" "$UV" venv --python 3.11 "$VENV"
fi
"$PY" -c 'import sys; assert sys.version_info[:2]==(3,11), "Use Python 3.11 for the locked Blender/MATLAB environment"'
LOCK=requirements.txt
if [[ "$BLENDER" == 1 ]]; then
  if command -v blender >/dev/null 2>&1 || [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then
    echo 'Blender executable available (require 4.2 LTS or compatible).'
  else LOCK=requirements-blender.txt; fi
fi
check_lock() {
  "$PY" - "$1" <<'PYCODE'
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    from packaging.requirements import Requirement
except ImportError:
    print('Missing packaging/dependency environment'); sys.exit(2)
missing=[]
for line in open(sys.argv[1]):
    line=line.strip()
    if not line or line.startswith('#'): continue
    req=Requirement(line)
    if req.marker and not req.marker.evaluate(): continue
    try: current=version(req.name)
    except PackageNotFoundError: current=None
    if current is None or current not in req.specifier:
        missing.append(f'{req.name}: {current or "missing"} -> {req.specifier}')
for item in missing: print(item)
sys.exit(2 if missing else 0)
PYCODE
}
STATUS=0
if ! check_lock "$LOCK"; then STATUS=2; fi
if [[ "$MATLAB" == yes ]]; then if ! check_lock requirements-matlab.txt; then STATUS=2; fi
else echo 'MATLAB unavailable/disabled: mock backend is usable; licensed MATLAB cannot be auto-installed.'; fi
if [[ "$CHECK" == 1 ]]; then
  if [[ "$STATUS" == 0 ]]; then echo "All selected dependency versions match $LOCK."; fi
  exit "$STATUS"
fi
if [[ "$STATUS" != 0 ]]; then
  if command -v uv >/dev/null 2>&1; then UV="$(command -v uv)";
  elif [[ -n "${UV:-}" ]]; then :;
  else
    "$PY" -m ensurepip --upgrade
    "$PY" -m pip install --requirement "$LOCK"
    if [[ "$MATLAB" == yes ]]; then "$PY" -m pip install --requirement requirements-matlab.txt; fi
    UV=''
  fi
  if [[ -n "$UV" ]]; then
    UV_CACHE_DIR="$ROOT/.tools/uv-cache" "$UV" pip install --python "$PY" --requirement "$LOCK"
    if [[ "$MATLAB" == yes ]]; then UV_CACHE_DIR="$ROOT/.tools/uv-cache" "$UV" pip install --python "$PY" --requirement requirements-matlab.txt; fi
  fi
fi
check_lock "$LOCK"
if [[ "$MATLAB" == yes ]]; then check_lock requirements-matlab.txt; "$PY" -c 'import matlab.engine; print("MATLAB Engine import OK")'; fi
if [[ "$LOCK" == requirements-blender.txt ]]; then "$PY" -c 'import bpy; assert bpy.app.version[:2] == (4,2); print("Blender runtime",bpy.app.version_string)'; fi
"$PY" -c 'import numpy,scipy,cv2,yaml,torch,PIL,matplotlib,onnx,onnxruntime,pytest; print("Core imports OK")'
# Check declared transitive requirements independently of broken WHEEL metadata:
# bpy 4.2's cp311 macOS wheel internally says cp39, despite working on cp311.
# Native import/version/render checks establish runtime compatibility separately.
"$PY" - <<'PYCODE'
from importlib.metadata import distributions, version, PackageNotFoundError
from packaging.requirements import Requirement
errors=[]
for dist in distributions():
    for declaration in dist.requires or []:
        req=Requirement(declaration)
        if req.marker and not req.marker.evaluate(): continue
        try: current=version(req.name)
        except PackageNotFoundError: current=None
        if current is None or current not in req.specifier:
            errors.append(f'{dist.metadata["Name"]} requires {req}; installed {current}')
if errors:
    print('\n'.join(errors)); raise SystemExit(2)
print('All declared transitive dependency constraints satisfied.')
PYCODE
echo 'Dependencies ready. Activate with: source .venv/bin/activate'
