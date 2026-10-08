#!/usr/bin/env bash
# Check/install into the selected existing Python environment; no .venv creation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/shell/python_environment.sh"
CHECK=0
BLENDER=1
MATLAB=auto
RTX5090=0
PYTHON_VERSION=3.11
PYTHON_ARG=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1; shift ;;
    --without-blender) BLENDER=0; shift ;;
    --with-matlab) MATLAB=yes; shift ;;
    --without-matlab) MATLAB=no; shift ;;
    --rtx5090) RTX5090=1; shift ;;
    --python)
      [[ $# -ge 2 && -n "$2" ]] || { echo 'Missing value for --python' >&2; exit 2; }
      PYTHON_ARG="$2"; shift 2 ;;
    --help)
      echo 'Usage: shell/check_dependencies.sh [--python PATH] [--check] [--rtx5090] [--without-blender] [--with-matlab|--without-matlab]'
      echo 'Uses --python, LANDING_PYTHON, or active python/python3, in that order. Does not create .venv.'
      echo 'Installs missing/mismatched locked packages into that environment; --check never installs.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
# Resolve relative interpreter paths before changing to the repository directory.
resolve_landing_python "$PYTHON_ARG"
PY="$LANDING_PYTHON"
cd "$ROOT"
if [[ "$RTX5090" == 1 ]]; then
  PYTHON_VERSION=3.12
  if [[ "$(uname -s):$(uname -m)" != Linux:x86_64 ]]; then
    echo 'The RTX 5090 dependency lock requires Linux x86_64 / Python 3.12.' >&2; exit 2
  fi
  if [[ "$MATLAB" == yes ]]; then
    echo '--rtx5090 selects training only; use a separate environment for MATLAB/Blender.' >&2; exit 2
  fi
  BLENDER=0
  MATLAB=no
fi
"$PY" -c 'import sys
expected = tuple(map(int, sys.argv[1].split(".")))
if sys.version_info[:2] != expected:
    raise SystemExit(f"This profile requires Python {sys.argv[1]}, but {sys.executable} uses {sys.version.split()[0]}. Activate the matching environment or pass --python /path/to/python{sys.argv[1]}.")
print(f"Selected Python: {sys.executable} ({sys.version.split()[0]})")' "$PYTHON_VERSION"

MATLAB_ROOT="${MATLAB_ROOT:-}"
if [[ "$MATLAB" != no && -z "$MATLAB_ROOT" ]] && command -v matlab >/dev/null 2>&1; then
  MATLAB_ROOT="$(matlab -batch 'disp(matlabroot)' 2>/dev/null | tail -1)"
fi
if [[ -z "$MATLAB_ROOT" && -d /Applications/MATLAB_R2024b.app ]]; then MATLAB_ROOT=/Applications/MATLAB_R2024b.app; fi
if [[ "$MATLAB" == auto ]]; then
  if [[ -n "$MATLAB_ROOT" ]]; then MATLAB=yes; else MATLAB=no; fi
fi
if [[ "$MATLAB" == yes && ! -x "$MATLAB_ROOT/bin/matlab" ]]; then
  echo 'MATLAB R2024b not found. Install/activate the licensed application and set MATLAB_ROOT.' >&2; exit 2
fi
if [[ "$MATLAB" == yes ]]; then export MATLAB_ROOT; export PATH="$MATLAB_ROOT/bin:$PATH"; fi
LOCK=requirements.txt
if [[ "$RTX5090" == 1 ]]; then
  LOCK=requirements-rtx5090.txt
elif [[ "$BLENDER" == 1 ]]; then
  if command -v blender >/dev/null 2>&1 || [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then
    echo 'Blender executable available (require 4.2 LTS or compatible).'
  else LOCK=requirements-blender.txt; fi
fi
LOCKS=("$LOCK")
if [[ "$MATLAB" == yes ]]; then LOCKS+=(requirements-matlab.txt); fi
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
    if not line or line.startswith(('#', '--')): continue
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
MISSING_LOCKS=()
for selected_lock in "${LOCKS[@]}"; do
  if ! check_lock "$selected_lock"; then MISSING_LOCKS+=("$selected_lock"); fi
done
if [[ "${#MISSING_LOCKS[@]}" -gt 0 ]]; then
  if [[ "$CHECK" == 1 ]]; then
    echo 'Dependencies are missing or mismatched. Rerun without --check to install.' >&2; exit 2
  fi
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "Bootstrapping pip for $PY..."
    if ! "$PY" -m ensurepip --upgrade; then
      echo "pip is unavailable for $PY. Install pip for this interpreter and rerun. Training has not started." >&2
      exit 2
    fi
  fi
  for selected_lock in "${MISSING_LOCKS[@]}"; do
    echo "Installing missing/mismatched packages from $selected_lock into $PY..."
    if ! "$PY" -m pip install --retries 5 --timeout 120 --requirement "$selected_lock"; then
      echo "Dependency installation failed for $PY. Check the pip error above (network, permissions or environment policy), then rerun. Training has not started." >&2
      exit 2
    fi
  done
fi
# Recheck installed metadata, actual imports and only the selected project's
# dependency declarations. Unrelated packages in a shared environment are not
# included (bpy's broken macOS WHEEL tags are also avoided).
for selected_lock in "${LOCKS[@]}"; do check_lock "$selected_lock"; done
if [[ "$MATLAB" == yes ]]; then "$PY" -c 'import matlab.engine; print("MATLAB Engine import OK")'; fi
if [[ "$LOCK" == requirements-blender.txt ]]; then "$PY" -c 'import bpy; assert bpy.app.version[:2] == (4,2); print("Blender runtime",bpy.app.version_string)'; fi
"$PY" -c 'import numpy,scipy,cv2,yaml,torch,PIL,matplotlib,onnx,onnxruntime,pytest; print("Core imports OK")'
"$PY" - "${LOCKS[@]}" <<'PYCODE'
import sys
from importlib.metadata import distribution, version, PackageNotFoundError
from packaging.requirements import Requirement
selected=set()
for path in sys.argv[1:]:
    for line in open(path):
        line=line.strip()
        if not line or line.startswith(('#', '--')): continue
        req=Requirement(line)
        if not req.marker or req.marker.evaluate(): selected.add(req.name)
errors=[]
for name in sorted(selected):
    for declaration in distribution(name).requires or []:
        req=Requirement(declaration)
        if req.marker and not req.marker.evaluate(): continue
        try: current=version(req.name)
        except PackageNotFoundError: current=None
        if current is None or current not in req.specifier:
            errors.append(f'{name} requires {req}; installed {current}')
if errors:
    print('\n'.join(errors)); raise SystemExit(2)
print('Selected project dependency constraints satisfied.')
PYCODE
if [[ "$RTX5090" == 1 ]]; then "$PY" -m visual_training.check_gpu; fi
echo "Dependencies ready in $PY (no new environment created)."
