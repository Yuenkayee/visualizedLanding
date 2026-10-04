#!/usr/bin/env bash
# Optional licensed small-boat reference, NOT a substitute frigate model.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT=41f2df50fbf75b30264eb4c7a2222d4235f6a70b
DEST="$ROOT/visual_modeling/assets/ships/downloads/vrx"
mkdir -p "$DEST"
BASE="https://raw.githubusercontent.com/osrf/vrx/$COMMIT"
for path in LICENSE vrx_gz/models/roboboat01/model.sdf vrx_gz/models/roboboat01/meshes/roboboat01.dae; do
  FILE="$DEST/$path"
  mkdir -p "$(dirname -- "$FILE")"
  curl -fL --retry 3 "$BASE/$path" -o "$FILE"
done
echo "Apache-2.0 reference assets saved to $DEST; see docs/model_sources.md."
