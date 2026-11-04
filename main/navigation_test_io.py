"""Lightweight experiment I/O, also usable by analysis without Torch/Blender."""
import hashlib
import json
from pathlib import Path

import numpy as np


def clean(value):
    """Strict JSON: absent/nonfinite evidence is null, never a fabricated zero."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(filename)


def digest(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(filename):
    with Path(filename).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
