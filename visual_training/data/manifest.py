"""Portable manifests shared by training, migration and sensor replay."""
import json
from pathlib import Path

WEATHERS = ("clear", "fog", "night", "night_fog")


def resolve_annotation(value, root):
    root, value = Path(root), Path(value)
    # Prefer assets inside the selected dataset, even if an old server path exists.
    if "annotations" in value.parts:
        local = root.joinpath(*value.parts[value.parts.index("annotations"):])
        if local.is_file():
            return local
    for candidate in (root / value, value):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Annotation {value} cannot be resolved under {root}")


def load_manifest(split, root, weathers=None):
    records = []
    for value in json.loads(Path(split).read_text()):
        path = resolve_annotation(value, root)
        row = json.loads(path.read_text())
        if weathers is None or row.get("weather") in weathers:
            records.append((path, row))
    if not records:
        raise ValueError(f"Empty dataset split after weather selection: {split}")
    return records
