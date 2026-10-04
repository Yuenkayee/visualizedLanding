"""Whole sequences are assigned once; adjacent frames cannot leak across splits."""

import json
from pathlib import Path
import numpy as np


def split_by_sequence(annotations, output, ratios=(0.7, 0.15, 0.15), seed=42):
    if len(ratios) != 3 or min(ratios) < 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("three nonnegative ratios must sum to one")
    groups = {}
    for path in sorted(map(Path, annotations)):
        row = json.loads(path.read_text())
        groups.setdefault(row["sequence_id"], []).append(str(path.resolve()))
    sequences = sorted(groups)
    np.random.default_rng(seed).shuffle(sequences)
    n = len(sequences)
    if n < 3:
        raise ValueError(
            "at least three independent sequences required for train/val/test"
        )
    counts = np.floor(np.array(ratios) * n).astype(int)
    # Reserve at least one sequence for every requested split.
    counts = np.maximum(counts, np.array(ratios) > 0)
    while counts.sum() > n:
        idx = int(np.argmax(counts))
        counts[idx] -= 1
    while counts.sum() < n:
        idx = int(np.argmax(np.array(ratios) * n - counts))
        counts[idx] += 1
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    offset = 0
    for name, count in zip(("train", "val", "test"), counts):
        paths = [p for s in sequences[offset : offset + count] for p in groups[s]]
        offset += count
        result[name] = paths
        (output / f"{name}.json").write_text(json.dumps(paths, indent=2) + "\n")
    return result
