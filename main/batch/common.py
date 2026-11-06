from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEATHERS = ("clear", "fog", "night", "night_fog")
METHODS = ("C0", "C1", "C2", "C3")


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seeds(master, weather, slot, replacement):
    values = np.random.SeedSequence([master, WEATHERS.index(weather), slot, replacement]).generate_state(3)
    # MATLAB block seeds and rng() support uint32; avoid zero for noise block.
    return dict(zip(("wave_seed", "wind_seed", "sensor_seed"), [int(x % 2147483646)+1 for x in values]))


def fingerprints(base):
    # Hash dirty working-tree contents as well as the git revision. Ignore docs
    # other than the read-only thesis, outputs, environments and .git.
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    selected = {p for p in paths if p and Path(p).suffix in (".py", ".m", ".yaml", ".slx", ".sh", ".txt", ".mat")
                and not p.startswith(("outputs/", "data/"))}
    selected.update([base["checkpoint"], base["model"], "docs/论文.docx"])
    return {p: sha(ROOT / p) for p in sorted(selected) if (ROOT / p).is_file()}


def output_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT / "outputs") or path == ROOT / "outputs":
        raise ValueError("Batch outputs must be a subdirectory of this repository's outputs/")
    return path
