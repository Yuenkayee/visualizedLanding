"""Create a portable four-weather dataset without rerendering or changing splits.

Hard links save space on one filesystem, with copy fallback across filesystems.
No symlinks: the output can be copied independently of the original dataset.
Test mode validates sources only; full mode refuses an existing destination.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil

from visual_training.data.manifest import WEATHERS, load_manifest


def prepare(source, output, mode="test", workers=8):
    source, output = Path(source).resolve(), Path(output).resolve()
    from visual_training.server_train import validate_data
    validate_data(source, formal=mode == "full")
    splits = {s: load_manifest(source / "splits" / f"{s}.json", source, WEATHERS)
              for s in ("train", "val", "test")}
    tasks, sequences = {}, {}
    for split, records in splits.items():
        for path, row in records:
            sequence = row["sequence_id"]
            if sequences.setdefault(sequence, split) != split:
                raise ValueError(f"Sequence leakage: {sequence}")
            for src in (path, path.with_name(path.stem + "_lidar.npz"),
                        source / row["image"], source / row["mask"],
                        source / "trajectories" / f"{sequence}.json"):
                if not src.is_file() or src.stat().st_size == 0:
                    raise FileNotFoundError(src)
                tasks[src] = output / src.relative_to(source)
    summary = dict(status="validated", mode=mode, source=str(source), output=str(output),
                   weathers=WEATHERS, counts={s: len(r) for s, r in splits.items()},
                   sequences=len(sequences), assets=len(tasks), method="hardlink_or_copy")
    if mode == "test":
        # Decode a stratified sample of each weather/split, without writing a dataset.
        import cv2
        import numpy as np
        checked = set()
        for records in splits.values():
            for path, row in records:
                key = (row["split"], row["weather"], row["stage"])
                if key in checked:
                    continue
                for field in ("image", "mask"):
                    if cv2.imread(str(source / row[field]), cv2.IMREAD_UNCHANGED) is None:
                        raise ValueError(f"Cannot decode {row[field]}")
                with np.load(path.with_name(path.stem + "_lidar.npz"), allow_pickle=False) as lidar:
                    if not np.isfinite(lidar["points"]).all():
                        raise ValueError(f"Nonfinite LiDAR: {path}")
                checked.add(key)
        summary["decoded_samples"] = len(checked)
        return summary
    output.mkdir(parents=True, exist_ok=False)
    (output / "preparation.json").write_text(json.dumps(dict(summary, status="running")) + "\n")
    def transfer(item):
        src, dst = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(transfer, tasks.items()))
    (output / "splits").mkdir()
    for split, records in splits.items():
        (output / "splits" / f"{split}.json").write_text(json.dumps(
            [str(path.relative_to(source)) for path, _ in records], indent=2) + "\n")
    summary["status"] = "complete"
    # Validate before publishing the completion marker; an interrupted copy is
    # rejected by the training launcher on subsequent attempts.
    validate_data(output, formal=True, allow_incomplete_preparation=True)
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "SUCCESS.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("test", "full"))
    p.add_argument("--source", default="data/landing")
    p.add_argument("--output", default="data/landing_four_weather")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    print(json.dumps(prepare(a.source, a.output, a.mode, a.workers), indent=2), flush=True)


if __name__ == "__main__":
    main()
