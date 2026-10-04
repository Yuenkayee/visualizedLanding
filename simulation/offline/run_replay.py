"""Replay chronological recorded RGB/LiDAR; pose truth stays out of estimator."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from navigation.common.contracts import CameraCalibration, CameraFrame, LidarFrame
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from simulation.offline.sensor_buffer import SensorBuffer
from simulation.offline.logger import JsonlLogger


def run_replay(root, sequence, output, config=None):
    root = Path(root)
    paths = sorted((root / "annotations" / sequence).glob("*.json"))
    if not paths:
        raise ValueError("sequence contains no annotations")
    first = json.loads(paths[0].read_text())
    cal = CameraCalibration(
        first["K"],
        first["width"],
        first["height"],
        first.get("distortion", [0] * 5),
        (config or {}).get("T_camera_lidar", np.eye(4)),
    )
    c = config or {}
    state = ESKFState(
        position=np.asarray(c.get("initial_position", [0.0, 0.0, 30.0]), float),
        rotation=np.asarray(c.get("initial_rotation", np.diag([1.0, -1.0, -1.0]))),
    )
    # Geometric model is allowed; ground-truth camera pose is not used to initialise.
    pipe = NavigationPipeline(cal, np.asarray(first["points_deck"]), c, state)
    buffer = SensorBuffer()
    count = 0
    with JsonlLogger(output) as logger:
        for path in paths:
            row = json.loads(path.read_text())
            image = cv2.imread(str(root / row["image"]))
            if image is None:
                raise FileNotFoundError(root / row["image"])
            buffer.push(
                "camera",
                CameraFrame(row["timestamp"], cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            )
            lidar = path.with_name(path.stem + "_lidar.npz")
            if lidar.exists():
                with np.load(lidar) as cloud:
                    buffer.push(
                        "lidar", LidarFrame(float(cloud["timestamp"]), cloud["points"])
                    )
            for kind, frame in buffer.pop_until(row["timestamp"]):
                getattr(pipe, "process_" + kind)(frame)
            estimate = pipe.estimate()
            logger.write(
                dict(
                    timestamp=row["timestamp"],
                    truth=dict(
                        timestamp=row["timestamp"],
                        T_world_deck=np.eye(4).tolist(),
                        T_deck_camera=row["T_deck_camera"],
                        velocity_deck=[0, 0, 0],
                    ),
                    estimate=estimate.to_dict(),
                )
            )
            count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--sequence", required=True)
    p.add_argument("--output", default="outputs/logs/replay.jsonl")
    p.add_argument("--config", default="configs/navigation.yaml")
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    camera = yaml.safe_load(Path("configs/camera.yaml").read_text())
    c["T_camera_lidar"] = camera.get("T_camera_lidar", np.eye(4))
    print(run_replay(a.root, a.sequence, a.output, c))


if __name__ == "__main__":
    main()
