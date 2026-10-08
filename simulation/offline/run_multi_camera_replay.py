"""Synchronized multi-camera/LiDAR replay; pose labels are used only for metrics."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from navigation.common.camera_rig import CameraRig, gear_feedback_ned
from navigation.common.contracts import CameraFrame, LidarFrame
from navigation.fusion.eskf_state import ESKFState
from navigation.multi_camera_pipeline import MultiCameraPipeline
from simulation.offline.logger import JsonlLogger


def run_multi_replay(root, sequence, output, config=None):
    root = Path(root)
    c = config or {}
    bundles = json.loads((root / "bundles" / f"{sequence}.json").read_text())
    calibration = json.loads((root / "calibration" / f"{sequence}.json").read_text())
    if (
        "T_reference_lidar" in calibration
        and "T_reference_lidar" in c
        and not np.allclose(calibration["T_reference_lidar"], c["T_reference_lidar"])
    ):
        raise ValueError("replay LiDAR extrinsic differs from dataset calibration")
    rig = CameraRig.from_dict(calibration, c.get("T_reference_lidar"))
    state = ESKFState(
        position=np.asarray(c.get("initial_position", [0, 0, 30]), float),
        rotation=np.asarray(c.get("initial_rotation", np.diag([1, -1, -1])), float),
    )
    first = json.loads(
        (root / bundles[0]["annotations"][rig.reference_camera_id]).read_text()
    )
    pipe = MultiCameraPipeline(rig, first["points_deck"], c, state)
    errors, jumps, accepted, previous, switches = [], [], 0, None, 0
    with JsonlLogger(output) as logger:
        for bundle in sorted(bundles, key=lambda b: b["timestamp"]):
            frames, labels = {}, {}
            for key, path in bundle["annotations"].items():
                row = labels[key] = json.loads((root / path).read_text())
                image = cv2.imread(str(root / row["image"]))
                if image is None:
                    raise FileNotFoundError(root / row["image"])
                frames[key] = CameraFrame(
                    row["timestamp"], cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )
            pipe.process_camera_bundle(frames)
            if bundle.get("lidar") and not c.get("disable_lidar", False):
                with np.load(root / bundle["lidar"]) as cloud:
                    pipe.process_lidar(
                        LidarFrame(float(cloud["timestamp"]), cloud["points"])
                    )
            estimate = pipe.estimate()
            # Ship attitude is the stipulated zero-delay communication channel.
            R = Rotation.from_euler(
                "xyz", bundle["ship_euler_rad"]
            ).as_matrix() @ np.diag([1, -1, -1])
            label = labels[rig.reference_camera_id]
            feedback = gear_feedback_ned(
                estimate,
                rig.reference.T_body_camera,
                label["geometry"]["gear_body_m"],
                R,
            )
            truth = np.asarray(label["gear_relative_ned_m"])
            relative = np.asarray(feedback["gear_relative_estimate_ned"])
            if estimate.healthy:
                errors.append((relative - truth).tolist())
            if estimate.diagnostics.get("camera_switched"):
                switches += 1
                if previous is not None:
                    jumps.append(float(np.linalg.norm(relative - previous)))
            previous = relative
            accepted += bool(estimate.diagnostics.get("vision_accepted"))
            logger.write(
                dict(
                    timestamp=estimate.timestamp,
                    estimate=estimate.to_dict(),
                    feedback=feedback,
                    truth_gear_relative_ned=truth.tolist(),
                    stage=bundle["stage"],
                    weather=bundle["weather"],
                )
            )
    return dict(
        bundles=len(bundles),
        accepted_visual_updates=accepted,
        switches=switches,
        healthy_ned_rmse_m=np.sqrt(np.mean(np.square(errors), axis=0)).tolist()
        if errors
        else None,
        max_switch_step_m=max(jumps) if jumps else None,
        note="Switch step includes motion between samples; not a pure switch-induced discontinuity",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/multi_camera")
    p.add_argument("--sequence", required=True)
    p.add_argument("--config", default="configs/multi_camera_navigation.yaml")
    p.add_argument("--output", default="outputs/logs/multi_camera_replay.jsonl")
    a = p.parse_args()
    print(
        json.dumps(
            run_multi_replay(
                a.root, a.sequence, a.output, yaml.safe_load(Path(a.config).read_text())
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
