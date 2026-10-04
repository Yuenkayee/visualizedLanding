"""CPU pinhole synthetic fallback; Blender renders share the exact same schema."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from navigation.common.contracts import CameraCalibration
from navigation.common.frames import transform
from navigation.pose.solve_pnp import project_points
from visual_modeling.blender.build_h_marker import marker_points, marker_polygons
from visual_modeling.blender.export_annotations import (
    make_annotation,
    export_annotations,
)
from visual_modeling.sensors.camera_degradation import degrade_camera
from visual_modeling.sensors.lidar_raycast import raycast_deck
from visual_modeling.sensors.lidar_degradation import degrade_lidar
from visual_training.data.split_by_sequence import split_by_sequence


def synthetic_frame(calibration, pose, rng, marker=None):
    marker = marker or {}
    width = marker.get("width", 6.0)
    length = marker.get("length", 6.0)
    stroke = marker.get("stroke", 1.2)
    h, w = calibration.height, calibration.width
    image = np.zeros((h, w, 3), np.uint8)
    image[:] = [15, 62, 80]
    image = np.clip(image.astype(float) + rng.normal(0, 3, (h, w, 1)), 0, 255).astype(
        "uint8"
    )
    deck = np.array([[-12, -7, 0], [12, -7, 0], [12, 7, 0], [-12, 7, 0.0]])
    cv2.fillConvexPoly(
        image,
        np.rint(project_points(deck, pose, calibration)).astype("int32"),
        (45, 58, 57),
    )
    mask = np.zeros((h, w), np.uint8)
    for polygon in marker_polygons(width, length, stroke):
        pixels = np.rint(project_points(polygon, pose, calibration)).astype("int32")
        cv2.fillConvexPoly(image, pixels, (245, 245, 240))
        cv2.fillConvexPoly(mask, pixels, 255)
    image = degrade_camera(
        image, rng, noise_std=rng.uniform(0, 3), brightness=rng.uniform(0.85, 1.05)
    )
    return image, mask


def build_dataset(
    output="data",
    sequences=12,
    frames=30,
    seed=42,
    camera_config="configs/camera.yaml",
    marker=None,
):
    if sequences < 3 or frames < 1:
        raise ValueError("need >=3 sequences and >=1 frames")
    root = Path(output)
    rng = np.random.default_rng(seed)
    paths = []
    cal = CameraCalibration.from_config(yaml.safe_load(Path(camera_config).read_text()))
    marker = marker or {}
    points = marker_points(marker.get("width", 6), marker.get("length", 6))
    for s in range(sequences):
        seq = f"synthetic_{seed}_{s:03d}"
        heading = rng.uniform(-0.7, 0.7)
        center = rng.uniform(-2, 2, 2)
        height = rng.uniform(16, 38)
        for i in range(frames):
            pose = transform(
                Rotation.from_euler(
                    "xyz",
                    [
                        np.pi + 0.12 * np.sin(i * 0.09 + s),
                        0.1 * np.cos(i * 0.1 + s),
                        heading + 0.03 * np.sin(i * 0.1),
                    ],
                ).as_matrix(),
                [*(center + 0.3 * np.sin(i * 0.1)), height - 0.03 * i],
            )
            image, mask = synthetic_frame(cal, pose, rng, marker)
            stem = f"{seq}/{i:06d}"
            image_path = f"rendered/{stem}.png"
            mask_path = f"annotations/{stem}_mask.png"
            (root / image_path).parent.mkdir(parents=True, exist_ok=True)
            (root / mask_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(root / image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ) or not cv2.imwrite(str(root / mask_path), mask):
                raise OSError("image write failed")
            path = root / f"annotations/{stem}.json"
            export_annotations(
                path,
                make_annotation(
                    i / 10, seq, i, image_path, mask_path, points, pose, cal
                ),
            )
            paths.append(path)
            cloud = degrade_lidar(raycast_deck(pose @ cal.T_camera_lidar), rng=rng)
            np.savez_compressed(
                root / f"annotations/{stem}_lidar.npz", timestamp=i / 10, points=cloud
            )
    split_by_sequence(paths, root / "splits", seed=seed)
    (root / "calibration").mkdir(parents=True, exist_ok=True)
    (root / "calibration/camera.json").write_text(
        json.dumps(
            dict(
                K=cal.K.tolist(),
                width=cal.width,
                height=cal.height,
                distortion=cal.distortion.tolist(),
                T_camera_lidar=cal.T_camera_lidar.tolist(),
            ),
            indent=2,
        )
    )
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--output")
    p.add_argument("--sequences", type=int)
    p.add_argument("--frames", type=int)
    p.add_argument("--from-rendered", action="store_true")
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    output = a.output or c.get("root", "data")
    if a.from_rendered:
        split_by_sequence(
            Path(output).glob("annotations/**/*.json"),
            Path(output) / "splits",
            seed=c.get("seed", 42),
        )
    else:
        build_dataset(
            output,
            a.sequences or c.get("sequences", 12),
            a.frames or c.get("frames_per_sequence", 30),
            c.get("seed", 42),
            c.get("camera_config", "configs/camera.yaml"),
            c.get("marker"),
        )


if __name__ == "__main__":
    main()
