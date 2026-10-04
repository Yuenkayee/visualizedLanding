"""blender --background --python this_file -- --config configs/scene.yaml"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Blender carries numpy but not project dependencies. Add only the selected project
# environment's pure Python/compatible extension packages, never another numpy.
import os

if os.environ.get("LANDING_SITE_PACKAGES"):
    sys.path.append(os.environ["LANDING_SITE_PACKAGES"])
import argparse
import json
import zlib
import numpy as np
import yaml
from navigation.common.contracts import CameraCalibration
from navigation.common.frames import transform
from scipy.spatial.transform import Rotation
from visual_modeling.blender.build_ship import build_ship
from visual_modeling.blender.build_sea_sky import build_sea_sky
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.blender.configure_camera import configure_camera
from visual_modeling.blender.configure_weather import configure_weather
from visual_modeling.blender.apply_poses import apply_poses
from visual_modeling.blender.render_frame import render_frame
from visual_modeling.blender.export_annotations import (
    make_annotation,
    export_annotations,
    keypoint_visibility,
)
from visual_modeling.sensors.lidar_raycast import raycast_scene, beam_directions
from visual_modeling.sensors.lidar_degradation import degrade_lidar


def main(argv=None):
    import bpy

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/scene.yaml"))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--output")
    parser.add_argument("--sequence-id", default="blender_000")
    parser.add_argument("--trajectory")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    c = yaml.safe_load(config_path.read_text())
    # Paths in distributed configs are relative to repository root.
    cam = yaml.safe_load(
        (ROOT / c.get("camera_config", "configs/camera.yaml")).read_text()
    )
    cal = CameraCalibration.from_config(cam)
    lidar = yaml.safe_load(
        (ROOT / c.get("lidar_config", "configs/lidar.yaml")).read_text()
    )
    directions = beam_directions(
        **{k: lidar[k] for k in ("horizontal_fov", "vertical_fov", "columns", "rows")}
    )
    ship_config = c.get("ship", {}).copy()
    if ship_config.get("asset_path"):
        ship_config["asset_path"] = str(ROOT / ship_config["asset_path"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    ship = build_ship(ship_config)
    build_sea_sky(c.get("sea"))
    configure_weather(c.get("weather"))
    camera = configure_camera(cam)
    marker = ship_config.get("marker", {})
    points = marker_points(marker.get("width", 6), marker.get("length", 6))
    out = Path(args.output or ROOT / "data")
    out.mkdir(parents=True, exist_ok=True)
    trajectory = None
    if args.trajectory:
        trajectory = json.loads(Path(args.trajectory).read_text())
    count = args.frames if args.frames is not None else c.get("frames", 100)
    if trajectory is not None:
        count = min(count, len(trajectory))
    # Separate IDs must also produce independent images/poses, otherwise a
    # sequence-level split could still leak identical renderings.
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [c.get("seed", 42), zlib.crc32(args.sequence_id.encode())]
        )
    )
    rate = c.get("fps", 10.0)
    if count < 1 or rate <= 0:
        raise ValueError("frames and fps must be positive")
    for i in range(count):
        t = i / rate
        if trajectory is None:
            T = transform(
                Rotation.from_euler(
                    "xyz",
                    [
                        np.pi + rng.uniform(-0.12, 0.12),
                        rng.uniform(-0.15, 0.15),
                        rng.uniform(-0.35, 0.35),
                    ],
                ).as_matrix(),
                [rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(15, 50)],
            )
            world = transform(
                Rotation.from_euler(
                    "xyz", [0.03 * np.sin(t), 0.02 * np.sin(t * 0.8), 0]
                ).as_matrix(),
                [0, 0, 0.4 * np.sin(t * 0.7)],
            )
        else:
            row = trajectory[i]
            t = row["timestamp"]
            T = np.asarray(row["T_deck_camera"])
            world = np.asarray(row.get("T_world_deck", np.eye(4)))
        apply_poses(ship, camera, world, T)
        bpy.context.view_layer.update()
        stem = f"{args.sequence_id}/{i:06d}"
        image = f"rendered/{stem}.png"
        mask = f"annotations/{stem}_mask.png"
        render_frame(out / image, out / mask, c.get("samples", 32))
        visibility = keypoint_visibility(points, ship, camera)
        export_annotations(
            out / f"annotations/{stem}.json",
            make_annotation(
                t, args.sequence_id, i, image, mask, points, T, cal, visibility
            ),
        )
        cloud = degrade_lidar(
            raycast_scene(
                world @ T @ cal.T_camera_lidar, directions, max_range=lidar["max_range"]
            ),
            rng=rng,
            **{k: lidar[k] for k in ("range_std", "dropout", "min_range", "max_range")},
        )
        np.savez_compressed(
            out / f"annotations/{stem}_lidar.npz", timestamp=t, points=cloud
        )
    path = ROOT / "visual_modeling/assets/scenes"
    path.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path / "frigate_landing.blend"))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
