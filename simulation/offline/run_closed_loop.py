"""Actual sensor feedback loop: truth -> rendered sensor -> navigation -> plant."""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from navigation.common.contracts import CameraCalibration, CameraFrame, LidarFrame
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.sensors.lidar_raycast import raycast_deck, beam_directions
from visual_modeling.sensors.lidar_degradation import degrade_lidar
from visual_training.data.build_dataset import synthetic_frame
from simulation.bridge.external_model_adapter import create_adapter
from simulation.offline.scheduler import Scheduler
from simulation.offline.logger import JsonlLogger
from simulation.evaluation.navigation_metrics import navigation_metrics
from simulation.evaluation.closed_loop_metrics import closed_loop_metrics


def run_closed_loop(config=None):
    c = config or {}
    rng = np.random.default_rng(c.get("seed", 42))
    rates = {
        "camera": c.get("camera_hz", 10.0),
        "lidar": c.get("lidar_hz", 5.0),
        "control": c.get("control_hz", 20.0),
    }
    if c.get("disable_vision"):
        rates.pop("camera")
    if c.get("disable_lidar"):
        rates.pop("lidar")
    cal = CameraCalibration.from_config(
        yaml.safe_load(Path(c.get("camera_config", "configs/camera.yaml")).read_text())
    )
    lidar = yaml.safe_load(
        Path(c.get("lidar_config", "configs/lidar.yaml")).read_text()
    )
    directions = beam_directions(
        **{k: lidar[k] for k in ("horizontal_fov", "vertical_fov", "columns", "rows")}
    )
    nav = yaml.safe_load(
        Path(c.get("navigation_config", "configs/navigation.yaml")).read_text()
    )
    nav.update(c.get("navigation", {}))
    init = ESKFState(
        position=np.array(nav.get("initial_position", [0.0, 0.0, 30.0])),
        rotation=np.array(nav.get("initial_rotation", np.diag([1.0, -1.0, -1.0]))),
    )
    marker = c.get("marker", {})
    points = marker_points(marker.get("width", 6), marker.get("length", 6))
    pipe = NavigationPipeline(cal, points, nav, init)
    rows = []
    log = Path(c.get("log", "outputs/logs/closed_loop.jsonl"))
    metrics = Path(c.get("metrics", "outputs/metrics/closed_loop.json"))
    with (
        create_adapter(c.get("backend", "mock"), c.get("plant", {})) as plant,
        JsonlLogger(log) as logger,
    ):
        time = 0.0
        for event in Scheduler(rates, c.get("duration", 20.0)):
            if event.timestamp - time > 1e-10:
                plant.advance(event.timestamp - time)
                time = event.timestamp
            truth = plant.read_sensor_truth()
            if event.kind == "camera":
                image, _ = synthetic_frame(cal, truth.T_deck_camera, rng, marker)
                pipe.process_camera(CameraFrame(time, image))
            elif event.kind == "lidar":
                cloud = degrade_lidar(
                    raycast_deck(
                        truth.T_deck_camera @ cal.T_camera_lidar,
                        directions,
                        max_range=lidar["max_range"],
                    ),
                    rng=rng,
                    **{
                        k: lidar[k]
                        for k in ("range_std", "dropout", "min_range", "max_range")
                    },
                )
                pipe.process_lidar(LidarFrame(time, cloud))
            else:
                pipe.advance(time)
                estimate = pipe.estimate()
                plant.write_navigation_estimate(estimate)
                row = dict(
                    timestamp=time, truth=truth.to_dict(), estimate=estimate.to_dict()
                )
                logger.write(row)
                rows.append(row)
    result = dict(
        navigation=navigation_metrics(rows),
        closed_loop=closed_loop_metrics(
            rows, c.get("plant", {}).get("target_height", 0.8)
        ),
        backend=c.get("backend", "mock"),
        sensor_renderer="cpu_pinhole",
    )
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/offline_simulation.yaml")
    p.add_argument("--backend", choices=["mock", "matlab"])
    p.add_argument("--duration", type=float)
    p.add_argument("--log")
    p.add_argument("--metrics")
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    for name in ("backend", "duration", "log", "metrics"):
        if getattr(a, name) is not None:
            c[name] = getattr(a, name)
    print(json.dumps(run_closed_loop(c), indent=2))


if __name__ == "__main__":
    main()
