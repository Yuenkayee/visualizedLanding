"""Actual sensor feedback loop: truth -> rendered sensor -> navigation -> plant."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import yaml
from navigation.common.contracts import CameraCalibration, CameraFrame, LidarFrame
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from navigation.multi_camera_pipeline import MultiCameraPipeline
from navigation.common.camera_rig import CameraRig, gear_feedback_ned
from visual_training.data.generate_multi_camera_dataset import load_config, nominal_rig
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.sensors.lidar_raycast import raycast_deck, beam_directions
from visual_modeling.sensors.lidar_degradation import degrade_lidar
from visual_training.data.build_dataset import synthetic_frame
from simulation.bridge.external_model_adapter import create_adapter
from simulation.offline.scheduler import Scheduler
from simulation.offline.logger import JsonlLogger
from simulation.evaluation.navigation_metrics import navigation_metrics
from simulation.evaluation.closed_loop_metrics import closed_loop_metrics
from simulation.offline.innerloop_alignment import (
    prepare_innerloop_config,
    verify_innerloop_feedback,
    validate_navigation_context,
)


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
    rig = None
    rig_config = None
    gear = None
    if c.get("camera_rig_config"):
        rig_config = load_config(c["camera_rig_config"])
        rig = nominal_rig(rig_config)
        # One LiDAR in the fixed C0 frame, irrespective of selected image source.
        rig = CameraRig.from_dict(rig.to_dict(), nav.get("T_reference_lidar"))
        cal = rig.reference.calibration
        pipe = MultiCameraPipeline(rig, points, nav, init)
    else:
        pipe = NavigationPipeline(cal, points, nav, init)
    plant_config = prepare_innerloop_config(c.get("plant", {}), rig, rig_config)
    innerloop = (
        c.get("backend") == "matlab"
        and plant_config.get("initialize_callback") == "innerloop_initialize"
    )
    if rig is not None:
        gear = plant_config.get("geometry", {}).get(
            "gear_body", rig_config["geometry"]["gear_body_m"]
        )
    rows = []
    log = Path(c.get("log", "outputs/logs/closed_loop.jsonl"))
    metrics = Path(c.get("metrics", "outputs/metrics/closed_loop.json"))
    audit = c.get("audit_log") or (
        str(log.with_name(log.stem + "_audit.jsonl")) if innerloop else None
    )
    counts = dict(
        state_reads=0,
        camera_bundles=0,
        camera_images=0,
        lidar_frames=0,
        visual_updates=0,
        lidar_updates=0,
        feedback_writes=0,
        advances=0,
        checked_slx_feedbacks=0,
    )
    max_conversion_difference = 0.0
    with (
        create_adapter(c.get("backend", "mock"), plant_config) as plant,
        JsonlLogger(log) as logger,
        JsonlLogger(audit) if audit else nullcontext() as auditor,
    ):
        initialization = getattr(plant, "initialization", None)
        time = 0.0
        for events in Scheduler(rates, c.get("duration", 20.0)).groups():
            t = events[0].timestamp
            operations = []
            if t - time > 1e-10:
                # advance() returns the newly read snapshot. Do not read twice.
                truth = plant.advance(t - time)
                counts["advances"] += 1
                operations.append("advance_innerloop")
            else:
                truth = plant.read_sensor_truth()
            time = t
            counts["state_reads"] += 1
            operations.append("read_current_state")
            if abs(truth.timestamp - time) > 1e-8:
                raise ValueError("sensor truth timestamp differs from scheduler time")
            kinds = {e.kind for e in events}
            context = plant.read_navigation_context() if rig is not None else None
            R_ned_deck = validate_navigation_context(context, time) if context else None
            frames = None
            cloud = None
            if "camera" in kinds:
                if rig is None:
                    image, _ = synthetic_frame(cal, truth.T_deck_camera, rng, marker)
                    frames = CameraFrame(time, image)
                    counts["camera_images"] += 1
                else:
                    frames = {}
                    for key, camera in rig.cameras.items():
                        pose = truth.T_deck_camera @ rig.reference_to_camera(key)
                        image, _ = synthetic_frame(
                            camera.calibration, pose, rng, marker
                        )
                        frames[key] = CameraFrame(time, image)
                    counts["camera_images"] += len(frames)
                counts["camera_bundles"] += 1
                operations.append("generate_camera_observations")
            if "lidar" in kinds:
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
                counts["lidar_frames"] += 1
                operations.append("generate_lidar_observations")
            pipe.diagnostics.update(
                vision_accepted=False, lidar_accepted=False, camera_switched=False
            )
            if frames is not None:
                if rig is None:
                    pipe.process_camera(frames)
                else:
                    pipe.process_camera_bundle(frames)
                counts["visual_updates"] += bool(
                    pipe.diagnostics.get("vision_accepted")
                )
            if cloud is not None:
                pipe.process_lidar(LidarFrame(time, cloud))
                counts["lidar_updates"] += bool(pipe.diagnostics.get("lidar_accepted"))
            pipe.advance(time)
            estimate = pipe.estimate()
            operations.append("navigation_solution")
            # Exactly one write at every event time, including sensor-only ticks.
            # A future controller still samples at its own control_hz.
            plant.write_navigation_estimate(estimate)
            counts["feedback_writes"] += 1
            operations.append("write_navigation_feedback")
            row = dict(
                timestamp=time,
                truth=truth.to_dict(),
                estimate=estimate.to_dict(),
                events=[e.kind for e in events],
            )
            if rig is not None:
                expected = gear_feedback_ned(
                    estimate, rig.reference.T_body_camera, gear, R_ned_deck
                )
                if innerloop:
                    raw = plant.read_navigation_feedback()
                    row["feedback"], difference = verify_innerloop_feedback(
                        raw, expected, time
                    )
                    max_conversion_difference = max(
                        max_conversion_difference, difference
                    )
                    counts["checked_slx_feedbacks"] += 1
                    operations.append("read_and_verify_slx_feedback")
                else:
                    row["feedback"] = expected
            if auditor is not None:
                auditor.write(dict(row, operations=operations))
            if "control" in kinds:
                logger.write(row)
                rows.append(row)
    result = dict(
        navigation=navigation_metrics(rows),
        closed_loop=dict(
            mode="interface_state_hold"
            if plant_config.get("allow_state_hold")
            and not plant_config.get("plant_step_callback")
            else "interface_callback",
            dynamics_tested=False,
        )
        if innerloop
        else closed_loop_metrics(rows, plant_config.get("target_height", 0.8)),
        backend=c.get("backend", "mock"),
        sensor_renderer="cpu_pinhole",
        cameras=len(rig.cameras) if rig else 1,
        reference_camera_id=rig.reference_camera_id if rig else "single",
        transactions=counts,
        audit_log=audit,
        innerloop_contract=initialization.get("interface_contract")
        if initialization
        else None,
        max_slx_python_feedback_difference=max_conversion_difference
        if innerloop
        else None,
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
