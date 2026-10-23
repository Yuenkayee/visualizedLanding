"""Replay a saved closed-loop sensor recording without MATLAB or re-rendering.

Recorded encoder/reference telemetry is fixed. Truth is used only for scoring,
including conversion of estimated DECK vectors to NED for plotting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import yaml
from main.config import load, path
from main.run import output_directory, write_json
from main.analyze import analyze
from navigation.common.contracts import CameraCalibration, CameraFrame, LidarFrame, GimbalSample, GimbalReference
from navigation.common.reference_points import body_point_estimate
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from visual_modeling.blender.build_h_marker import marker_points


def replay(source, output, config="main/navigation_observer.yaml", device="cpu", compare_baseline=False):
    import torch
    torch.set_num_threads(4)
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / "run.json").read_text())
    rows = [json.loads(line) for line in (source / "steps.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No recorded iterations")
    c, landing, fallback_cal, _ = load(config)
    camera = original["camera"]
    # Older recordings did not serialize lidar extrinsics. Retain the camera
    # calibration loader for those recordings and explicitly record this limit.
    cal = CameraCalibration(camera["K"], camera["width"], camera["height"],
        camera.get("distortion", fallback_cal.distortion),
        camera.get("T_camera_lidar", fallback_cal.T_camera_lidar),
        camera.get("lidar_mount", fallback_cal.lidar_mount),
        camera.get("T_body_lidar", fallback_cal.T_body_lidar))
    base = original["navigation"]
    current = {**base, **yaml.safe_load(path(c["navigation_config"]).read_text()), **c.get("navigation", {})}
    # Same recorded nominal initial prior for every variant, not aircraft truth.
    for key in ("initial_position", "initial_rotation", "initial_velocity", "filter_point_body", "require_gimbal"):
        if key in base:
            current[key] = base[key]
    checkpoint = path(c["checkpoint"])
    variants = {"current": current}
    if compare_baseline:
        variants["baseline"] = dict(base, roi_recovery=False, mask_fallback_minimum_side_px=0,
                                     reacquisition_confirmations=1)
    geometry = original["geometry"]
    marker = original.get("marker", landing["marker"])
    points = marker_points(marker["width"], marker["length"])
    summaries = {}
    D = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    for name, nav in variants.items():
        nav = {**nav, "checkpoint": str(checkpoint), "device": device}
        state = ESKFState(position=np.array(nav["initial_position"]),
            velocity=np.array(nav.get("initial_velocity", [0., 0., 0.])),
            rotation=np.array(nav["initial_rotation"]), timestamp=rows[0]["timestamp"])
        pipe = NavigationPipeline(cal, points, nav, state)
        target = output / name
        target.mkdir(parents=True, exist_ok=False)
        metadata = {**original, "kind": "recorded_sensor_replay", "source": str(source),
            "status": "running", "navigation": nav, "navigation_applied_to_control": False,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "recorded_gimbal_reference": True, "legacy_calibration_fallback": "lidar_mount" not in camera}
        write_json(target / "run.json", metadata)
        start = time.perf_counter()
        try:
            with (target / "steps.jsonl").open("w", buffering=1) as log:
                for row in rows:
                    i, t = row["iteration"], row["timestamp"]
                    encoder = GimbalSample(**row["estimate"]["gimbal"])
                    pipe.process_gimbal(encoder, t)
                    if row["sensors"]["camera"] or row["sensors"]["lidar"]:
                        filename = source / "sensors" / f"{i:06d}.npz"
                        with np.load(filename, allow_pickle=False) as data:
                            if abs(float(data["timestamp"])-t) > 1e-8:
                                raise ValueError(f"Sensor timestamp mismatch: {filename}")
                            if row["sensors"]["camera"]:
                                pipe.process_camera(CameraFrame(t, data["image"], encoder))
                            if row["sensors"]["lidar"]:
                                pipe.process_lidar(LidarFrame(t, data["points"], encoder))
                    estimate = pipe.estimate()
                    reference = row["estimate"].get("gimbal_reference")
                    if reference:
                        estimate.gimbal_reference = GimbalReference(**reference)
                        pipe.process_gimbal_reference(estimate.gimbal_reference)
                    position, velocity, covariance = body_point_estimate(estimate, geometry["gear_body"])
                    # Scoring/output conversion only: never fed into navigation.
                    R = D @ np.array(row["truth"]["T_world_deck"])[:3, :3]
                    feedback = dict(timestamp=t, valid=estimate.healthy,
                        gear_relative_ned_m=(R @ position).tolist(),
                        velocity_gear_ned_mps=(R @ velocity).tolist(), covariance_ned=(R @ covariance @ R.T).tolist())
                    result = {**row, "estimate": estimate.to_dict(), "feedback": feedback,
                        "gear_estimate_deck_m": pipe.state.position.tolist(),
                        "gear_relative_estimate_ned": feedback["gear_relative_ned_m"],
                        "navigation_applied_to_control": False}
                    log.write(json.dumps(result, allow_nan=False)+"\n")
                    if i % 250 == 0:
                        print(json.dumps(dict(variant=name, time=t, healthy=estimate.healthy,
                                              elapsed_s=time.perf_counter()-start)), flush=True)
            metadata["status"] = "complete"
        except BaseException as exc:
            metadata.update(status="failed", error=str(exc))
            raise
        finally:
            metadata["elapsed_s"] = time.perf_counter()-start
            write_json(target / "run.json", metadata)
        summaries[name] = analyze(target)
    write_json(output / "comparison.json", summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Run directory containing run.json, steps.jsonl and sensors/")
    parser.add_argument("--config", default="main/navigation_observer.yaml")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    destination = output_directory(args.output, "sensor_replay")
    replay(args.source, destination, args.config, args.device, args.compare_baseline)
    print(json.dumps(dict(status="complete", output=str(destination))), flush=True)
