"""Run read -> sensors -> navigation -> feedback -> Simulink step -> next state."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
import yaml
from main.config import ROOT, load, path, validate
from main.analyze import analyze
from main.sensors import Sensors
from navigation.common.contracts import CameraFrame, LidarFrame
from navigation.common.gimbal import Gimbal
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from simulation.bridge.external_model_adapter import MatlabExternalModel
from simulation.bridge.matlab_session import MatlabSession
from visual_modeling.blender.build_h_marker import marker_points


def write_json(filename, value):
    Path(filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def output_directory(value=None, kind="run"):
    result = (
        path(value)
        if value
        else ROOT
        / "outputs/closed_loop"
        / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    result.mkdir(parents=True, exist_ok=False)
    return result


class LivePlant(MatlabExternalModel):
    def __enter__(self):
        self.session.__enter__()
        try:
            self.session.engine.eval(
                "Simulink.fileGenControl('set','CacheFolder',fullfile(tempdir,'landing_live_cache'),'CodeGenFolder',fullfile(tempdir,'landing_live_cache','codegen'),'createDir',true);",
                nargout=0,
            )
            if self.config.get("interface_fixture"):
                self.config["model"] = self.session.call(
                    "landing_build_interface_fixture",
                    self.config["workdir"],
                    self.config["geometry"],
                )
            self.session.call("initialize_external_model", self.config)
        except BaseException:
            self.session.__exit__()
            raise
        return self


def run(c, landing, cal, geometry, output, smoke=False):
    validate(c)
    checkpoint = c.get("checkpoint")
    if checkpoint:
        checkpoint = path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    elif not smoke:
        raise ValueError(
            "Production run requires --checkpoint; smoke alone permits the existing classical detector"
        )
    if not smoke and not path(c["model"]).is_file():
        raise FileNotFoundError(c["model"])
    nav = yaml.safe_load(path(c["navigation_config"]).read_text())
    nav.update(c.get("navigation", {}))
    nav.update(
        checkpoint=str(checkpoint) if checkpoint else None,
        device=c["device"],
        filter_point_body=geometry["gear_body"],
        require_gimbal=True,
    )
    # configs/navigation.yaml contains a legacy camera orientation. Use the
    # training/gimbal reference unless explicitly overridden for this scenario.
    nav["initial_rotation"] = c.get("navigation", {}).get(
        "initial_rotation",
        Gimbal(geometry["T_body_camera"], landing.get("gimbal")).reference.tolist(),
    )
    import torch

    torch.set_num_threads(int(c.get("torch_threads", 4)))
    init = ESKFState(
        position=np.asarray(nav["initial_position"]),
        velocity=np.asarray(nav.get("initial_velocity", [0.0, 0.0, 0.0])),
        rotation=np.asarray(nav["initial_rotation"]),
    )
    pipe = NavigationPipeline(
        cal,
        marker_points(landing["marker"]["width"], landing["marker"]["length"]),
        nav,
        init,
    )
    lidar = yaml.safe_load(path(c["lidar_config"]).read_text())
    work = output / "simulink"
    work.mkdir()
    plant_config = dict(
        initialize_callback="landing_live_initialize",
        geometry=geometry,
        gimbal=landing.get("gimbal", {}),
        model=str(path(c["model"])),
        inputs=c["inputs"],
        outputs=c["outputs"],
        workdir=str(work),
        plant_step=c["plant_step"],
        interface_fixture=smoke,
        setup_script=str(path(c["setup_script"])) if c.get("setup_script") else "",
        seed=c["seed"],
    )
    checkpoint_hash = None
    if checkpoint:
        with checkpoint.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = dict(
        status="running",
        kind="interface_acceptance" if smoke else "aircraft_closed_loop",
        config=c,
        geometry=geometry,
        navigation=nav,
        checkpoint_sha256=checkpoint_hash,
        truth_policy="Truth only drives sensor synthesis and evaluation; no truth poses enter NavigationPipeline.",
        order=[
            "read_current",
            "generate_sensors",
            "solve_navigation",
            "write_feedback",
            "advance_innerloop",
            "read_next",
        ],
    )
    write_json(output / "run.json", metadata)
    start = time.perf_counter()
    try:
        with (
            LivePlant(plant_config) as plant,
            Sensors(c, landing, lidar, output) as sensors,
            (output / "steps.jsonl").open("w", buffering=1) as log,
            (output / "events.jsonl").open("w", buffering=1) as events,
        ):
            metadata["sensors"] = sensors.metadata
            metadata["matlab"] = plant.session.engine.version()
            metadata["model_used"] = plant.config["model"]
            write_json(output / "run.json", metadata)

            def event(phase, index, t):
                events.write(
                    json.dumps(dict(iteration=index, phase=phase, timestamp=t)) + "\n"
                )

            n = round(c["duration"] / c["dt"])
            camera_period = round(1 / (c["dt"] * c["camera_hz"]))
            lidar_period = round(1 / (c["dt"] * c["lidar_hz"]))
            for i in range(n):
                wall = time.perf_counter()
                truth = plant.read_sensor_truth()
                t = truth.timestamp
                if abs(t - i * c["dt"]) > 1e-8:
                    raise ValueError("Plant/host clock mismatch")
                event("read_current", i, t)
                encoder = plant.read_gimbal()
                has_camera, has_lidar = i % camera_period == 0, i % lidar_period == 0
                observed = sensors.observe(truth, has_camera, has_lidar, i)
                event("generate_sensors", i, t)
                pipe.process_gimbal(encoder, t)
                if has_camera:
                    pipe.process_camera(CameraFrame(t, observed["image"], encoder))
                if has_lidar:
                    pipe.process_lidar(LidarFrame(t, observed["points"], encoder))
                estimate = pipe.estimate()
                event("solve_navigation", i, t)
                plant.write_navigation_estimate(estimate)
                feedback = plant.read_navigation_feedback()
                event("write_feedback", i, t)
                event("advance_innerloop", i, t)
                following = plant.advance(c["dt"])
                if abs(following.timestamp - t - c["dt"]) > 1e-8:
                    raise ValueError("Inner loop failed to advance exactly dt")
                event("read_next", i, following.timestamp)
                row = dict(
                    iteration=i,
                    timestamp=t,
                    dt=c["dt"],
                    truth=truth.to_dict(),
                    estimate=estimate.to_dict(),
                    gear_estimate_deck_m=pipe.state.position.tolist(),
                    feedback=feedback,
                    sensors=dict(camera=has_camera, lidar=has_lidar),
                    next_timestamp=following.timestamp,
                    next_truth=following.to_dict(),
                    wall_s=time.perf_counter() - wall,
                )
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if i % max(1, round(1 / c["dt"])) == 0 or i == n - 1:
                    print(
                        json.dumps(
                            dict(
                                event="step",
                                iteration=i,
                                time=following.timestamp,
                                feedback_valid=feedback["valid"],
                                wall_s=row["wall_s"],
                            )
                        ),
                        flush=True,
                    )
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_s"] = time.perf_counter() - start
        write_json(output / "run.json", metadata)
        if (output / "steps.jsonl").exists() and (
            output / "steps.jsonl"
        ).stat().st_size:
            try:
                analyze(output)
            except Exception as exc:
                write_json(output / "analysis_error.json", dict(error=str(exc)))
                if metadata["status"] == "complete":
                    raise
    return dict(
        status=metadata["status"],
        output=str(output),
        kind=metadata["kind"],
        elapsed_s=metadata["elapsed_s"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "smoke", "configure", "inspect"])
    p.add_argument("--config", default="main/closed_loop.yaml")
    p.add_argument("--model")
    p.add_argument("--checkpoint")
    p.add_argument("--output")
    p.add_argument("--duration", type=float)
    p.add_argument("--renderer", choices=["cpu", "blender"])
    p.add_argument("--render-device", choices=["CPU", "CUDA", "OPTIX"])
    p.add_argument("--device", choices=["cpu", "cuda", "mps"])
    p.add_argument("--render-python")
    p.add_argument("--render-width", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument("--weather", choices=["clear", "fog", "night", "night_fog"])
    p.add_argument("--save-sensors", action="store_true")
    a = p.parse_args()
    c = yaml.safe_load(path(a.config).read_text())
    for name in (
        "model",
        "checkpoint",
        "duration",
        "renderer",
        "render_device",
        "device",
        "render_python",
        "render_width",
        "samples",
        "weather",
    ):
        if getattr(a, name) is not None:
            c[name] = getattr(a, name)
    if a.save_sensors:
        c["save_sensors"] = True
    if a.command == "smoke":
        if a.duration is None:
            c["duration"] = 1.0
        if a.renderer is None:
            c["renderer"] = "cpu"
    # Resolve optics after CLI overrides, without modifying source config.
    output = output_directory(a.output, a.command)
    write_json(output / "requested.json", c)
    c, landing, cal, geometry = load(output / "requested.json")
    print(f"Output: {output}", flush=True)
    if a.command == "configure":
        backup = output / "backups"
        backup.mkdir()
        models = (
            [str(path(a.model))]
            if a.model
            else [
                str(ROOT / "external/innerloop/simu" / name)
                for name in ("innerLoop.slx", "innerLoop_landing.slx")
            ]
        )
        with MatlabSession() as session:
            result = session.call(
                "landing_configure_geometry",
                dict(models=models, geometry=geometry, backup_dir=str(backup)),
            )
    elif a.command == "inspect":
        with MatlabSession() as session:
            result = session.call(
                "landing_audit_model",
                dict(
                    model=str(path(c["model"])),
                    inputs=c["inputs"],
                    outputs=c["outputs"],
                ),
            )
    else:
        result = run(c, landing, cal, geometry, output, smoke=a.command == "smoke")
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
