"""Scenario planning and immutable sensor records for navigation experiments.

Trajectory truth is confined to synthesis/scoring. A fixed N3 observer commands
the camera during recording; no navigation output can change the aircraft path.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from main.config import path
from main.sensors import Sensors
from navigation.common.contracts import CameraFrame, LidarFrame
from navigation.common.frames import invert, transform
from navigation.common.gimbal import Gimbal
from navigation.common.gimbal_tracking import GimbalTargetTracker
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from simulation.bridge.exchange_contracts import SensorTruth
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.trajectories.landing import make_sortie

NED_FROM_WORLD = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
STAGE_NAMES = {"window_hold": "hover", "touchdown": "descent"}


def clean(value):
    """Strict JSON: absent/nonfinite evidence is null, never a fabricated zero."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(filename)


def digest(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cases(config, smoke=False):
    """Stable seeds do not depend on method ordering, scheduling or recovery."""
    result = []
    suite = config.get("suite", "all")
    for wi, weather in enumerate(config["weather"]):
        for mi, motion in enumerate(config["motion_scales"]):
            for replicate in range(config["seeds_per_cell"]):
                seed = int(np.random.SeedSequence([config["seed"], wi, mi, replicate]).generate_state(1)[0])
                pair = f"{weather}_{motion}_{replicate:03d}"
                base = dict(pair_id=pair, weather=weather, motion=motion, seed=seed,
                            replicate=replicate, gimbal_mode="tracking", condition="baseline",
                            excluded_from_formal=smoke)
                # Baselines also supply paired references for the special studies.
                result.append(dict(base, id=pair, record_id=pair, suite="main",
                                   methods=config["methods"] if suite in ("main", "all") else ["N3"]))
                if suite in ("gimbal", "all"):
                    result.extend([
                        dict(base, id=pair+"_tracking", record_id=pair, suite="gimbal", methods=["N3"]),
                        dict(base, id=pair+"_fixed", record_id=pair+"_fixed", suite="gimbal",
                             gimbal_mode="fixed", methods=["N3"])])
                if suite in ("stress", "all") and replicate < config["stress_seeds_per_cell"]:
                    for fault in config["stress"]:
                        result.append(dict(base, id=pair+"_"+fault["name"], record_id=pair,
                                           suite="stress", condition=fault["name"], fault=fault, methods=["N3"]))
    return result


def camera_dict(cal):
    return dict(K=cal.K, width=cal.width, height=cal.height, distortion=cal.distortion,
                T_camera_lidar=cal.T_camera_lidar, lidar_mount=cal.lidar_mount,
                T_body_lidar=cal.T_body_lidar)


def navigation_config(base, landing, geometry, checkpoint, device, recorded=None):
    nav = yaml.safe_load(path(base["navigation_config"]).read_text())
    nav.update(base.get("navigation", {}))
    nav.update(checkpoint=str(checkpoint), device=device, filter_point_body=geometry["gear_body"], require_gimbal=True)
    if recorded is None:
        # Prescribed mission prior, BEFORE any randomized truth is generated.
        nav["initial_position"] = list(landing["mission"]["initial_gear_heading_m"])
        nav["initial_velocity"] = [0., 0., 0.]
        nav["initial_rotation"] = Gimbal(geometry["T_body_camera"], landing.get("gimbal")).reference.tolist()
    else:
        # Preserve the prior and all original runtime navigation settings.
        nav = {**recorded, "checkpoint": str(checkpoint), "device": device}
    return nav


def state_from_config(nav, timestamp=0.):
    return ESKFState(position=np.asarray(nav["initial_position"]),
                     velocity=np.asarray(nav.get("initial_velocity", [0., 0., 0.])),
                     rotation=np.asarray(nav["initial_rotation"]), timestamp=timestamp)


def scale_motion(landing, factor):
    c = deepcopy(landing)
    for key in ("ship_roll_deg", "ship_pitch_deg", "ship_heave_m"):
        c["motion"][key] = (np.asarray(c["motion"][key]) * factor).tolist()
    c["motion"]["camera_vibration_deg"] *= factor
    for key in ("approach_start_amplitude_rad", "approach_end_amplitude_rad", "hold_amplitude_rad", "touchdown_amplitude_rad"):
        c["motion"]["attitude"][key] = (np.asarray(c["motion"]["attitude"][key]) * factor).tolist()
    return c


def read_jsonl(filename):
    with Path(filename).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def source_provenance(source):
    """A saved fragment or CPU preview cannot become a formal flight by import."""
    source=Path(source)
    meta=json.loads((source/"run.json").read_text())
    if meta.get("status")!="complete":
        raise ValueError("Source must be a complete recording; failed/interrupted fragments are not full flights")
    planned=meta.get("planned_duration_s")
    completed=meta.get("completed_duration_s")
    if planned is not None and (completed is None or completed < planned-1e-8):
        raise ValueError("Source recording does not cover its planned duration")
    rows=list(read_jsonl(source/"steps.jsonl"))
    if not rows:
        raise ValueError("Source has no recorded steps")
    original_case=meta.get("case",{})
    original_config=meta.get("config",{})
    labels=stage_labels([r["truth"]["gear_position_deck"] for r in rows],
                        [r["timestamp"] for r in rows],5.)
    stages={STAGE_NAMES.get(r.get("stage"),r.get("stage",labels[i])) for i,r in enumerate(rows)}
    full_mission={"approach","hover","descent"}<=stages
    excluded=bool(meta.get("excluded_from_formal") or original_case.get("excluded_from_formal")
        or meta.get("kind")=="interface_acceptance" or meta.get("renderer",original_config.get("renderer"))=="cpu"
        or not full_mission)
    return dict(excluded_from_formal=excluded,source_full_mission_coverage=full_mission,
        source_recording_status=meta["status"],source_original_case=original_case,
        source_renderer=meta.get("renderer",original_config.get("renderer")),
        source_weather=original_case.get("weather",original_config.get("weather","unknown")),
        source_seed=original_case.get("seed",original_config.get("seed",0)),
        source_stop_reason=meta.get("stop_reason",meta.get("termination_reason")))


def stage_labels(gears, times, hover_height):
    """Frozen geometric event definition for legacy actual trajectories."""
    labels, entered, descending = [], False, False
    for p in gears:
        if np.linalg.norm(p[:2]) <= 3. and p[2] <= hover_height + 1.:
            entered = True
        if entered and p[2] < hover_height - .5:
            descending = True
        labels.append("descent" if descending else "hover" if entered else "approach")
    return labels


def trajectory(landing, case, dt, duration=None, source=None):
    """Interpolate BODY/DECK truth only; never interpolate sensor observations."""
    c = scale_motion(landing, case["motion_scale"])
    if source is None:
        c["trajectory_hz"] = 1 / dt
        raw = make_sortie(c, case["seed"], case["pair_id"], case["weather"])
        rows = raw["rows"]
        times = np.array([r["timestamp"] for r in rows])
        bodies = np.array([r["T_world_body"] for r in rows])
        decks = np.array([r["T_world_deck"] for r in rows])
        labels = [STAGE_NAMES.get(r["stage"], r["stage"]) for r in rows]
        description = "prescribed kinematic landing; no flight dynamics or MATLAB; original training motion family with independent test seeds"
    else:
        source_provenance(source)
        rows = list(read_jsonl(Path(source) / "steps.jsonl"))
        times = np.array([r["timestamp"] for r in rows])
        decks = np.array([r["truth"]["T_world_deck"] for r in rows])
        bodies = np.array([decks[i] @ np.asarray(r["truth"]["T_deck_camera"]) @ invert(r["truth"]["T_body_camera"]) for i,r in enumerate(rows)])
        gears = [np.asarray(r["truth"]["gear_position_deck"]) for r in rows]
        labels = stage_labels(gears, times, landing["mission"]["hover_height_m"])
        # Preserve the actual path and add smooth independent motion residuals.
        # No circular time shift: that would introduce a discontinuity at the
        # wrap point and corrupt the high-dynamic test.
        rng = np.random.default_rng(np.random.SeedSequence([case["seed"], 191]))
        factor = case["motion_scale"]
        for poses, amplitude in ((decks,np.deg2rad([landing["motion"]["ship_roll_deg"][1],landing["motion"]["ship_pitch_deg"][1],0.])),
                                  (bodies,np.asarray(landing["motion"]["attitude"]["hold_amplitude_rad"]))):
            rotations = Rotation.from_matrix(poses[:, :3, :3])
            mean = rotations.mean()
            residual = (mean.inv() * rotations).as_rotvec()
            phase=rng.uniform(0,2*np.pi,3);period=rng.uniform(*landing["motion"]["wave_period_s"],3)
            wave=np.sin(2*np.pi*(times-times[0])[:,None]/period+phase)-np.sin(phase)
            poses[:, :3, :3] = (mean * Rotation.from_rotvec(factor*(residual+.25*wave*amplitude))).as_matrix()
        phase=rng.uniform(0,2*np.pi);period=rng.uniform(*landing["motion"]["wave_period_s"])
        decks[:,2,3] += .25*factor*landing["motion"]["ship_heave_m"][1]*(np.sin(2*np.pi*(times-times[0])/period+phase)-np.sin(phase))
        description = "recorded BODY/DECK path with scaled attitude excursions and 25%-amplitude independent attitude/heave residuals; no control rerun"
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory must contain at least two strictly increasing timestamps")
    start = float(times[0])
    end = float(times[-1]) if duration is None else min(float(times[-1]), start + duration)
    count = int(np.floor((end - start) / dt + 1e-7))
    if count < 1:
        raise ValueError("Trajectory is shorter than one host step")
    t = start + np.arange(count + 1) * dt
    def sample(poses):
        out = np.tile(np.eye(4), (len(t), 1, 1))
        out[:, :3, :3] = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(t).as_matrix()
        for j in range(3):
            out[:, j, 3] = np.interp(t, times, poses[:, j, 3])
        return out
    indices = np.minimum(np.searchsorted(times, t, side="right") - 1, len(labels)-1)
    return t, sample(bodies), sample(decks), [labels[i] for i in indices], c, description


def generate_record(target, case, config, base, landing, cal, geometry, nav, duration=None, source=None):
    """Record once; every algorithm sees byte-identical archived observations."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    dt = config["dt"]
    times, bodies, decks, stages, scene, description = trajectory(landing, case, dt, duration, source)
    options = {**base, "dt": dt, "seed": case["seed"], "weather": case["weather"],
               "renderer": config["renderer"], "render_device": config["render_device"],
               "render_threads": config["render_threads"], "save_sensors": True,
               "save_sensor_labels": True, "sensor_seed_by_tick": True}
    if config.get("render_python"):
        options["render_python"] = config["render_python"]
    servo_cfg = deepcopy(scene.get("gimbal", {}))
    servo_cfg.setdefault("tracking", {})["enabled"] = case["gimbal_mode"] != "fixed"
    servo = Gimbal(geometry["T_body_camera"], servo_cfg)
    tracker = GimbalTargetTracker(geometry["T_body_camera"], servo.config)
    controller = NavigationPipeline(cal, marker_points(scene["marker"]["width"], scene["marker"]["length"]),
                                    nav, state_from_config(nav, times[0]))
    lidar = yaml.safe_load(path(base["lidar_config"]).read_text())
    meta = dict(status="running", case=case, navigation=nav, geometry=geometry,
                camera=camera_dict(cal), marker=scene["marker"], cg_signal=base.get("cg_signal", {}),
                dt=dt, planned_duration_s=float(times[-1]-times[0]), start_time_s=float(times[0]),
                trajectory_source=description, motion=scene["motion"], gimbal=servo.config,
                renderer=config["renderer"], navigation_applied_to_control=False,
                execution=config.get("execution",{}),
                attempt_id=config.get("attempt_id"),runtime=config.get("runtime"),
                camera_reference_source="causal N3 observer (fixed during method replay)",
                noise_policy="independent SeedSequence per physical tick and channel")
    write_json(target / "run.json", meta)
    camera_period, lidar_period = [round(1/(dt*config[k])) for k in ("camera_hz", "lidar_hz")]
    previous_pose = previous_gear = None
    started = time.perf_counter()
    try:
        with Sensors(options, scene, lidar, target) as sensors, (target / "steps.jsonl").open("w", buffering=1) as f:
            meta["render_environment"] = sensors.metadata
            for i,t in enumerate(times[:-1]):
                wall = time.perf_counter()
                actual = servo.advance(float(t), bodies[i,:3,:3], decks[i,:3,:3])
                pose = invert(decks[i]) @ bodies[i] @ actual.T_body_camera
                body_deck = invert(decks[i]) @ bodies[i]
                gear = (body_deck @ np.r_[geometry["gear_body"], 1])[:3]
                velocity = np.zeros(3) if previous_pose is None else (pose[:3,3] - previous_pose[:3,3])/dt
                gear_v = np.zeros(3) if previous_gear is None else (gear - previous_gear)/dt
                truth = SensorTruth(float(t), decks[i], pose, velocity, actual.T_body_camera, gear, gear_v)
                encoder = servo.sample(np.random.default_rng(np.random.SeedSequence([case["seed"],i,7919])))
                camera, laser = i % camera_period == 0, i % lidar_period == 0
                rs = time.perf_counter()
                observed = sensors.observe(truth, camera, laser, i)
                render_s = time.perf_counter()-rs
                controller.process_gimbal(encoder, float(t))
                if camera:
                    controller.process_camera(CameraFrame(float(t), observed["image"], encoder))
                if laser:
                    controller.process_lidar(LidarFrame(float(t), observed["points"], encoder))
                estimate = controller.estimate()
                reference_used = servo.reference.copy()
                command = tracker.apply(estimate)
                controller.process_gimbal_reference(command)
                servo.set_reference(command)
                omega = (np.zeros(3) if i == 0 else Rotation.from_matrix(decks[i-1,:3,:3].T @ decks[i,:3,:3]).as_rotvec()/dt)
                row = dict(iteration=i, timestamp=float(t), dt=dt, stage=stages[i], truth=truth.to_dict(),
                    gimbal=encoder.to_dict(), gimbal_reference=command.to_dict(),
                    ship_rotation_ned_deck=NED_FROM_WORLD @ decks[i,:3,:3],
                    ship_angular_velocity_deck=omega, sensors=dict(camera=camera,lidar=laser),
                    sensor_file=f"sensors/{i:06d}.npz", gimbal_tracking=tracker.diagnostics,
                    gimbal_reference_error_rad=float(Rotation.from_matrix(reference_used.T @ pose[:3,:3]).magnitude()),
                    timing=dict(render_s=render_s, generation_total_s=time.perf_counter()-wall, matlab_s=None))
                f.write(json.dumps(clean(row), allow_nan=False)+"\n")
                previous_pose, previous_gear = pose, gear
                if i % 250 == 0:
                    print(json.dumps(dict(event="record", case=case["id"], time=float(t),
                        duration=meta["planned_duration_s"], wall_s=time.perf_counter()-started)),flush=True)
        meta.update(status="complete", completed_duration_s=meta["planned_duration_s"], rows=len(times)-1)
        write_json(target / "sensor_hashes.json", {p.name:digest(p) for p in sorted((target/"sensors").glob("*.npz"))})
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta["wall_s"] = time.perf_counter()-started
        write_json(target / "run.json", meta)
        if meta["status"]=="complete":
            write_json(target/"record_hashes.json",{name:digest(target/name) for name in ("run.json","steps.jsonl","sensor_hashes.json")})
    return meta


def import_record(source, target, base, duration=None):
    """Index existing records once; never relabel one flight as independent seeds."""
    source, target = Path(source).resolve(), Path(target)
    original = json.loads((source/"run.json").read_text())
    provenance=source_provenance(source)
    if original.get("status") != "complete":
        raise ValueError("Source recording must be complete; partial runs need separate failure analysis")
    rows = list(read_jsonl(source/"steps.jsonl"))
    if not rows:
        raise ValueError("Source has no steps")
    times = np.array([r["timestamp"] for r in rows]);dt=float(rows[0].get("dt", .02))
    if not np.allclose(np.diff(times), .02, atol=1e-8) or abs(dt-.02)>1e-8:
        raise ValueError("Source must be chronological at the required 0.02 s host clock")
    if duration is not None:
        rows = [r for r in rows if r["timestamp"] < times[0]+duration-1e-8]
    # A STOP read at the same time as its predecessor's endpoint contributes
    # no elapsed interval. Retain its reason in metadata, not a fictitious dt.
    rows = [r for r in rows if float(r.get("advanced_s",r.get("next_timestamp",r["timestamp"]+dt)-r["timestamp"]))>1e-10]
    if not rows:
        raise ValueError("Source contains no positive-duration navigation intervals")
    labels = stage_labels([r["truth"]["gear_position_deck"] for r in rows], times, 5.)
    target.mkdir(parents=True)
    checks = {}
    with (target/"steps.jsonl").open("w") as stream:
        for i,r in enumerate(rows):
            filename=source/"sensors"/f"{r['iteration']:06d}.npz"
            if r["sensors"]["camera"] or r["sensors"]["lidar"]:
                if not filename.is_file():
                    raise FileNotFoundError(f"Source requires saved sensors: {filename}")
                checks[str(filename)] = digest(filename)
            encoder=r.get("gimbal",r.get("estimate",{}).get("gimbal"))
            if encoder is None:
                raise ValueError("Source lacks synchronized encoder history")
            interval=float(r.get("advanced_s",r.get("next_timestamp",r["timestamp"]+dt)-r["timestamp"]))
            if not 0 <= interval <= dt+1e-8:
                raise ValueError("Invalid recorded physical time interval")
            record=dict(r,dt=interval, stage=STAGE_NAMES.get(r.get("stage"), r.get("stage",labels[i])),
                gimbal=encoder, gimbal_reference=r.get("gimbal_reference",r.get("estimate",{}).get("gimbal_reference")),
                sensor_file=str(filename), ship_rotation_ned_deck=NED_FROM_WORLD @ np.asarray(r["truth"]["T_world_deck"])[:3,:3],
                timing={"source_total_s":r.get("wall_s"),"render_s":None,"matlab_s":None})
            stream.write(json.dumps(clean(record),allow_nan=False)+"\n")
    cg=original.get("cg_signal",{})
    cg=cg.get("parameters",cg) if cg else base.get("cg_signal",{})
    meta=dict(original, **provenance, status="complete", source=str(source), dt=dt,
        cg_signal=cg,
        start_time_s=float(times[0]), planned_duration_s=rows[-1]["timestamp"]-times[0]+float(rows[-1].get("advanced_s",dt)),
        completed_duration_s=rows[-1]["timestamp"]-times[0]+float(rows[-1].get("advanced_s",dt)),
        trajectory_source="one recorded actual flight; fixed observation replay, not independent scene replication",
        source_navigation_applied_to_control=original.get("navigation_applied_to_control"),
        navigation_applied_to_control=False, legacy_quality_labels_missing=True)
    write_json(target/"run.json",meta)
    write_json(target/"sensor_hashes.json",checks)
    write_json(target/"record_hashes.json",{name:digest(target/name) for name in ("run.json","steps.jsonl","sensor_hashes.json")})
    return meta
