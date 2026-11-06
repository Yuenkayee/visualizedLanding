"""Per-sortie metrics. A sortie, never a frame, is the sampling unit."""
import gzip
import json
import numpy as np
from .common import read_json
from .recording import KNOT

TERMINALS = ("candidate_stop_proxy", "contact_unqualified", "navigation_abort", "timeout",
             "stop_geometry_inconsistent", "interface_numerical", "runaway")


def read_series(case):
    with gzip.open(case/"series.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream]


def arrays(rows):
    return dict(t=np.array([r["t"] for r in rows]), t1=np.array([r["t1"] for r in rows]),
        dt=np.array([r["t1"]-r["t"] for r in rows]), stage=np.array([r["stage"] for r in rows]),
        p=np.array([r["s"]["p"] for r in rows]), v=np.array([r["s"]["v"] for r in rows]),
        command=np.array([r["u"]["command"] for r in rows]).reshape(-1,3)*KNOT,
        relative=np.array([r["u"]["relative_command"] for r in rows]).reshape(-1,3)*KNOT,
        feedforward=np.array([r["u"]["feedforward"] for r in rows]).reshape(-1,3)*KNOT,
        control_valid=np.array([bool(r["u"]["control_valid"]) for r in rows]),
        nav_valid=np.array([r["nav_valid"] for r in rows]),
        shaper_valid=np.array([r["shaper_valid"] for r in rows]))


def nominal_point(s, model, geometry):
    rd, db = np.asarray(s["rd"]), np.asarray(s["db"])
    return rd.T @ np.asarray(model["nominal_cg_target_ned_m"]).reshape(3) + db @ geometry["gear_body"] - np.diag([1,-1,-1]) @ geometry["H_ship_body"]


def allowed_region(db, landing, geometry):
    """Erode 24x14 procedural deck by projected provisional wheel ellipsoids.

    Geometry display only; not a certified envelope or new STOP criterion.
    """
    low, high = [], []
    for wheel in landing["geometry"]["airframe_proxy"]["ellipsoids"]:
        if "wheel" not in wheel["name"]: continue
        center = np.asarray(db) @ (np.asarray(wheel["centre_body_m"])-geometry["gear_body"])
        radius = np.sqrt((np.asarray(db)**2) @ np.asarray(wheel["semiaxes_m"])**2)
        low.append(center[:2]-radius[:2]); high.append(center[:2]+radius[:2])
    return (-np.array([12,7])-np.min(low,axis=0)).tolist(), (np.array([12,7])-np.max(high,axis=0)).tolist()


def motion(rows, model):
    component = model["wave_component"]
    t = np.array([r["t"] for r in rows])
    speed = np.array([np.linalg.norm(np.asarray(r["u"]["wave_command"]).ravel()[:2]) for r in rows])
    omega = np.asarray(component["omega"]).ravel()
    k = np.asarray(component["waveNumber"]).ravel()
    encounter = omega[None,:]-k[None,:]*speed[:,None]*np.cos(np.asarray(component["beta"]).ravel())
    wave = np.sum(np.asarray(component["amplitude"]).ravel()*np.cos(encounter*t[:,None]+np.asarray(component["phase"]).ravel()), axis=1)
    ship = np.array([r["s"]["ship"] for r in rows])
    h = np.array([r["s"]["H_up_m"] for r in rows])
    return np.column_stack((wave, np.rad2deg(ship[:,6:8]), h-h[0], [r["s"]["H_normal_mps"] for r in rows]))


def summarize(case, policy):
    metadata = read_json(case/"run.json"); model = read_json(case/"simulink/batch_model.json")
    rows = read_series(case)
    if not rows: raise ValueError("No samples: cannot classify an empty run as a completed task")
    a = arrays(rows); b = metadata["config"]["batch"]; dt = a["dt"]
    candidate = metadata.get("termination_reason") == "model_stop"
    terminal = "candidate_stop_proxy" if candidate else "runaway" if metadata.get("termination_reason") == "runaway" else "timeout"
    last = rows[-1]["s1"]; target = nominal_point(last, model, metadata["geometry"])
    error = np.asarray(last["p"])-target
    limit = np.asarray(model["command_limit_mps"]).ravel()
    derivative = np.diff(a["command"],axis=0)/np.maximum(np.diff(a["t1"]),1e-12)[:,None]
    cost = np.sum((np.diff(a["command"],axis=0)/limit)**2*np.asarray(policy["command_cost_weights"])/np.maximum(np.diff(a["t1"]),1e-12)[:,None])
    phases = []
    nxt = np.array([int(np.asarray(r["u"]["stage"]).item()) for r in rows])
    # Entry at the terminal endpoint is an entered, zero-dwell stage; do not
    # lose it when STOP occurs exactly on a transition.
    for stage in range(3):
        mask = a["stage"] == stage
        entered = bool(mask.any() or (nxt == stage).any())
        complete = bool(np.any(mask & (nxt>stage)) or (stage == 2 and candidate and (mask.any() or nxt[-1]==2)))
        dwell = float(dt[mask].sum()) if entered else None
        phases.append(dict(stage=stage, entered=entered, completed=complete,
            completed_duration_s=dwell if complete else None, dwell_s=dwell,
            invalid_wait_s=float(dt[mask & ~a["control_valid"]].sum()) if entered else None,
            valid_dwell_s=float(dt[mask & a["control_valid"]].sum()) if entered else None))
    invalid = ~a["nav_valid"]
    longest = current = 0.
    for bad, span in zip(invalid,dt):
        current = current+span if bad else 0.; longest = max(longest,current)
    first_anomaly = None; accompanied = []
    for r in rows:
        diag = r["diagnostics"]
        reasons = []
        if not r["nav_valid"]: reasons.append("navigation_invalid")
        if not r["shaper_valid"]: reasons.append("shaper_invalid")
        if r["sampled_camera"] and not diag.get("vision_accepted",False): reasons.append("vision_rejected")
        if b["method"] != "C1" and r["sampled_lidar"] and not diag.get("lidar_accepted",False): reasons.append("lidar_rejected")
        if reasons and first_anomaly is None:
            first_anomaly = dict(time_s=r["t"], events=reasons, diagnostics=diag,
                                 cg_used=r["u"]["cg_used"], control_valid=r["u"]["control_valid"], stage=r["stage"])
        accompanied.extend(reasons)
    values = motion(rows,model)
    phase_error = []
    for stage in range(3):
        selection = a["stage"]==stage
        error_guidance = np.array([r["u"]["target"] for r in rows]).reshape(-1,3)-np.array([r["cg_true"] for r in rows])
        phase_error.append(dict(stage=stage, samples=int(selection.sum()),
            cg_tracking_rmse_m=float(np.sqrt(np.mean(np.sum(error_guidance[selection]**2,axis=1)))) if selection.any() else None,
            horizontal_speed_median_mps=float(np.median(np.linalg.norm(a["v"][selection,:2],axis=1))) if selection.any() else None,
            normal_speed_median_mps=float(np.median(a["v"][selection,2])) if selection.any() else None))
    return dict(case_id=str(case.name), group=b["group"], slot=b["slot"], replacement=b["replacement"],
        weather=metadata["config"]["weather"], sea=b["sea"], method=b["method"],
        wave_seed=b["wave_seed"], wind_seed=b["wind_seed"], sensor_seed=metadata["config"]["seed"],
        termination=terminal, candidate=candidate, qualified=candidate,
        qualification_policy="existing model STOP proxy; all candidates qualify by user definition",
        duration_s=float(metadata["final_timestamp"]), Jctrl=float(cost),
        error_x_m=float(error[0]) if candidate else None, error_y_m=float(error[1]) if candidate else None,
        radial_error_m=float(np.linalg.norm(error[:2])) if candidate else None,
        horizontal_speed_mps=float(np.linalg.norm(last["v"][:2])) if candidate else None,
        normal_speed_mps=float(last["v"][2]) if candidate else None,
        touchdown_position=last["p"] if candidate else None, nominal_gear_point=target.tolist(),
        terminal_truth=last, terminal_stage=int(nxt[-1]), phases=phases, phase_tracking=phase_error,
        tilt_body_deck_deg=float(np.rad2deg(np.arccos(np.clip(-np.asarray(last["db"])[2,2],-1,1)))),
        regressions=int(np.sum(nxt<a["stage"])), invalid_wait_s=float(dt[~a["control_valid"]].sum()),
        nav_invalid_s=float(dt[invalid].sum()), longest_nav_invalid_s=float(longest),
        preterminal_nav_invalid_s=float(current),
        last_valid_s=float(a["t"][a["nav_valid"]][-1]) if a["nav_valid"].any() else None,
        first_anomaly=first_anomaly, accompanied_events=sorted(set(accompanied)),
        command_peak_mps=np.max(abs(a["command"]),axis=0).tolist(),
        command_rate_peak_mps2=np.max(abs(derivative),axis=0).tolist() if len(derivative) else [0.]*3,
        near_bound_s=np.sum((abs(a["relative"]) >= policy["near_bound_fraction"]*limit)*dt[:,None],axis=0).tolist(),
        motion_rms=np.sqrt(np.mean(values**2,axis=0)).tolist(), motion_peak=np.max(abs(values),axis=0).tolist(),
        wave_speed_mps=np.asarray(rows[0]["u"]["wave_command"]).ravel().tolist(),
        feedforward_residual_max_mps=float(np.max(abs(a["command"]-a["relative"]-a["feedforward"]))),
        elapsed_wall_s=metadata["elapsed_s"], samples=len(rows), runaway_evidence=metadata.get("runaway_evidence"))
