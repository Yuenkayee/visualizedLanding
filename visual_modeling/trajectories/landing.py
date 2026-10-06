"""Kinematic dataset generator; wake is a correlated residual, not CFD/flight dynamics."""

import numpy as np
from scipy.spatial.transform import Rotation
from navigation.common.frames import transform, invert, ned_to_enu, transform_points
from navigation.common.contracts import CameraCalibration
from navigation.pose.solve_pnp import project_points
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.sensors.airframe_visibility import (
    airframe_primitives,
    validate_external_camera,
    airframe_occlusion,
)

STAGES = ("approach", "window_hold", "touchdown")


def cruise_progress(s, ramp=0.12):
    """C1 velocity, mostly constant, zero velocity at both transition endpoints."""
    if not 0 < ramp < 0.5:
        raise ValueError("acceleration_fraction must be between 0 and .5")
    s = np.clip(np.asarray(s, float), 0, 1)
    u = np.minimum(s, 1 - s) / ramp
    edge = ramp * (u**3 - 0.5 * u**4) / (1 - ramp)
    return np.where(
        s < ramp, edge, np.where(s > 1 - ramp, 1 - edge, (s - ramp / 2) / (1 - ramp))
    )


def camera_geometry(config, rng=None):
    c = config["camera"]
    w, h = c["width"], c["height"]
    if (
        min(w, h) <= 0
        or not 0 < c["horizontal_fov_deg"] < 170
        or not 0 < c["vertical_fov_deg"] < 170
    ):
        raise ValueError("invalid pinhole resolution/FOV")
    fx = w / (2 * np.tan(np.deg2rad(c["horizontal_fov_deg"] / 2)))
    fy = h / (2 * np.tan(np.deg2rad(c["vertical_fov_deg"] / 2)))
    cal = CameraCalibration(
        np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]]), w, h
    )
    a = np.deg2rad(c["optical_down_deg"])
    # Camera +x right, +y down, +z forward; body forward/right/down.
    R = np.array([[0, -np.sin(a), np.cos(a)], [1, 0, 0], [0, np.cos(a), np.sin(a)]])
    if rng is not None:
        R = (
            Rotation.from_rotvec(
                np.deg2rad(rng.normal(0, c.get("mounting_error_deg", 0), 3))
            ).as_matrix()
            @ R
        )
    mount = transform(R, c["position_body_m"])
    validate_external_camera(mount[:3, 3], airframe_primitives(config["geometry"]))
    gear = np.asarray(config["geometry"]["gear_body_m"], float)
    if gear.shape != (3,) or gear[2] - mount[2, 3] <= 0.1:
        raise ValueError(
            "camera must be above wheel-bottom plane; fill calibrated geometry"
        )
    return cal, mount


def correlated_noise(times, rng, std, correlation):
    if std < 0 or correlation <= 0:
        raise ValueError("invalid wake residual settings")
    a = np.zeros((len(times), 3))
    a[0] = rng.normal(0, std, 3)
    for i in range(1, len(times)):
        decay = np.exp(-(times[i] - times[i - 1]) / correlation)
        a[i] = decay * a[i - 1] + np.sqrt(1 - decay**2) * rng.normal(0, std, 3)
    return a


def _draw(rng, bounds):
    return float(rng.uniform(*bounds))


def make_sortie(config, seed, sequence_id, weather, window_wait=None):
    rng = np.random.default_rng(seed)
    c = config
    m = c["mission"]
    motion = c["motion"]
    if c["trajectory_hz"] <= 0:
        raise ValueError("trajectory_hz must be positive")
    durations = [
        _draw(rng, m["approach_duration_s"]),
        _draw(rng, m["hover_duration_s"])
        if window_wait is None
        else float(window_wait),
        _draw(rng, m["descent_duration_s"]),
    ]
    if min(durations) <= 0:
        raise ValueError("stage durations must be positive")
    # Each interval includes its endpoint, shared boundaries occur only once.
    starts = np.r_[0, np.cumsum(durations)[:-1]]
    times = []
    stage = []
    local = []
    for k, (start, duration) in enumerate(zip(starts, durations)):
        n = max(2, int(np.ceil(duration * c["trajectory_hz"])))
        fraction = np.linspace(0, 1, n + 1)
        if k:
            fraction = fraction[1:]
        times.extend(start + duration * fraction)
        stage.extend([k] * len(fraction))
        local.extend(fraction)
    t = np.array(times)
    stage = np.array(stage)
    s = np.array(local)
    initial = np.asarray(m["initial_gear_heading_m"], float)
    height = m["hover_height_m"]
    if initial.shape != (3,) or initial[0] >= 0 or initial[2] <= height or height <= 0:
        raise ValueError(
            "approach must begin astern and above the positive hover height"
        )
    relative = np.zeros((len(t), 3))
    relative[:, 2] = height
    p = cruise_progress(s, m["acceleration_fraction"])
    idx = stage == 0
    relative[idx] = initial + (np.array([0, 0, height]) - initial) * p[idx, None]
    idx = stage == 2
    relative[idx, 2] = height * (1 - p[idx])
    # Correlated wake peaks astern/near deck, not independent random camera poses.
    noise = correlated_noise(
        t, rng, motion["wake_position_std_m"], motion["wake_correlation_s"]
    )
    spatial = np.exp(-np.abs(relative[:, 0]) / 70) * np.exp(-relative[:, 2] / 25)
    envelope = np.sin(np.pi * s) ** 2
    hold_scale = np.where(stage == 1, motion["hover_residual_scale"], 1.0)
    disturbance = noise * spatial[:, None] * envelope[:, None] * hold_scale[:, None]
    relative += disturbance
    relative[:, 2] = np.maximum(relative[:, 2], 0)
    relative[-1] = 0  # Reference-point touch, not aircraft CG or camera contact.
    period = _draw(rng, motion["wave_period_s"])
    phase = rng.uniform(0, 2 * np.pi, 3)
    roll = np.deg2rad(_draw(rng, motion["ship_roll_deg"])) * np.sin(
        2 * np.pi * t / period + phase[0]
    )
    pitch = np.deg2rad(_draw(rng, motion["ship_pitch_deg"])) * np.sin(
        2 * np.pi * t / (period * 0.83) + phase[1]
    )
    heave = _draw(rng, motion["ship_heave_m"]) * np.sin(
        2 * np.pi * t / (period * 1.1) + phase[2]
    )
    heading = rng.uniform(-np.pi, np.pi)
    speed = _draw(rng, m["ship_speed_knots"]) * 1852 / 3600
    A = ned_to_enu()[:3, :3]
    D = np.diag([1.0, -1.0, -1.0])
    ship_R = Rotation.from_euler(
        "xyz", np.c_[roll, pitch, np.full(len(t), heading)]
    ).as_matrix()
    ship_cg = np.c_[speed * t * np.cos(heading), speed * t * np.sin(heading), -heave]
    H_offset = np.asarray(c["geometry"]["H_ship_body_m"], float)
    h_ned = ship_cg + np.einsum("nij,j->ni", ship_R, H_offset)
    deck_R = ship_R @ D
    # "Directly above" is gravity vertical, not the normal of a rocking deck.
    # Use a level heading frame for the approach/hold/descent targets, while
    # retaining the actual tilted deck frame for camera poses and PnP labels.
    heading_R = Rotation.from_euler("z", heading).as_matrix() @ D
    gear_relative_ned = relative @ heading_R.T
    gear_relative_deck = np.einsum("nji,nj->ni", deck_R, gear_relative_ned)
    gear_ned = h_ned + gear_relative_ned
    gear_velocity = np.gradient(gear_ned, t, axis=0)
    # Yaw follows horizontal flight velocity to suppress side slip; no orbit/look-at.
    yaw = np.unwrap(np.arctan2(gear_velocity[:, 1], gear_velocity[:, 0]))
    attitudes = correlated_noise(
        t,
        rng,
        np.deg2rad(motion["wake_attitude_std_deg"]),
        motion["wake_correlation_s"],
    )
    attitudes *= spatial[:, None]
    vibration = np.deg2rad(motion["camera_vibration_deg"]) * np.sin(
        t[:, None] * np.array([19.0, 23.0, 29.0]) + phase
    )
    heli_roll = attitudes[:, 0] + vibration[:, 0]
    heli_pitch = (
        np.deg2rad(motion["helicopter_pitch_deg"]) + attitudes[:, 1] + vibration[:, 1]
    )
    yaw += np.deg2rad(motion["yaw_tracking_error_deg"]) * np.sin(t * 0.8 + phase[0])
    heli_R = Rotation.from_euler("xyz", np.c_[heli_roll, heli_pitch, yaw]).as_matrix()
    gear_body = np.asarray(c["geometry"]["gear_body_m"], float)
    heli_cg = gear_ned - np.einsum("nij,j->ni", heli_R, gear_body)
    heli_velocity = np.gradient(heli_cg, t, axis=0)
    ship_velocity = np.gradient(ship_cg, t, axis=0)
    body_velocity = np.einsum("nji,nj->ni", heli_R, heli_velocity)
    # Align heading with CG flight direction, including lever motion caused by
    # roll/pitch, rather than only with gear-centre velocity.
    for _ in range(3):
        yaw = np.unwrap(
            np.arctan2(heli_velocity[:, 1], heli_velocity[:, 0])
        ) + np.deg2rad(motion["yaw_tracking_error_deg"]) * np.sin(t * 0.8 + phase[0])
        heli_R = Rotation.from_euler(
            "xyz", np.c_[heli_roll, heli_pitch, yaw]
        ).as_matrix()
        heli_cg = gear_ned - np.einsum("nij,j->ni", heli_R, gear_body)
        heli_velocity = np.gradient(heli_cg, t, axis=0)
        body_velocity = np.einsum("nji,nj->ni", heli_R, heli_velocity)
    beta = np.arctan2(body_velocity[:, 1], np.maximum(body_velocity[:, 0], 1e-8))
    cal, mount = camera_geometry(c, rng)
    rows = []
    for i in range(len(t)):
        world_deck = transform(A @ deck_R[i], A @ h_ned[i])
        world_body = transform(A @ heli_R[i], A @ heli_cg[i])
        camera_pose = invert(world_deck) @ world_body @ mount
        rows.append(
            dict(
                timestamp=float(t[i]),
                frame_id=i,
                stage=STAGES[stage[i]],
                stage_progress=float(s[i]),
                gear_relative_heading_m=relative[i].tolist(),
                gear_relative_ned_m=gear_relative_ned[i].tolist(),
                gear_relative_deck_m=gear_relative_deck[i].tolist(),
                gear_relative_nominal_heading_m=(relative[i] - disturbance[i]).tolist(),
                wake_position_residual_m=disturbance[i].tolist(),
                T_world_deck=world_deck.tolist(),
                T_world_body=world_body.tolist(),
                T_deck_camera=camera_pose.tolist(),
                helicopter_cg_ned_m=heli_cg[i].tolist(),
                helicopter_velocity_body_knots=(
                    body_velocity[i] * 3600 / 1852
                ).tolist(),
                helicopter_euler_rad=[
                    float(heli_roll[i]),
                    float(heli_pitch[i]),
                    float(yaw[i]),
                ],
                ship_cg_ned_m=ship_cg[i].tolist(),
                ship_velocity_ned_knots=(ship_velocity[i] * 3600 / 1852).tolist(),
                ship_euler_rad=[float(roll[i]), float(pitch[i]), float(heading)],
                sideslip_rad=float(beta[i]),
                touchdown=bool(i == len(t) - 1),
                window_open=bool(stage[i] == 2),
            )
        )
    return dict(
        sequence_id=sequence_id,
        seed=int(seed),
        weather=weather,
        stage_durations_s=durations,
        window_source="provided_event"
        if window_wait is not None
        else "sampled_wait_not_predictor",
        ship_speed_knots=speed * 3600 / 1852,
        trajectory_reference_frame="ship_heading_forward_port_up_gravity_level",
        geometry=c["geometry"],
        T_body_camera=mount.tolist(),
        camera=dict(
            width=cal.width,
            height=cal.height,
            K=cal.K.tolist(),
            distortion=cal.distortion.tolist(),
        ),
        rows=rows,
    )


def selected_frames(sortie, per_stage):
    if per_stage < 1:
        raise ValueError("frames_per_stage must be positive")
    rows = sortie["rows"]
    chosen = []
    for stage in STAGES:
        indices = [i for i, r in enumerate(rows) if r["stage"] == stage]
        # Small previews still show actual terminal contact, not just descent start.
        picks = (
            np.unique(
                np.rint(
                    np.linspace(0, len(indices) - 1, min(per_stage, len(indices)))
                ).astype(int)
            )
            if per_stage > 1
            else [len(indices) // 2 if stage != "touchdown" else len(indices) - 1]
        )
        chosen.extend(rows[indices[i]] for i in picks)
    return chosen


def optical_report(sortie, marker):
    c = sortie["camera"]
    cal = CameraCalibration(c["K"], c["width"], c["height"])
    points = marker_points(marker["width"], marker["length"])
    centre = np.array([[0.0, 0.0, 0.025]])
    targets = np.vstack([points, centre])
    mount = np.asarray(sortie["T_body_camera"])
    primitives = airframe_primitives(sortie["geometry"])
    result = {}
    for stage in STAGES:
        full = []
        seen = []
        extent = []
        clear_centre = []
        clear_corners = []
        visible_centre = []
        visible_corners = []
        camera_height = []
        for row in sortie["rows"]:
            if row["stage"] != stage:
                continue
            T = np.array(row["T_deck_camera"])
            pixels = project_points(points, T, cal)
            z = transform_points(invert(T), points)[:, 2]
            inside = (
                (z > 0)
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] < cal.width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < cal.height)
            )
            uv = project_points(centre, T, cal)[0]
            depth = transform_points(invert(T), centre)[0, 2]
            seen.append(
                bool(depth > 0 and 0 <= uv[0] < cal.width and 0 <= uv[1] < cal.height)
            )
            blocked = airframe_occlusion(
                mount[:3, 3], transform_points(mount @ invert(T), targets), primitives
            )
            clear_centre.append(bool(not blocked[-1]))
            clear_corners.append(bool(not blocked[:4].any()))
            visible_centre.append(bool(seen[-1] and not blocked[-1]))
            visible_corners.append(bool((inside & ~blocked[:4]).all()))
            camera_height.append(float(T[2, 3]))
            full.append(bool(inside.all()))
            extent.append(float(min(np.ptp(pixels[:, 0]), np.ptp(pixels[:, 1]))))
        result[stage] = dict(
            centre_in_frame_fraction=float(np.mean(seen)),
            four_corners_in_frame_fraction=float(np.mean(full)),
            centre_unoccluded_fraction=float(np.mean(clear_centre)),
            four_corners_unoccluded_fraction=float(np.mean(clear_corners)),
            centre_visible_fraction=float(np.mean(visible_centre)),
            four_corners_visible_fraction=float(np.mean(visible_corners)),
            min_camera_height_above_deck_normal_m=min(camera_height),
            min_marker_extent_px=min(extent),
        )
    return result
