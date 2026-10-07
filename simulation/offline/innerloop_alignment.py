"""Fixed-reference SLX configuration and verification of its actual feedback."""

import copy
import numpy as np
from navigation.common.contracts import finite_array, covariance
from navigation.common.frames import transform


def prepare_innerloop_config(plant_config, rig, rig_config):
    c = copy.deepcopy(plant_config)
    if c.get("initialize_callback") != "innerloop_initialize":
        return c
    if rig is None:
        raise ValueError("innerLoop navigation requires a calibrated camera_rig_config")
    # The interface fixture opts into current design geometry. Real experiments
    # supply calibrated values explicitly; conflicting values are never replaced.
    if c.pop("geometry_from_camera_rig", False):
        expected = dict(
            T_body_camera=rig.reference.T_body_camera.tolist(),
            H_ship_body=rig_config["geometry"]["H_ship_body_m"],
            gear_body=rig_config["geometry"]["gear_body_m"],
        )
        provided = c.get("geometry", {})
        if any(k not in expected for k in provided):
            raise ValueError("unknown innerLoop geometry parameter")
        for key, value in provided.items():
            if np.shape(value) != np.shape(expected[key]) or not np.allclose(
                value, expected[key]
            ):
                raise ValueError(f"innerLoop geometry conflicts with camera rig: {key}")
        c["geometry"] = expected
    geometry = c.get("geometry", {})
    mount = finite_array(
        geometry.get("T_body_camera"), (4, 4), "innerLoop T_body_camera"
    )
    if not np.allclose(mount, rig.reference.T_body_camera):
        raise ValueError(
            "innerLoop T_body_camera must equal the fixed rig reference C0 mount"
        )
    for key in ("H_ship_body", "gear_body"):
        finite_array(geometry.get(key), (3,), f"innerLoop {key}")
    for key in ("helicopter_state", "ship_state"):
        finite_array(c.get(key), (12,), f"innerLoop {key}")
    if c.get("reference_camera_id", rig.reference_camera_id) != rig.reference_camera_id:
        raise ValueError("innerLoop reference camera differs from navigation rig")
    c["reference_camera_id"] = rig.reference_camera_id
    return c


def verify_innerloop_feedback(raw, expected, timestamp, atol=1e-8):
    """Compare actual SLX conversion with independently computed Python output."""
    if abs(float(raw["timestamp"]) - timestamp) > 1e-8:
        raise ValueError("innerLoop feedback timestamp differs from current state")
    position = finite_array(
        np.asarray(raw["gear_relative_ned_m"]).reshape(-1),
        (3,),
        "innerLoop NED feedback",
    )
    P = covariance(raw["covariance_ned"], 3)
    distance = float(raw["distance_m"])
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("invalid innerLoop feedback distance")
    pairs = [
        (position, expected["gear_relative_estimate_ned"]),
        (P, expected["relative_covariance_ned"]),
        (distance, expected["relative_distance_estimate_m"]),
    ]
    errors = [float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in pairs]
    if max(errors) > atol or bool(raw["valid"]) != expected["feedback_valid"]:
        raise ValueError(
            "innerLoop feedback differs from Python fixed-reference conversion"
        )
    return dict(
        timestamp=timestamp,
        gear_relative_estimate_ned=position.tolist(),
        relative_distance_estimate_m=distance,
        relative_covariance_ned=P.tolist(),
        feedback_valid=bool(raw["valid"]),
        source="innerLoop.slx",
    ), max(errors)


def validate_navigation_context(context, timestamp):
    if abs(float(context["timestamp"]) - timestamp) > 1e-8:
        raise ValueError("ship attitude communication timestamp differs from state")
    return transform(context["R_ned_deck"])[:3, :3]
