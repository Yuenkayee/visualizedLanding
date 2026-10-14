"""Convert a CAMERA estimate to a fixed BODY point using encoder telemetry."""

import numpy as np
from navigation.common.frames import invert, skew


def body_point_estimate(estimate, point_body):
    mount = estimate.gimbal
    if (
        mount is None
        or not mount.valid
        or abs(mount.timestamp - estimate.timestamp) > 1e-8
    ):
        raise ValueError("body feedback requires synchronized gimbal telemetry")
    T = estimate.T_deck_camera
    lever = mount.T_body_camera[:3, :3].T @ (
        np.asarray(point_body) - mount.T_body_camera[:3, 3]
    )
    R = T[:3, :3]
    position = T[:3, 3] + R @ lever
    # d(R_dc*l_c)/dt, removing camera motion relative to body.
    velocity = estimate.velocity + R @ np.cross(
        estimate.angular_velocity_camera - mount.angular_velocity_camera, lever
    )
    J = np.column_stack((np.eye(3), -R @ skew(lever)))
    Jg = R @ skew(lever)
    covariance = J @ estimate.covariance @ J.T + Jg @ mount.covariance @ Jg.T
    return position, velocity, covariance


def body_pose_estimate(estimate):
    body_point_estimate(estimate, np.zeros(3))  # Same synchronization requirements.
    return estimate.T_deck_camera @ invert(estimate.gimbal.T_body_camera)
