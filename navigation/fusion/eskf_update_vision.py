import numpy as np
from scipy.spatial.transform import Rotation
from navigation.fusion.measurement_gating import update


def update_vision(state, measurement):
    if abs(measurement.timestamp - state.timestamp) > 1e-8:
        raise ValueError("predict to measurement timestamp first")
    T = measurement.T_deck_camera
    r = np.r_[
        T[:3, 3] - state.position,
        Rotation.from_matrix(state.rotation.T @ T[:3, :3]).as_rotvec(),
    ]
    H = np.zeros((6, 12))
    H[:3, :3] = np.eye(3)
    H[3:, 6:9] = np.eye(3)
    return update(state, r, H, measurement.covariance)
