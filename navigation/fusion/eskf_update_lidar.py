import numpy as np
from navigation.common.frames import skew
from navigation.fusion.measurement_gating import update


def update_lidar(state, plane, normal_sigma=0.02, distance_sigma=0.08):
    n = state.rotation.T @ np.array([0.0, 0.0, 1.0])
    r = np.r_[plane.normal - n, plane.offset - state.position[2]]
    H = np.zeros((4, 12))
    H[:3, 6:9] = skew(n)
    H[3, 2] = 1.0
    return update(state, r, H, np.diag([normal_sigma**2] * 3 + [distance_sigma**2]))
