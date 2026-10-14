import numpy as np
from navigation.common.frames import skew
from navigation.fusion.measurement_gating import update


def update_lidar(
    state, plane, normal_sigma=0.02, distance_sigma=0.08,
    mount_covariance=None, mount_jacobian=None,
):
    n = state.rotation.T @ np.array([0.0, 0.0, 1.0])
    r = np.r_[plane.normal - n, plane.offset - state.position[2]]
    H = np.zeros((4, 12))
    H[:3, 6:9] = skew(n)
    H[3, 2] = 1.0
    noise = np.diag([normal_sigma**2] * 3 + [distance_sigma**2])
    if mount_covariance is not None:
        # Rotating the cloud about the camera origin perturbs its normal, not
        # plane distance. Body-fixed LiDAR translation was transformed above.
        J = mount_jacobian
        if J is None:
            J = np.zeros((4, 3))
            J[:3] = skew(plane.normal)
        noise += J @ mount_covariance @ J.T
    return update(state, r, H, noise)
