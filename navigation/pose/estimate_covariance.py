"""Pixel Jacobian covariance in [deck translation, camera-local right rotation]."""

import numpy as np
from scipy.spatial.transform import Rotation
from navigation.pose.solve_pnp import project_points


def estimate_covariance(points, pose, calibration, pixel_sigma=1.5):
    if pixel_sigma <= 0:
        raise ValueError("pixel_sigma must be positive")
    J = np.zeros((len(points) * 2, 6))
    eps = 1e-5
    for i in range(6):
        plus = pose.copy()
        minus = pose.copy()
        if i < 3:
            plus[i, 3] += eps
            minus[i, 3] -= eps
        else:
            dr = np.zeros(3)
            dr[i - 3] = eps
            plus[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(dr).as_matrix()
            minus[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(-dr).as_matrix()
        J[:, i] = (
            (
                project_points(points, plus, calibration)
                - project_points(points, minus, calibration)
            )
            / (2 * eps)
        ).ravel()
    A = J.T @ J / pixel_sigma**2
    vals, vecs = np.linalg.eigh(A)
    P = (vecs * (1 / np.maximum(vals, 1e-8))) @ vecs.T
    return 0.5 * (P + P.T) + np.eye(6) * 1e-8
