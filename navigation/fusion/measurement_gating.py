import numpy as np
from scipy.stats import chi2
from scipy.spatial.transform import Rotation
from navigation.common.frames import skew


def mahalanobis_gate(residual, innovation_covariance, probability=0.997):
    r = np.asarray(residual, dtype=float)
    d2 = float(r @ np.linalg.solve(innovation_covariance, r))
    return bool(np.isfinite(d2) and d2 <= chi2.ppf(probability, r.size)), d2


def update(state, residual, H, R, probability=0.997):
    S = H @ state.P @ H.T + R
    accepted, d2 = mahalanobis_gate(residual, S, probability)
    if not accepted:
        return False, d2
    K = np.linalg.solve(S, H @ state.P).T
    dx = K @ residual
    state.position += dx[:3]
    state.velocity += dx[3:6]
    state.rotation = state.rotation @ Rotation.from_rotvec(dx[6:9]).as_matrix()
    state.angular_velocity += dx[9:12]
    A = np.eye(12) - K @ H
    P = A @ state.P @ A.T + K @ R @ K.T
    J = np.eye(12)
    J[6:9, 6:9] -= 0.5 * skew(dx[6:9])
    state.P = J @ P @ J.T
    state.P = 0.5 * (state.P + state.P.T)
    return True, d2
