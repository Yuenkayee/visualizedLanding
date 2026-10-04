"""Constant-velocity/angular-rate error-state prediction, with optional acceleration."""

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from navigation.common.frames import skew
from navigation.common.timestamps import require_monotonic


def predict(
    state,
    timestamp,
    acceleration=None,
    acceleration_noise=0.5,
    angular_acceleration_noise=0.05,
):
    timestamp = require_monotonic(state.timestamp, timestamp, allow_equal=True)
    dt = timestamp - state.timestamp
    if dt == 0:
        return state
    a = np.zeros(3) if acceleration is None else np.asarray(acceleration, dtype=float)
    state.position += state.velocity * dt + 0.5 * a * dt * dt
    state.velocity += a * dt
    state.rotation = (
        state.rotation @ Rotation.from_rotvec(state.angular_velocity * dt).as_matrix()
    )
    F = np.zeros((12, 12))
    F[:3, 3:6] = np.eye(3)
    F[6:9, 6:9] = -skew(state.angular_velocity)
    F[6:9, 9:12] = np.eye(3)
    G = np.zeros((12, 6))
    G[3:6, :3] = np.eye(3)
    G[9:12, 3:] = np.eye(3)
    Qc = (
        G
        @ np.diag([acceleration_noise**2] * 3 + [angular_acceleration_noise**2] * 3)
        @ G.T
    )
    # Van Loan discretization retains position/velocity process cross covariance.
    M = np.block([[F, Qc], [np.zeros_like(F), -F.T]]) * dt
    E = expm(M)
    Phi = E[:12, :12]
    Q = E[:12, 12:] @ Phi.T
    state.P = Phi @ state.P @ Phi.T + Q
    state.P = 0.5 * (state.P + state.P.T)
    state.timestamp = timestamp
    return state
