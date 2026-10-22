"""Causal, slow target-centering outer loop, shared by runtime and synthesis.

Only estimated camera position enters this controller. Reference shaping is
performed here once per host tick, not again inside MATLAB. Fast stabilization
and actuator limits remain in Gimbal/gimbal_step; the command is not a pose
measurement. An invalid estimate brakes the reference without truth fallback.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from navigation.common.contracts import GimbalReference
from navigation.common.gimbal import Gimbal
from navigation.common.timestamps import require_monotonic


def bounded(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1e-15))


class GimbalTargetTracker:
    def __init__(self, neutral, config=None):
        servo = Gimbal(neutral, config)
        self.config = servo.config["tracking"]
        self.enabled = bool(servo.config["enabled"] and self.config["enabled"])
        self.nominal = servo.reference.copy()
        self.reference = self.nominal.copy()
        self.goal = self.nominal.copy()
        self.rate = np.zeros(3)  # reference angular velocity expressed in DECK
        self.timestamp = None
        self.status = "waiting_for_navigation" if self.enabled else "disabled"
        self.limited = False

    def update(self, timestamp, T_deck_camera, valid, source="navigation"):
        """Advance using the previous goal, then latch this tick's observation.

        Synthetic datasets explicitly label their noisy pose proxy as synthetic;
        runtime callers must supply a synchronized, healthy navigation estimate.
        """
        previous = self.timestamp
        require_monotonic(0 if previous is None else previous, timestamp, allow_equal=True)
        if previous is not None and timestamp == previous:
            return GimbalReference(timestamp, self.reference.copy(), self.rate.copy())
        if self.enabled and previous is not None:
            dt = timestamp - previous
            n = max(1, int(np.ceil(dt / .002)))
            h, tau = dt / n, self.config["time_constant_s"]
            for _ in range(n):
                error = Rotation.from_matrix(self.goal @ self.reference.T).as_rotvec()
                accel = -2 * self.rate / tau
                if self.status == "tracking":
                    accel += error / tau**2
                accel = bounded(accel, self.config["max_acceleration_rad_s2"])
                self.rate = bounded(self.rate + h * accel, self.config["max_rate_rad_s"])
                self.reference = Rotation.from_rotvec(h * self.rate).as_matrix() @ self.reference
        self.timestamp = float(timestamp)
        self.limited = False
        if self.enabled:
            self.status = "hold_invalid_navigation"
            if valid:
                position = np.asarray(T_deck_camera, float)[:3, 3]
                direction = np.asarray(self.config["target_deck_m"]) - position
                distance = np.linalg.norm(direction)
                if np.isfinite(direction).all() and distance >= self.config["min_target_distance_m"]:
                    # Angular centroid of the extended H, using known geometry
                    # and estimated camera position only. A point look-at can
                    # crop the nearest H edge during the last metres of descent.
                    x, y = np.asarray(self.config["target_extent_m"]) / 2
                    rays = direction + np.array([[-x, -y, 0], [-x, y, 0], [x, -y, 0], [x, y, 0]])
                    direction = np.sum(rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9), axis=0)
                    direction /= max(np.linalg.norm(direction), 1e-15)
                    z = self.nominal[:, 2]
                    axis = np.cross(z, direction)
                    sine = np.linalg.norm(axis)
                    angle = np.arctan2(sine, np.dot(z, direction))
                    # Reject a target behind the nominal camera (e.g. below deck).
                    if np.dot(z, direction) > 0:
                        correction = max(0., angle - self.config["centering_deadband_rad"])
                        self.limited = correction > self.config["max_tilt_rad"]
                        turn = axis / max(sine, 1e-15) * min(correction, self.config["max_tilt_rad"])
                        self.goal = Rotation.from_rotvec(turn).as_matrix() @ self.nominal
                        self.status = "tracking"
                    else:
                        self.status = "hold_target_behind_camera"
                else:
                    self.status = "hold_target_too_close"
        self.diagnostics = dict(
            status=self.status, source=source, target_limited=bool(self.limited),
            R_deck_camera_reference=self.reference.tolist(),
            R_deck_camera_goal=self.goal.tolist(),
            reference_rate_deck_rad_s=self.rate.tolist(),
            reference_tilt_rad=float(Rotation.from_matrix(self.reference @ self.nominal.T).magnitude()),
        )
        return GimbalReference(timestamp, self.reference.copy(), self.rate.copy())

    def apply(self, estimate):
        encoder = estimate.gimbal
        valid = (estimate.healthy and encoder is not None and encoder.valid
                 and abs(encoder.timestamp - estimate.timestamp) <= 1e-8)
        estimate.gimbal_reference = self.update(estimate.timestamp, estimate.T_deck_camera, valid)
        estimate.diagnostics["gimbal_tracking"] = self.diagnostics.copy()
        return estimate.gimbal_reference
