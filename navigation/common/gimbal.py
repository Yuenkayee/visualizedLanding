"""Finite-bandwidth, co-located XYZ gimbal servo shared by all Python simulators.

R_body_camera = Rz(qz) Ry(qy) Rx(qx) R_body_camera_neutral.
J*qdd + B*qd = torque is reduced to a damped second-order position servo,
with optional rate feedforward, torque/acceleration, rate and travel limits.
Only rotation changes. Neither pivot dimensions nor arm translation is added.
"""

from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from navigation.common.frames import transform
from navigation.common.contracts import GimbalSample, finite_array
from navigation.common.timestamps import require_monotonic


def gimbal_config(overrides=None):
    path = Path(__file__).resolve().parents[2] / "configs/gimbal.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(overrides or {})
    for key in ("natural_frequency_hz", "damping_ratio", "integration_step_s"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"gimbal {key} must be finite and positive")
    for key in ("max_rate_rad_s", "max_acceleration_rad_s2", "angle_limits_rad"):
        a = finite_array(config[key], (3,), key)
        if (a <= 0).any():
            raise ValueError(f"gimbal {key} must be positive")
    limits = config["angle_limits_rad"]
    if max(limits) >= np.pi or limits[1] >= np.pi / 2:
        raise ValueError("gimbal travel must avoid Euler pitch singularity and pi wraps")
    if not np.isfinite(config["encoder_std_rad"]) or config["encoder_std_rad"] < 0:
        raise ValueError("invalid gimbal encoder uncertainty")
    return config


class Gimbal:
    def __init__(self, T_body_camera_neutral, config=None):
        self.neutral = GimbalSample(0, T_body_camera_neutral).T_body_camera
        self.config = gimbal_config(config)
        reference = self.config["R_deck_camera_reference"]
        self.reference = transform(
            np.diag([1.0, -1.0, -1.0]) @ self.neutral[:3, :3]
            if reference is None
            else reference
        )[:3, :3]
        self.angles = np.zeros(3)
        self.rates = np.zeros(3)
        self.timestamp = None
        self.command = None
        self.saturated = False

    def advance(self, timestamp, R_world_body, R_world_deck):
        if self.timestamp is not None and timestamp == self.timestamp:
            return self.sample()
        target = Rotation.from_matrix(
            np.asarray(R_world_body).T
            @ R_world_deck
            @ self.reference
            @ self.neutral[:3, :3].T
        ).as_euler("xyz")
        limits = np.asarray(self.config["angle_limits_rad"])
        commanded = np.clip(target, -limits, limits)
        self.saturated = bool(np.any(np.abs(target) > limits))
        if self.timestamp is None:
            require_monotonic(0, timestamp, allow_equal=True)
            initial = self.config["initial_angles_rad"]
            self.angles = (
                commanded.copy()
                if initial is None
                else finite_array(initial, (3,), "initial angles")
            )
            if (np.abs(self.angles) > limits).any():
                raise ValueError("initial gimbal angles exceed travel")
        else:
            dt = (
                require_monotonic(self.timestamp, timestamp, allow_equal=True)
                - self.timestamp
            )
            if dt > 0 and self.config["enabled"]:
                # Interpolate measured attitude commands; never integrate twice on reads.
                wn = 2 * np.pi * self.config["natural_frequency_hz"]
                n = int(np.ceil(dt / min(self.config["integration_step_s"], 0.1 / wn)))
                h = dt / n
                speed = (commanded - self.command) / dt
                feedforward = speed if self.config["rate_feedforward"] else np.zeros(3)
                max_a = np.asarray(self.config["max_acceleration_rad_s2"])
                max_v = np.asarray(self.config["max_rate_rad_s"])
                for k in range(n):
                    desired = self.command + speed * (k + 0.5) * h
                    accel = (
                        wn**2 * (desired - self.angles)
                        + 2 * self.config["damping_ratio"] * wn
                        * (feedforward - self.rates)
                    )
                    clipped_a = np.clip(accel, -max_a, max_a)
                    raw_v = self.rates + clipped_a * h
                    self.rates = np.clip(raw_v, -max_v, max_v)
                    raw_q = self.angles + self.rates * h
                    self.angles = np.clip(raw_q, -limits, limits)
                    stop = (np.abs(raw_q) >= limits) & (raw_q * self.rates > 0)
                    self.rates[stop] = 0
                    self.saturated |= bool(
                        np.any(accel != clipped_a)
                        or np.any(raw_v != self.rates)
                        or np.any(stop)
                    )
        if not self.config["enabled"]:
            self.angles[:] = 0
            self.rates[:] = 0
            self.saturated = False
        self.command = commanded
        self.timestamp = float(timestamp)
        return self.sample()

    def sample(self, rng=None):
        if self.timestamp is None:
            raise ValueError("initialize gimbal before reading")
        R = Rotation.from_euler("xyz", self.angles).as_matrix() @ self.neutral[:3, :3]
        sigma = self.config["encoder_std_rad"] if self.config["enabled"] else 0.0
        if rng is not None:
            R = R @ Rotation.from_rotvec(rng.normal(0, sigma, 3)).as_matrix()
        roll, pitch, _ = self.angles
        E = np.array(
            [[1, 0, -np.sin(pitch)],
             [0, np.cos(roll), np.sin(roll) * np.cos(pitch)],
             [0, -np.sin(roll), np.cos(roll) * np.cos(pitch)]]
        )
        omega = self.neutral[:3, :3].T @ E @ self.rates
        return GimbalSample(
            self.timestamp, transform(R, self.neutral[:3, 3]), omega,
            np.eye(3) * sigma**2, True, self.saturated,
        )
