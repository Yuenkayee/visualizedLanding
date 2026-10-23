"""Causal CG signal resampling; mirrors cg_signal_step.m for sensor replay.

This conditions an existing navigation estimate. It is not a second estimator:
held inputs are propagated as references, without any covariance reduction.
"""
import numpy as np

DEFAULTS = dict(time_constant_s=0.04, max_correction_speed_mps=5.,
                max_age_s=0.30, max_tracking_error_m=1.)


def parameters(config=None):
    config = config or {}
    if set(config) - set(DEFAULTS):
        raise ValueError(f"Unknown cg_signal parameters: {set(config) - set(DEFAULTS)}")
    values = [float(config.get(k, v)) for k, v in DEFAULTS.items()]
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError("cg_signal parameters must be finite and positive")
    return values


class CgSignal:
    def __init__(self, config=None):
        self.parameters = parameters(config)
        self.initialized = False
        self.time = self.sample_time = self.valid_time = -1.
        self.healthy = False
        self.position = np.zeros(3)
        self.feed_velocity = np.zeros(3)
        self.velocity = np.zeros(3)
        self.reference_position = np.zeros(3)
        self.tracking_error = 0.

    def step(self, t, position, velocity, healthy, sample_t):
        tau, speed, max_age, max_error = self.parameters
        if not np.isfinite(t) or t < 0 or t < self.time - 1e-8:
            raise ValueError("Nonmonotonic output clock")
        position, velocity = np.asarray(position).reshape(3), np.asarray(velocity).reshape(3)
        dt = max(0., t - self.time) if self.time >= 0 else 0.
        packet_ok = np.isfinite(sample_t) and 0 <= sample_t <= t + 1e-8
        good = bool(healthy) and np.isfinite(position).all() and np.isfinite(velocity).all()
        fresh = packet_ok and (sample_t > self.sample_time + 1e-8 or (
            not self.initialized and good and sample_t >= self.sample_time))
        if not good or not packet_ok:
            self.healthy = False
        if fresh:
            self.sample_time = sample_t
            self.healthy = good and t - sample_t <= max_age + 1e-8
        usable = fresh and self.healthy
        if usable and not self.initialized:
            self.position = position + velocity * (t - sample_t)
            self.feed_velocity, self.velocity = velocity.copy(), velocity.copy()
            self.reference_position = position.copy()
            self.valid_time, self.initialized, self.tracking_error = sample_t, True, 0.
        elif self.initialized:
            previous = self.position.copy()
            coast = max(0., min(t, self.valid_time + max_age) - max(0., self.time))
            predicted = self.position + self.feed_velocity * coast
            if usable:
                predicted += .5 * (velocity - self.feed_velocity) * coast
                self.reference_position = position.copy()
                self.feed_velocity, self.valid_time = velocity.copy(), sample_t
            if self.healthy and t - self.valid_time <= max_age + 1e-8:
                residual = self.reference_position + self.feed_velocity * (t - self.valid_time) - predicted
                correction = -np.expm1(-dt / tau) * residual
                correction *= min(1., speed * dt / max(np.linalg.norm(correction), np.finfo(float).eps))
                self.position = predicted + correction
                self.tracking_error = float(np.linalg.norm(residual - correction))
            else:
                self.position = predicted
            if dt > 1e-8:
                self.velocity = (self.position - previous) / dt
        self.time = t
        age = max(0., t - self.valid_time) if self.initialized else max_age + 1
        valid = self.initialized and self.healthy and age <= max_age + 1e-8 and self.tracking_error <= max_error
        return dict(cg_relative_estimate_50hz_ned=self.position.tolist(),
                    cg_relative_velocity_50hz_ned_mps=self.velocity.tolist(),
                    cg_signal_valid=bool(valid), cg_signal_age_s=age)


def relative_cg(estimate, geometry, rotation_ned_deck, omega_deck):
    """Ship attitude/rates are communicated inputs, never helicopter truth.

    Returns heli CG minus ship CG and its derivative in NED. Covariance assumes
    the same exact ship attitude and installation geometry as the convert kernel.
    """
    from navigation.common.reference_points import body_point_estimate
    p, v, covariance = body_point_estimate(estimate, np.zeros(3))
    R = np.asarray(rotation_ned_deck)
    offset = np.diag([1., -1., -1.]) @ geometry["H_ship_body"]
    return dict(cg_relative_estimate_ned=(R @ (p + offset)).tolist(),
                navigation_velocity_cg_ned_mps=(R @ (v + np.cross(omega_deck, p + offset))).tolist(),
                cg_relative_covariance_ned=(R @ covariance @ R.T).tolist())
