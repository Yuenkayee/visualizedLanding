import numpy as np


class HealthMonitor:
    def __init__(self, max_age=1.0, max_position_std=5.0):
        self.max_age = max_age
        self.max_position_std = max_position_std
        self.last_measurement = None

    def record(self, timestamp):
        self.last_measurement = timestamp

    def check(self, state):
        age = (
            float("inf")
            if self.last_measurement is None
            else state.timestamp - self.last_measurement
        )
        std = float(np.sqrt(max(np.linalg.eigvalsh(state.P[:3, :3]).max(), 0)))
        healthy = (
            age <= self.max_age
            and std <= self.max_position_std
            and np.isfinite(state.P).all()
        )
        return bool(healthy), dict(
            measurement_age=age if np.isfinite(age) else None, position_std=std
        )
