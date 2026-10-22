"""Common adapter API; mock is a bounded-acceleration point-mass landing plant."""

import numpy as np
from scipy.spatial.transform import Rotation
from navigation.common.frames import transform, invert, ned_to_enu
from simulation.bridge.exchange_contracts import SensorTruth
from simulation.bridge.matlab_session import MatlabSession


class MockExternalModel:
    def __init__(self, config=None):
        self.config = config or {}
        self.timestamp = 0.0
        self.position = np.array(
            self.config.get("initial_position", [2.0, -1.0, 30.0]), float
        )
        self.velocity = np.zeros(3)
        self.estimate = None
        self.closed = False

    def deck_pose(self, t):
        return transform(
            Rotation.from_euler(
                "xyz", [0.025 * np.sin(0.6 * t), 0.02 * np.sin(0.8 * t), 0]
            ).as_matrix(),
            [0, 0, 0.2 * np.sin(0.7 * t)],
        )

    def read_sensor_truth(self):
        world = self.deck_pose(self.timestamp)
        camera = transform(np.diag([1.0, -1.0, -1.0]), self.position)
        # Derivative in deck coordinates includes deck angular/linear motion.
        eps = 1e-4
        future = self.deck_pose(self.timestamp + eps)
        relative = invert(world) @ camera
        next_camera = transform(camera[:3, :3], self.position + self.velocity * eps)
        v = ((invert(future) @ next_camera)[:3, 3] - relative[:3, 3]) / eps
        return SensorTruth(self.timestamp, world, relative, v)

    def write_navigation_estimate(self, estimate):
        if abs(estimate.timestamp - self.timestamp) > 1e-8:
            raise ValueError("feedback timestamp differs from plant timestamp")
        self.estimate = estimate

    def read_navigation_context(self):
        """Ideal synchronous ship attitude channel, separate from aircraft truth."""
        return dict(
            timestamp=self.timestamp,
            R_ned_deck=(
                ned_to_enu()[:3, :3] @ self.deck_pose(self.timestamp)[:3, :3]
            ).tolist(),
        )

    def advance(self, dt):
        if self.closed or dt <= 0:
            raise ValueError("model closed or dt invalid")
        acceleration = np.zeros(3)
        if self.estimate is not None and self.estimate.healthy:
            target = np.array([0.0, 0.0, self.config.get("target_height", 0.8)])
            acceleration = (
                self.config.get("kp", 0.25)
                * (target - self.estimate.T_deck_camera[:3, 3])
                - self.config.get("kd", 1.0) * self.estimate.velocity
            )
            maximum = self.config.get("max_acceleration", 2.0)
            acceleration *= min(1.0, maximum / max(np.linalg.norm(acceleration), 1e-9))
            acceleration = self.deck_pose(self.timestamp)[:3, :3] @ acceleration
        else:
            acceleration = -self.velocity  # health loss -> brake/hold, no blind descent
        self.position += self.velocity * dt + 0.5 * acceleration * dt * dt
        self.velocity += acceleration * dt
        self.timestamp += dt
        return self.read_sensor_truth()

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class MatlabExternalModel:
    def __init__(self, config=None):
        self.config = config or {}
        self.session = MatlabSession()
        self.initialization = None

    def __enter__(self):
        self.session.__enter__()
        try:
            self.initialization = self.session.call(
                "initialize_external_model", self.config
            )
        except Exception:
            self.session.__exit__()
            raise
        return self

    def read_sensor_truth(self):
        return SensorTruth.from_dict(self.session.call("read_sensor_truth"))

    def read_navigation_feedback(self):
        """Inspect wheel-plane-centre minus H-centre NED feedback (interface backend)."""
        return self.session.call("read_innerloop_feedback")

    def read_navigation_context(self):
        return self.session.call("read_navigation_context")

    def write_navigation_estimate(self, estimate):
        self.session.call("write_navigation_estimate", estimate.to_dict())

    def advance(self, dt):
        self.session.call("advance_external_model", float(dt))
        return self.read_sensor_truth()

    def __exit__(self, *exc):
        self.session.__exit__(*exc)


def create_adapter(backend="mock", config=None):
    if backend == "mock":
        return MockExternalModel(config)
    if backend == "matlab":
        return MatlabExternalModel(config)
    raise ValueError(f"unknown backend: {backend}")
