"""Common adapter API; mock is a bounded-acceleration point-mass landing plant."""

import numpy as np
from scipy.spatial.transform import Rotation
from navigation.common.frames import transform, invert
from navigation.common.contracts import GimbalSample
from navigation.common.gimbal import Gimbal, gimbal_config
from navigation.common.reference_points import body_point_estimate
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
        reference = self.config.get("position_reference", "body")
        if reference not in ("body", "gear"):
            raise ValueError("mock position_reference must be body or gear")
        self.reference_body = np.asarray(self.config.get("gear_body", [0., 0., 0.]) if reference == "gear" else [0., 0., 0.])
        self.gimbal = Gimbal(self.config.get("T_body_camera", np.eye(4)), self.config.get("gimbal"))
        self.encoder_rng = np.random.default_rng(self.config.get("encoder_seed", 73))
        self.gimbal.advance(0., self.body_rotation(0.), self.deck_pose(0.)[:3, :3])
        self.encoder = self.gimbal.sample(self.encoder_rng)

    def body_rotation(self, t):
        amplitude = np.asarray(self.config.get("attitude_amplitude_rad", [.6, .6, .6]))
        period = np.asarray(self.config.get("attitude_period_s", [3., 3.1, 2.9]))
        angles = amplitude * np.sin(2*np.pi*t/period + [0., .4, .9])
        return np.diag([1., -1., -1.]) @ Rotation.from_euler("xyz", angles).as_matrix()

    def read_gimbal(self):
        return self.encoder

    def deck_pose(self, t):
        return transform(
            Rotation.from_euler(
                "xyz", [0.025 * np.sin(0.6 * t), 0.02 * np.sin(0.8 * t), 0]
            ).as_matrix(),
            [0, 0, 0.2 * np.sin(0.7 * t)],
        )

    def read_sensor_truth(self):
        world = self.deck_pose(self.timestamp)
        mount = self.gimbal.sample().T_body_camera
        Rb = self.body_rotation(self.timestamp)
        body = transform(Rb, self.position-Rb @ self.reference_body)
        camera = body @ mount
        # Derivative in deck coordinates includes deck angular/linear motion.
        eps = 1e-4
        future = self.deck_pose(self.timestamp + eps)
        relative = invert(world) @ camera
        next_Rb = self.body_rotation(self.timestamp+eps)
        next_body = transform(next_Rb, self.position+self.velocity*eps-next_Rb @ self.reference_body)
        next_camera = next_body @ mount
        v = ((invert(future) @ next_camera)[:3, 3] - relative[:3, 3]) / eps
        point = np.asarray(self.config.get("gear_body", [0., 0., 0.]))
        gear = (invert(world) @ body @ transform(translation=point))[:3, 3]
        next_gear = (invert(future) @ next_body @ transform(translation=point))[:3, 3]
        return SensorTruth(self.timestamp, world, relative, v, mount, gear, (next_gear-gear)/eps)

    def write_navigation_estimate(self, estimate):
        if abs(estimate.timestamp - self.timestamp) > 1e-8:
            raise ValueError("feedback timestamp differs from plant timestamp")
        self.estimate = estimate

    def advance(self, dt):
        if self.closed or dt <= 0:
            raise ValueError("model closed or dt invalid")
        acceleration = np.zeros(3)
        if self.estimate is not None and self.estimate.healthy:
            target = np.array([0.0, 0.0, self.config.get("target_height", 0.8)])
            if self.estimate.gimbal is not None:
                position, velocity, _ = body_point_estimate(self.estimate, self.config.get("gear_body", [0., 0., 0.]))
            else:  # Legacy point-mass fixtures have coincident reference points.
                position, velocity = self.estimate.T_deck_camera[:3, 3], self.estimate.velocity
            acceleration = (
                self.config.get("kp", 0.25)
                * (target - position)
                - self.config.get("kd", 1.0) * velocity
            )
            maximum = self.config.get("max_acceleration", 2.0)
            acceleration *= min(1.0, maximum / max(np.linalg.norm(acceleration), 1e-9))
            acceleration = self.deck_pose(self.timestamp)[:3, :3] @ acceleration
        else:
            acceleration = -self.velocity  # health loss -> brake/hold, no blind descent
        self.position += self.velocity * dt + 0.5 * acceleration * dt * dt
        self.velocity += acceleration * dt
        for t in np.linspace(self.timestamp, self.timestamp+dt, int(np.ceil(dt/.02))+1)[1:]:
            self.gimbal.advance(t, self.body_rotation(t), self.deck_pose(t)[:3, :3])
        self.timestamp += dt
        self.encoder = self.gimbal.sample(self.encoder_rng)
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
        self.config = {**self.config, "gimbal": gimbal_config(self.config.get("gimbal"))}
        self.session = MatlabSession()

    def __enter__(self):
        self.session.__enter__()
        try:
            self.session.call("initialize_external_model", self.config)
        except Exception:
            self.session.__exit__()
            raise
        return self

    def read_sensor_truth(self):
        return SensorTruth.from_dict(self.session.call("read_sensor_truth"))

    def read_navigation_feedback(self):
        """Inspect wheel-plane-centre minus H-centre NED feedback (interface backend)."""
        return self.session.call("read_innerloop_feedback")

    def read_gimbal(self):
        return GimbalSample(**self.session.call("read_gimbal_telemetry"))

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
