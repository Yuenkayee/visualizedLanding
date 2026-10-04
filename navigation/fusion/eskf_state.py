"""12D error state [dp,dv,dtheta,domega], right-multiplicative attitude error."""

from dataclasses import dataclass, field
import numpy as np
from navigation.common.frames import transform
from navigation.common.contracts import covariance, finite_array
from navigation.common.timestamps import validate_timestamp


@dataclass
class ESKFState:
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 30.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.diag([1.0, -1.0, -1.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(
        default_factory=lambda: np.diag(
            [100.0] * 3 + [4.0] * 3 + [0.2] * 3 + [0.05] * 3
        )
    )
    timestamp: float = 0.0

    @property
    def pose(self):
        return transform(self.rotation, self.position)

    @property
    def pose_covariance(self):
        i = [0, 1, 2, 6, 7, 8]
        return self.P[np.ix_(i, i)].copy()

    def __post_init__(self):
        self.position = finite_array(self.position, (3,), "position")
        self.velocity = finite_array(self.velocity, (3,), "velocity")
        self.angular_velocity = finite_array(
            self.angular_velocity, (3,), "angular_velocity"
        )
        self.timestamp = validate_timestamp(self.timestamp)
        self.rotation = transform(self.rotation)[:3, :3]
        self.P = covariance(self.P, 12)
