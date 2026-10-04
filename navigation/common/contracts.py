"""Validated sensor/estimate contracts shared by rendering, replay and navigation."""

from dataclasses import dataclass, field
import numpy as np
from navigation.common.frames import transform
from navigation.common.timestamps import validate_timestamp


def finite_array(value, shape, name):
    value = np.array(value, dtype=float, copy=True)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return value


def covariance(value, n):
    a = finite_array(value, (n, n), "covariance")
    if not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() < -1e-9:
        raise ValueError("covariance must be symmetric positive semidefinite")
    return a


@dataclass
class CameraCalibration:
    K: np.ndarray
    width: int
    height: int
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    T_camera_lidar: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        self.K = finite_array(self.K, (3, 3), "K")
        if self.width <= 0 or self.height <= 0 or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("invalid camera intrinsics")
        if not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError("K last row must be [0,0,1]")
        self.distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if (
            self.distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(self.distortion).all()
        ):
            raise ValueError("invalid distortion")
        self.T_camera_lidar = finite_array(self.T_camera_lidar, (4, 4), "extrinsics")
        transform(self.T_camera_lidar[:3, :3], self.T_camera_lidar[:3, 3])
        if not np.allclose(self.T_camera_lidar[3], [0, 0, 0, 1]):
            raise ValueError("invalid extrinsic last row")

    @classmethod
    def from_config(cls, c):
        return cls(
            np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]),
            c["width"],
            c["height"],
            c.get("distortion", [0] * 5),
            c.get("T_camera_lidar", np.eye(4)),
        )


@dataclass
class CameraFrame:
    timestamp: float
    image: np.ndarray

    def __post_init__(self):
        self.timestamp = validate_timestamp(self.timestamp)
        self.image = np.asarray(self.image)
        if (
            self.image.dtype != np.uint8
            or self.image.ndim != 3
            or self.image.shape[2] != 3
        ):
            raise ValueError("image must be HxWx3 uint8 RGB")


@dataclass
class LidarFrame:
    timestamp: float
    points: np.ndarray

    def __post_init__(self):
        self.timestamp = validate_timestamp(self.timestamp)
        a = np.asarray(self.points, dtype=float)
        if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
            raise ValueError("points must be finite Nx3")
        self.points = a


@dataclass
class PoseMeasurement:
    timestamp: float
    T_deck_camera: np.ndarray
    covariance: np.ndarray
    reprojection_error: float = 0.0
    source: str = "vision"

    def __post_init__(self):
        self.timestamp = validate_timestamp(self.timestamp)
        self.T_deck_camera = finite_array(self.T_deck_camera, (4, 4), "pose")
        transform(self.T_deck_camera[:3, :3], self.T_deck_camera[:3, 3])
        if not np.allclose(self.T_deck_camera[3], [0, 0, 0, 1]):
            raise ValueError("invalid pose last row")
        self.covariance = covariance(self.covariance, 6)
        if not np.isfinite(self.reprojection_error) or self.reprojection_error < 0:
            raise ValueError("invalid reprojection error")


@dataclass
class NavigationEstimate:
    timestamp: float
    T_deck_camera: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    healthy: bool
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        PoseMeasurement(self.timestamp, self.T_deck_camera, self.covariance)
        self.velocity = finite_array(self.velocity, (3,), "velocity")

    def to_dict(self):
        return dict(
            timestamp=float(self.timestamp),
            T_deck_camera=self.T_deck_camera.tolist(),
            velocity=self.velocity.tolist(),
            covariance=self.covariance.tolist(),
            healthy=bool(self.healthy),
            diagnostics=self.diagnostics,
        )
