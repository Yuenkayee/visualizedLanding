"""Validated sensor/estimate contracts shared by rendering, replay and navigation."""

from dataclasses import dataclass, field
import numpy as np
from navigation.common.frames import transform, invert
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
    lidar_mount: str = "camera"
    T_body_lidar: np.ndarray | None = None

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
        if self.lidar_mount not in ("camera", "body"):
            raise ValueError("lidar_mount must be camera or body")
        if self.lidar_mount == "body":
            if self.T_body_lidar is None:
                raise ValueError("body-mounted LiDAR requires T_body_lidar")
            self.T_body_lidar = GimbalSample(0, self.T_body_lidar).T_body_camera

    def camera_lidar_at(self, gimbal=None, timestamp=None):
        if self.lidar_mount == "camera":
            return self.T_camera_lidar.copy()
        if gimbal is None or not gimbal.valid or (timestamp is not None and abs(gimbal.timestamp-timestamp) > 1e-8):
            raise ValueError("body LiDAR requires synchronized valid gimbal encoder")
        return invert(gimbal.T_body_camera) @ self.T_body_lidar

    @classmethod
    def from_config(cls, c):
        return cls(
            np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]),
            c["width"],
            c["height"],
            c.get("distortion", [0] * 5),
            c.get("T_camera_lidar", np.eye(4)),
            c.get("lidar_mount", "camera"),
            c.get("T_body_lidar"),
        )


@dataclass
class GimbalSample:
    """Encoder telemetry only, never aircraft/deck ground-truth pose.

    Covariance is camera-local right rotation error. omega is camera relative
    to BODY, expressed in camera, not aircraft inertial angular velocity.
    """
    timestamp: float
    T_body_camera: np.ndarray
    angular_velocity_camera: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    valid: bool = True
    saturated: bool = False

    def __post_init__(self):
        self.timestamp = validate_timestamp(self.timestamp)
        self.T_body_camera = finite_array(self.T_body_camera, (4, 4), "T_body_camera")
        transform(self.T_body_camera[:3, :3], self.T_body_camera[:3, 3])
        if not np.allclose(self.T_body_camera[3], [0, 0, 0, 1]):
            raise ValueError("invalid mount transform")
        self.angular_velocity_camera = finite_array(self.angular_velocity_camera, (3,), "mount angular velocity")
        self.covariance = covariance(self.covariance, 3)

    def to_dict(self):
        return dict(timestamp=self.timestamp, T_body_camera=self.T_body_camera.tolist(),
                    angular_velocity_camera=self.angular_velocity_camera.tolist(),
                    covariance=self.covariance.tolist(), valid=bool(self.valid), saturated=bool(self.saturated))


@dataclass
class CameraFrame:
    timestamp: float
    image: np.ndarray
    gimbal: GimbalSample | None = None

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
    gimbal: GimbalSample | None = None

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
class GimbalReference:
    """Slow reference command in DECK, distinct from actual encoder telemetry."""
    timestamp: float
    R_deck_camera: np.ndarray
    angular_velocity_deck: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.timestamp = validate_timestamp(self.timestamp)
        self.R_deck_camera = transform(self.R_deck_camera)[:3, :3]
        self.angular_velocity_deck = finite_array(self.angular_velocity_deck, (3,), "reference angular velocity")

    def to_dict(self):
        return dict(timestamp=float(self.timestamp), R_deck_camera=self.R_deck_camera.tolist(),
                    angular_velocity_deck=self.angular_velocity_deck.tolist())


@dataclass
class NavigationEstimate:
    timestamp: float
    T_deck_camera: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    healthy: bool
    diagnostics: dict = field(default_factory=dict)
    angular_velocity_camera: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gimbal: GimbalSample | None = None
    gimbal_reference: GimbalReference | None = None

    def __post_init__(self):
        PoseMeasurement(self.timestamp, self.T_deck_camera, self.covariance)
        self.velocity = finite_array(self.velocity, (3,), "velocity")
        self.angular_velocity_camera = finite_array(self.angular_velocity_camera, (3,), "angular velocity")

    def to_dict(self):
        return dict(
            timestamp=float(self.timestamp),
            T_deck_camera=self.T_deck_camera.tolist(),
            velocity=self.velocity.tolist(),
            covariance=self.covariance.tolist(),
            healthy=bool(self.healthy),
            diagnostics=self.diagnostics,
            angular_velocity_camera=self.angular_velocity_camera.tolist(),
            gimbal=self.gimbal.to_dict() if self.gimbal is not None else None,
            gimbal_reference=self.gimbal_reference.to_dict() if self.gimbal_reference is not None else None,
        )
