"""Rigid calibrated camera rig; all navigation outputs refer to one fixed camera."""

from dataclasses import dataclass
import numpy as np
from navigation.common.contracts import CameraCalibration, finite_array, PoseMeasurement
from navigation.common.frames import transform, invert, skew


@dataclass
class RigCamera:
    camera_id: str
    calibration: CameraCalibration
    T_body_camera: np.ndarray

    def __post_init__(self):
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera id must be a nonempty string")
        self.T_body_camera = finite_array(self.T_body_camera, (4, 4), "camera mount")
        transform(self.T_body_camera[:3, :3], self.T_body_camera[:3, 3])
        if not np.allclose(self.T_body_camera[3], [0, 0, 0, 1]):
            raise ValueError("invalid mount last row")


class CameraRig:
    def __init__(self, cameras, reference_camera_id="C0"):
        cameras = list(cameras)
        self.cameras = {c.camera_id: c for c in cameras}
        if not cameras or len(self.cameras) != len(cameras):
            raise ValueError("camera ids must be unique and nonempty")
        if reference_camera_id not in self.cameras:
            raise ValueError("missing rig reference camera")
        self.reference_camera_id = reference_camera_id

    @property
    def reference(self):
        return self.cameras[self.reference_camera_id]

    def reference_to_camera(self, camera_id):
        """T_C0_Ci for converting a camera pose from the fixed reference pose."""
        return (
            invert(self.reference.T_body_camera) @ self.cameras[camera_id].T_body_camera
        )

    def measurement_to_reference(self, camera_id, measurement):
        S = invert(self.cameras[camera_id].T_body_camera) @ self.reference.T_body_camera
        T = measurement.T_deck_camera
        # Covariance convention: deck translation and right-local camera rotation.
        J = np.zeros((6, 6))
        J[:3, :3] = np.eye(3)
        J[:3, 3:] = -T[:3, :3] @ skew(S[:3, 3])
        J[3:, 3:] = S[:3, :3].T
        P = J @ measurement.covariance @ J.T
        return PoseMeasurement(
            measurement.timestamp,
            T @ S,
            (P + P.T) / 2,
            measurement.reprojection_error,
            camera_id,
        )

    def to_dict(self):
        return dict(
            reference_camera_id=self.reference_camera_id,
            T_reference_lidar=self.reference.calibration.T_camera_lidar.tolist(),
            cameras={
                key: dict(
                    K=c.calibration.K.tolist(),
                    width=c.calibration.width,
                    height=c.calibration.height,
                    distortion=c.calibration.distortion.tolist(),
                    T_body_camera=c.T_body_camera.tolist(),
                )
                for key, c in self.cameras.items()
            },
        )

    @classmethod
    def from_dict(cls, data, T_reference_lidar=None):
        rig = cls(
            [
                RigCamera(
                    key,
                    CameraCalibration(
                        v["K"], v["width"], v["height"], v.get("distortion", [0] * 5)
                    ),
                    v["T_body_camera"],
                )
                for key, v in data["cameras"].items()
            ],
            data["reference_camera_id"],
        )
        if T_reference_lidar is None:
            T_reference_lidar = data.get("T_reference_lidar")
        if T_reference_lidar is not None:
            rig.reference.calibration.T_camera_lidar = CameraCalibration(
                rig.reference.calibration.K,
                rig.reference.calibration.width,
                rig.reference.calibration.height,
                T_camera_lidar=T_reference_lidar,
            ).T_camera_lidar
        return rig


def gear_feedback_ned(estimate, T_body_reference, gear_body, R_ned_deck):
    """Wheel-centre minus H in NED; same lever arm/covariance as MATLAB convert."""
    mount = finite_array(T_body_reference, (4, 4), "reference mount")
    transform(mount[:3, :3], mount[:3, 3])
    if not np.allclose(mount[3], [0, 0, 0, 1]):
        raise ValueError("invalid reference mount last row")
    Rn = transform(R_ned_deck)[:3, :3]
    lever = mount[:3, :3].T @ (
        finite_array(gear_body, (3,), "gear body") - mount[:3, 3]
    )
    T = estimate.T_deck_camera
    relative = Rn @ (T[:3, 3] + T[:3, :3] @ lever)
    J = np.column_stack([Rn, -Rn @ T[:3, :3] @ skew(lever)])
    P = J @ estimate.covariance @ J.T
    return dict(
        timestamp=estimate.timestamp,
        gear_relative_estimate_ned=relative.tolist(),
        relative_distance_estimate_m=float(np.linalg.norm(relative)),
        relative_covariance_ned=((P + P.T) / 2).tolist(),
        feedback_valid=bool(estimate.healthy),
    )
