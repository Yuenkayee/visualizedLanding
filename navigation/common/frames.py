"""SE(3): T_ab maps coordinates in b to a; metres, radians, xyzw quaternion."""

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def transform(rotation=None, translation=None):
    T = np.eye(4)
    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
        if R.shape == (4,):
            R = Rotation.from_quat(R).as_matrix()
        if (
            R.shape != (3, 3)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            or np.linalg.det(R) < 0
        ):
            raise ValueError(
                "rotation must be a proper SO(3) matrix or xyzw quaternion"
            )
        T[:3, :3] = R
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    if not np.isfinite(T).all():
        raise ValueError("non-finite transform")
    return T


def invert(T):
    T = np.asarray(T, dtype=float)
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def transform_points(T, points):
    points = np.asarray(points, dtype=float)
    return points @ T[:3, :3].T + T[:3, 3]


def opencv_to_blender():
    return transform(np.diag([1.0, -1.0, -1.0]))


def ned_to_enu():
    return transform(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))
