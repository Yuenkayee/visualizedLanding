import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from navigation.common.frames import (
    transform,
    invert,
    transform_points,
    opencv_to_blender,
    ned_to_enu,
)


def test_inverse_and_composition():
    T = transform(Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix(), [1, 2, 3])
    points = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(transform_points(invert(T), transform_points(T, points)), points)
    assert np.allclose(T @ invert(T), np.eye(4))


def test_camera_and_world_conventions():
    assert np.allclose(
        transform_points(opencv_to_blender(), [[0, 1, 1]]), [[0, -1, -1]]
    )
    assert np.allclose(transform_points(ned_to_enu(), [[1, 2, 3]]), [[2, 1, -3]])
    assert np.linalg.det(ned_to_enu()[:3, :3]) == 1


def test_reject_reflection():
    with pytest.raises(ValueError):
        transform(np.diag([1, 1, -1]))
