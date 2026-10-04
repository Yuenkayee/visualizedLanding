import numpy as np
from scipy.spatial.transform import Rotation
from navigation.common.contracts import CameraCalibration
from navigation.common.frames import transform
from navigation.pose.solve_pnp import project_points, solve_pnp
from navigation.pose.resolve_planar_ambiguity import resolve_planar_ambiguity
from navigation.pose.estimate_covariance import estimate_covariance
from navigation.vision.associate_keypoints import associate_keypoints
from visual_modeling.blender.build_h_marker import marker_points


def test_projection_pnp_roundtrip():
    cal = CameraCalibration(
        np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]]), 640, 480
    )
    truth = transform(
        Rotation.from_euler("xyz", [np.pi + 0.1, 0.15, 0.2]).as_matrix(), [2, -1, 25]
    )
    points = marker_points()
    u = project_points(points, truth, cal)
    result = resolve_planar_ambiguity(solve_pnp(points, u, cal), truth)
    assert result is not None
    assert np.allclose(result[0], truth, atol=1e-6)
    P = estimate_covariance(points, truth, cal)
    assert np.linalg.eigvalsh(P).min() > 0


def test_h_symmetry_association_uses_prior():
    cal = CameraCalibration(
        np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]]), 640, 480
    )
    truth = transform(
        Rotation.from_euler("xyz", [np.pi + 0.2, 0.1, 0.3]).as_matrix(), [1, -1, 30]
    )
    points = marker_points()
    u = project_points(points, truth, cal)
    result = associate_keypoints(points, np.roll(u[::-1], 2, axis=0), cal, truth)
    assert np.allclose(result[0], truth, atol=1e-5)
