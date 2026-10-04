import numpy as np
from navigation.common.contracts import PoseMeasurement
from navigation.common.frames import transform
from navigation.fusion.eskf_state import ESKFState
from navigation.fusion.eskf_predict import predict
from navigation.fusion.eskf_update_vision import update_vision
from navigation.fusion.eskf_update_lidar import update_lidar
from navigation.lidar.extract_deck_plane import extract_deck_plane
from visual_modeling.sensors.lidar_raycast import raycast_deck


def test_prediction_and_covariance():
    state = ESKFState(velocity=np.array([1.0, 2.0, -1.0]))
    predict(state, 1.0)
    assert np.allclose(state.position, [1, 2, 29])
    assert np.linalg.eigvalsh(state.P).min() > 0
    assert np.allclose(state.rotation.T @ state.rotation, np.eye(3))


def test_pose_update_and_outlier_rejection():
    state = ESKFState()
    prior = state.P.copy()
    pose = transform(state.rotation, [1, -1, 29])
    accepted, _ = update_vision(state, PoseMeasurement(0, pose, np.eye(6) * 0.01))
    assert accepted and np.linalg.norm(state.position - pose[:3, 3]) < 0.05
    assert np.trace(state.P) < np.trace(prior)
    bad = transform(state.rotation, [1000, 1000, 1000])
    saved = state.position.copy()
    assert not update_vision(state, PoseMeasurement(0, bad, np.eye(6) * 0.01))[0]
    assert np.array_equal(state.position, saved)


def test_lidar_ransac_with_outliers_and_unobserved_xy():
    state = ESKFState(position=np.array([0.0, 0.0, 20.0]))
    points = raycast_deck(transform(state.rotation, [0, 0, 19.0]))
    rng = np.random.default_rng(2)
    points += rng.normal(0, 0.01, points.shape)
    points = np.vstack([points, rng.uniform(-30, 30, (100, 3))])
    plane = extract_deck_plane(points)
    assert plane is not None and abs(plane.offset - 19) < 0.04
    xy = state.P[:2, :2].copy()
    accepted, _ = update_lidar(state, plane)
    assert accepted and abs(state.position[2] - 19) < 0.05
    assert np.allclose(state.P[:2, :2], xy)
