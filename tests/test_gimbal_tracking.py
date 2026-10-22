import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from navigation.common.contracts import GimbalReference, GimbalSample, NavigationEstimate
from navigation.common.frames import transform
from navigation.common.gimbal import Gimbal, gimbal_config
from navigation.common.gimbal_tracking import GimbalTargetTracker

D = np.diag([1., -1., -1.])


def test_causal_centering_and_reference_dynamic_limits():
    tracker = GimbalTargetTracker(np.eye(4), {"tracking": {"centering_deadband_rad": 0.}})
    pose = transform(D, [-13., 1., 4.2])
    previous, rate = tracker.reference.copy(), np.zeros(3)
    for t in np.arange(0, 8, .02):
        command = tracker.update(t, pose, True)
        delta = Rotation.from_matrix(command.R_deck_camera @ previous.T).magnitude()
        assert delta <= .35 * .02 + 1e-12
        assert np.linalg.norm(tracker.rate - rate) <= .7 * .02 + 1e-12
        assert np.linalg.det(command.R_deck_camera) == pytest.approx(1)
        if t == 0:
            np.testing.assert_allclose(command.R_deck_camera, D)
        previous, rate = command.R_deck_camera.copy(), tracker.rate.copy()
    # Limited tilt still brings a formerly peripheral H close to the centre.
    direction = np.array([0., 0., .025]) - pose[:3, 3]
    direction /= np.linalg.norm(direction)
    assert np.arccos(direction @ tracker.reference[:, 2]) < .3
    assert tracker.limited
    before = tracker.reference.copy()
    tracker.update(t, transform(D, [100, 0, 1]), True)
    np.testing.assert_array_equal(tracker.reference, before)
    with pytest.raises(ValueError):
        tracker.update(t-.1, pose, True)


def test_invalid_or_stale_navigation_brakes_without_using_its_position():
    a = GimbalTargetTracker(np.eye(4))
    for t in np.arange(0, 2, .02):
        a.update(t, transform(D, [-20, 0, 10]), True)
    b = copy.deepcopy(a)
    for t in np.arange(2, 6, .02):
        a.update(t, transform(D, [1000, 2, -1]), False)
        b.update(t, transform(D, [-500, 30, 100]), False)
        np.testing.assert_array_equal(a.reference, b.reference)
    assert np.linalg.norm(a.rate) < 1e-5
    e = NavigationEstimate(6, transform(D, [-20, 0, 10]), np.zeros(3), np.eye(6), True,
                           gimbal=GimbalSample(5, np.eye(4)))
    a.apply(e)
    assert e.diagnostics["gimbal_tracking"]["status"] == "hold_invalid_navigation"
    assert e.gimbal_reference.timestamp == 6


def test_command_is_not_telemetry_and_keeps_physical_limits():
    mount = transform(np.eye(3), [2, 0, 1.11])
    cfg = dict(max_rate_rad_s=[.2]*3, max_acceleration_rad_s2=[.4]*3)
    g = Gimbal(mount, cfg)
    g.advance(0., D, np.eye(3))
    measured = g.sample().T_body_camera.copy()
    cmd = GimbalReference(0., Rotation.from_euler("y", .5).as_matrix() @ D)
    g.set_reference(cmd)
    np.testing.assert_array_equal(g.sample().T_body_camera, measured)
    previous = g.rates.copy()
    for t in np.arange(.002, 1, .002):
        g.advance(t, D, np.eye(3))
        assert max(abs(g.rates)) <= .2
        assert max(abs(g.rates-previous)) <= .4*.002 + 1e-12
        np.testing.assert_array_equal(g.sample().T_body_camera[:3, 3], mount[:3, 3])
        previous = g.rates.copy()
    with pytest.raises(ValueError, match="timestamp"):
        g.set_reference(cmd)


def test_disabled_tracking_and_partial_config():
    tracker = GimbalTargetTracker(np.eye(4), {"tracking": {"enabled": False}})
    for t in (0, 1, 2):
        command = tracker.update(t, transform(D, [-13, 0, 4]), True)
        np.testing.assert_array_equal(command.R_deck_camera, D)
    assert tracker.diagnostics["status"] == "disabled"
    assert gimbal_config({"tracking": {"max_rate_rad_s": .1}})["tracking"]["time_constant_s"] == .6
    with pytest.raises(ValueError):
        gimbal_config({"tracking": {"max_rate_rad_s": -1}})


def test_reference_motion_prediction_does_not_create_measurements():
    from navigation.common.contracts import CameraCalibration
    from navigation.fusion.eskf_state import ESKFState
    from navigation.pipeline import NavigationPipeline
    from visual_modeling.blender.build_h_marker import marker_points
    pipe = NavigationPipeline(CameraCalibration(np.eye(3), 10, 10), marker_points(),
                              state=ESKFState(rotation=np.eye(3)))
    pose, covariance = pipe.state.pose.copy(), pipe.state.P.copy()
    pipe.process_gimbal_reference(GimbalReference(0, Rotation.from_euler("x", 1).as_matrix(), [0, .2, 0]))
    np.testing.assert_array_equal(pipe.state.pose, pose)  # no commanded attitude reset
    np.testing.assert_array_equal(pipe.state.P, covariance)  # no pseudo-measurement
    assert not pipe.estimate().healthy
    pipe.advance(.02)
    np.testing.assert_allclose(pipe.state.rotation, Rotation.from_rotvec([0, .004, 0]).as_matrix())
    pipe.process_gimbal_reference(GimbalReference(.02, np.eye(3), [0, 0, 0]))
    stopped = pipe.state.rotation.copy()
    pipe.advance(.04)
    np.testing.assert_allclose(pipe.state.rotation, stopped)  # braking ends rotation
    assert not pipe.estimate().healthy
    with pytest.raises(ValueError, match="timestamp"):
        pipe.process_gimbal_reference(GimbalReference(0, np.eye(3)))


def test_central_band_and_extended_target_near_touchdown():
    from visual_modeling.trajectories.landing import camera_geometry
    from navigation.pose.solve_pnp import project_points
    from visual_modeling.blender.build_h_marker import marker_points
    import yaml
    from pathlib import Path
    c = yaml.safe_load(Path("configs/landing_dataset.yaml").read_text())
    cal, mount = camera_geometry(c)
    tracker = GimbalTargetTracker(mount)
    # Near contact, centre-only look-at would crop the nearest H edge.
    for t in np.arange(0, 8, .02):
        tracker.update(t, transform(tracker.nominal, [2, 0, 1.79]), True)
    uv = project_points(marker_points(), transform(tracker.reference, [2, 0, 1.79]), cal)
    assert (uv > 0).all() and (uv < [cal.width, cal.height]).all()
