import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from navigation.common.frames import transform, invert
from navigation.common.gimbal import Gimbal
from navigation.common.contracts import CameraCalibration, GimbalSample, LidarFrame, NavigationEstimate
from navigation.common.reference_points import body_point_estimate, body_pose_estimate
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.sensors.lidar_raycast import raycast_deck

D = np.diag([1., -1., -1.])


def test_stabilization_finite_bandwidth_and_fixed_optical_centre():
    neutral = transform(np.eye(3), [2., .1, 1.11])
    servo = Gimbal(neutral)
    rigid_error, error = [], []
    for t in np.arange(0, 6, .01):
        body = D @ Rotation.from_euler("xyz", .6*np.sin(2*np.pi*t/3 + np.array([0, .7, 1.4]))).as_matrix()
        deck = Rotation.from_euler("xyz", [.03*np.sin(t), .02*np.sin(t), 0]).as_matrix()
        m = servo.advance(t, body, deck)
        assert np.array_equal(m.T_body_camera[:3, 3], neutral[:3, 3])
        error.append(Rotation.from_matrix(D.T @ deck.T @ body @ m.T_body_camera[:3, :3]).magnitude())
        rigid_error.append(Rotation.from_matrix(D.T @ deck.T @ body).magnitude())
    assert max(error) < .03  # Includes multi-axis coupling/servo startup (<1.72 deg).
    assert np.sqrt(np.mean(np.square(error))) < .02*np.sqrt(np.mean(np.square(rigid_error)))
    assert max(error) > 1e-5  # finite actuator response, not ideal cancellation
    # Repeated reads cannot advance the actuator state.
    assert servo.sample().to_dict() == servo.sample().to_dict()


def test_servo_travel_velocity_acceleration_and_disabled_mode():
    g = Gimbal(np.eye(4), dict(initial_angles_rad=[0, 0, 0], max_rate_rad_s=[.2]*3,
                             max_acceleration_rad_s2=[.4]*3, angle_limits_rad=[.1]*3))
    body = D @ Rotation.from_euler("x", 1).as_matrix()
    previous = np.zeros(3)
    for t in np.arange(0, .4, .002):
        m = g.advance(t, body, np.eye(3))
        assert (np.abs(g.angles) <= .1).all()
        assert (np.abs(g.rates) <= .2).all()
        assert (np.abs(g.rates-previous) <= .4*.002+1e-12).all()
        previous = g.rates.copy()
    assert m.saturated
    assert g.advance(t, body, np.eye(3)).to_dict() == m.to_dict()
    rigid = Gimbal(transform(D, [1, 2, 3]), dict(enabled=False))
    assert np.allclose(rigid.advance(0, body, np.eye(3)).T_body_camera, rigid.neutral)
    with pytest.raises(ValueError):
        g.advance(-1, body, np.eye(3))


def test_reference_point_position_velocity_and_covariance():
    dt = 1e-6
    Rb = Rotation.from_euler("xyz", [.4, -.3, .7]).as_matrix()
    Rc = Rotation.from_euler("xyz", [-.5, .2, -.4]).as_matrix()
    wb, wg = np.array([.8, -.5, .4]), np.array([-.6, .3, .2])
    lc, point = np.array([2., .1, 1.11]), np.array([.2, -.1, 2.9])
    pb, vb = np.array([1., 2., 20.]), np.array([.3, .2, -.8])
    mount = GimbalSample(0, transform(Rc, lc), wg, np.eye(3)*.001)
    pose = transform(Rb @ Rc, pb+Rb @ lc)
    omega = Rc.T @ wb + wg
    vcam = vb + Rb @ np.cross(wb, lc)
    estimate = NavigationEstimate(0, pose, vcam, np.eye(6)*.01, True, angular_velocity_camera=omega, gimbal=mount)
    p, v, P = body_point_estimate(estimate, point)
    assert np.allclose(p, pb+Rb @ point)
    assert np.allclose(body_pose_estimate(estimate), transform(Rb, pb))
    next_Rb = Rb @ Rotation.from_rotvec(wb*dt).as_matrix()
    assert np.allclose(v, (pb+vb*dt+next_Rb @ point-p)/dt, atol=2e-6)
    # Numerical Jacobian checks BOTH camera and encoder right-angle conventions.
    J = np.zeros((3, 9))
    for k in range(9):
        perturbed = copy.deepcopy(estimate)
        if k < 3:
            perturbed.T_deck_camera[k, 3] += dt
        else:
            delta = np.eye(3)[(k-3)%3]*dt
            mat = perturbed.T_deck_camera if k < 6 else perturbed.gimbal.T_body_camera
            mat[:3, :3] = mat[:3, :3] @ Rotation.from_rotvec(delta).as_matrix()
        J[:, k] = (body_point_estimate(perturbed, point)[0]-p)/dt
    assert np.allclose(P, J @ np.diag([.01]*6+[.001]*3) @ J.T, atol=1e-7)


@pytest.mark.parametrize("lidar_mount", ["body", "camera"])
def test_lidar_uses_measured_mount_and_rejects_stale_encoder(lidar_mount):
    mount = GimbalSample(0, transform(Rotation.from_euler("xyz", [.5, -.4, .3]).as_matrix(), [1, .1, .8]))
    cal = CameraCalibration(np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.]]), 640, 480,
                            lidar_mount=lidar_mount, T_body_lidar=transform(np.eye(3), [1.2, -.2, .9]))
    pose = transform(D, [0, 0, 20])
    T_cl = cal.camera_lidar_at(mount, 0)
    if lidar_mount == "body":
        assert np.allclose(mount.T_body_camera @ T_cl, cal.T_body_lidar)
    cloud = raycast_deck(pose @ T_cl)
    point = np.zeros(3)
    p_ref = pose[:3, 3] + pose[:3, :3] @ mount.T_body_camera[:3, :3].T @ (point-mount.T_body_camera[:3, 3])
    pipe = NavigationPipeline(cal, marker_points(), dict(require_gimbal=True, filter_point_body=point), ESKFState(position=p_ref))
    pipe.process_lidar(LidarFrame(0, cloud, mount))
    assert pipe.diagnostics["lidar_accepted"]
    pipe.process_lidar(LidarFrame(.1, cloud, mount))
    assert not pipe.diagnostics["lidar_accepted"]
    assert not pipe.estimate().healthy
    if lidar_mount == "body":
        with pytest.raises(ValueError, match="synchronized"):
            cal.camera_lidar_at(mount, .1)
    with pytest.raises(ValueError, match="synchronized"):
        body_point_estimate(NavigationEstimate(.1, pose, np.zeros(3), np.eye(6), True, gimbal=mount), [0, 0, 0])


def test_filter_removes_camera_lever_translation_during_body_rocking():
    from navigation.common.contracts import PoseMeasurement
    cal = CameraCalibration(np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.]]), 640, 480)
    pipe = NavigationPipeline(cal, marker_points(), dict(filter_point_body=[0, 0, 0], require_gimbal=True),
                              ESKFState(position=np.array([0., 0., 20.])))
    lc = np.array([2., .1, 1.11])
    for t in np.arange(0, 3, .05):
        Rb = D @ Rotation.from_euler("xyz", [.6*np.sin(2*t), .4*np.sin(2.1*t), .5*np.sin(2.2*t)]).as_matrix()
        mount = GimbalSample(t, transform(Rb.T @ D, lc))
        camera = transform(D, np.array([0, 0, 20])+Rb @ lc)
        pipe.process_gimbal(mount, t)
        pipe.process_pose(PoseMeasurement(t, camera, np.eye(6)*1e-4))
        assert pipe.diagnostics["vision_accepted"]
        assert np.allclose(pipe.estimate().T_deck_camera, camera, atol=1e-8)
    assert np.linalg.norm(pipe.state.velocity) < 1e-8
    assert np.allclose(pipe.state.position, [0, 0, 20])


def test_replay_encoder_channel_does_not_use_pose_truth(tmp_path):
    import json
    from visual_training.data.build_dataset import build_dataset
    from simulation.offline.run_replay import run_replay

    paths = build_dataset(tmp_path, sequences=3, frames=3)
    sequence = json.loads(paths[0].read_text())["sequence_id"]
    config = dict(require_gimbal=True, filter_point_body=[0, 0, 0])
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert run_replay(tmp_path, sequence, first, config) == 3
    for path in paths:
        row = json.loads(path.read_text())
        row["T_deck_camera"][0][3] += 1000
        row["T_body_camera"] = np.eye(4).tolist()
        path.write_text(json.dumps(row))
    run_replay(tmp_path, sequence, second, config)
    estimates = lambda path: [json.loads(line)["estimate"] for line in path.read_text().splitlines()]
    assert estimates(first) == estimates(second)
