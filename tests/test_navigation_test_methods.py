import copy
import json
import sys
from types import SimpleNamespace
from inspect import signature

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

import main.navigation_test_methods as methods
import navigation.pipeline as core
from navigation.common.contracts import (
    CameraCalibration, CameraFrame, GimbalReference, GimbalSample, LidarFrame,
)
from navigation.common.frames import invert, transform, transform_points
from navigation.fusion.eskf_state import ESKFState
from navigation.fusion.measurement_gating import update as measurement_update
from navigation.pose.estimate_covariance import estimate_covariance
from navigation.pose.solve_pnp import project_points
from navigation.vision.infer_h import HDetection, HDetector
from visual_modeling.blender.build_h_marker import marker_points, marker_polygons


class Detector:
    """Scripted stateless inference output; counters are only test observations."""

    def __init__(self, full, roi=None):
        self.full, self.roi = full, roi
        self.calls = self.roi_calls = 0

    def __call__(self, image):
        self.calls += 1
        return self.full

    def detect_roi(self, image, detection, config):
        self.roi_calls += 1
        return self.roi, dict(status="evaluated", box=[0, 0, 640, 480])


@pytest.fixture
def scene():
    calibration = CameraCalibration(
        np.array([[300., 0, 320], [0, 300, 240], [0, 0, 1]]), 640, 480
    )
    rotation = np.diag([1., -1., -1.]) @ Rotation.from_euler("xyz", [.06, -.08, .03]).as_matrix()
    pose = transform(rotation, [.2, -.1, 8.])
    points = marker_points()
    image = np.zeros((480, 640, 3), np.uint8)
    mask = np.zeros((480, 640), np.uint8)
    for polygon in marker_polygons():
        cv2.fillConvexPoly(mask, np.rint(project_points(polygon, pose, calibration)).astype(np.int32), 1)
    image[mask != 0] = 255
    detection = HDetection(project_points(points, pose, calibration), mask, .99, True,
                           np.ones(4), mask.copy())
    state = ESKFState(position=pose[:3, 3], rotation=rotation)
    config = dict(mask_geometry=False, require_gimbal=False)
    return SimpleNamespace(cal=calibration, pose=pose, points=points, image=image,
                           detection=detection, state=state, config=config)


def build(scene, name="N3", config=None, state=None, detector=None):
    return methods.make_method(
        name, scene.cal, scene.points, scene.config if config is None else config,
        scene.state if state is None else state,
        detector=Detector(scene.detection) if detector is None else detector,
    )


def assert_same_state(actual, expected):
    for name in ("position", "velocity", "rotation", "angular_velocity", "P"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-12)
    assert actual.timestamp == expected.timestamp


def assert_same_estimate(actual, expected):
    for name in ("T_deck_camera", "velocity", "covariance", "angular_velocity_camera"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-12)
    assert actual.timestamp == expected.timestamp
    assert actual.healthy == expected.healthy
    for key, value in expected.diagnostics.items():
        assert actual.diagnostics[key] == value


def forbidden(*args, **kwargs):
    pytest.fail("disabled algorithm path was called")


def test_n0_classical_detector_keeps_native_backend_and_never_loads_checkpoint(scene):
    cfg = {**scene.config, "checkpoint": "/does/not/exist.pt"}
    ignored = Detector(None)
    method = build(scene, "N0", config=cfg, detector=ignored)
    native = core.NavigationPipeline(scene.cal, scene.points, {**cfg, "checkpoint": None},
                                     copy.deepcopy(scene.state))
    assert isinstance(method.detector.backend, HDetector)
    assert method.detector.model is None and method.detector.session is None
    assert method.config == {**cfg, "checkpoint": None}
    for timestamp in (0., .1):
        frame = CameraFrame(timestamp, scene.image)
        actual = method.process_camera(frame)
        assert_same_estimate(actual, native.process_camera(frame))
        assert_same_state(method.state, native.state)
    assert method.last_detection is not None and not method.last_detection.ordered
    assert ignored.calls == 0
    assert cfg["checkpoint"] == "/does/not/exist.pt"


@pytest.mark.parametrize("name,overrides", [
    ("N1", dict(mask_geometry=False, roi_recovery=False, filter_point_body=None,
                gimbal_reference_feedforward=False)),
    ("N2", {}), ("N3", {}),
    ("A1", dict(mask_geometry=False, roi_recovery=False)),
    ("A2", dict(gimbal_reference_feedforward=False)),
    ("A3", {}),
])
def test_exact_overrides_and_shared_model_keep_config_and_state_isolated(scene, monkeypatch, name, overrides):
    cfg = dict(checkpoint="weights.pt", device="cpu", mask_geometry=True,
               roi_recovery=True, filter_point_body=[.1, .2, .3],
               gimbal_reference_feedforward=True, reacquisition_confirmations=4,
               pixel_sigma=2.7, acceleration_noise=1.23, custom={"untouched": [1, 2]})
    saved_cfg, saved_state = copy.deepcopy(cfg), copy.deepcopy(scene.state)
    constructor_calls = []

    def constructor(checkpoint, device):
        constructor_calls.append((checkpoint, device))
        assert checkpoint is None  # Injected detector must bypass model loading.
        return HDetector()

    monkeypatch.setattr(core, "HDetector", constructor)
    detector = Detector(scene.detection)
    first = build(scene, name, cfg, detector=detector)
    second = build(scene, name, cfg, detector=detector)
    assert first.config == {**saved_cfg, **overrides}
    assert first.detector.backend is second.detector.backend is detector
    assert first.detector is not second.detector
    assert len(constructor_calls) == 2
    first.config["custom"]["untouched"][0] = 999
    first.state.position[0] = 999
    first.state.P[0, 0] = 999
    first.diagnostics["private"] = True
    first.timing["geometry_s"] = 123.
    assert cfg == saved_cfg and second.config["custom"] == saved_cfg["custom"]
    assert_same_state(scene.state, saved_state)
    assert_same_state(second.state, saved_state)
    assert second.timing["geometry_s"] == 0 and "private" not in second.diagnostics


def test_unknown_method_fails_before_loading_model(scene, monkeypatch):
    monkeypatch.setattr(core, "HDetector", forbidden)
    with pytest.raises(ValueError, match="Unknown navigation method"):
        build(scene, "N9")


def test_n3_native_roi_attempts_and_full_detection_are_preserved(scene):
    bad = copy.deepcopy(scene.detection)
    bad.keypoints[[0, 1]] = bad.keypoints[[1, 0]]
    detector = Detector(bad, scene.detection)
    method = build(scene, detector=detector)
    native = core.NavigationPipeline(scene.cal, scene.points, scene.config, copy.deepcopy(scene.state))
    native.detector = Detector(bad, scene.detection)
    result = method.process_camera(CameraFrame(0., scene.image))
    assert_same_estimate(result, native.process_camera(CameraFrame(0., scene.image)))
    assert_same_state(method.state, native.state)
    assert method.last_detection is bad
    assert result.diagnostics["vision_primary_rejection"] == "invalid_quad"
    assert [a["vision_path"] for a in method.diagnostics["vision_attempts"]] == ["full", "roi"]
    assert sum(a["vision_accepted"] for a in method.diagnostics["vision_attempts"]) == 1
    assert detector.calls == detector.roi_calls == 1
    np.testing.assert_allclose(method.raw_pose.T_deck_camera, scene.pose, atol=1e-5)


def test_n3_equals_existing_pipeline_with_untouched_defaults(scene):
    method = build(scene, config={})
    native = core.NavigationPipeline(scene.cal, scene.points, state=copy.deepcopy(scene.state))
    native.detector = Detector(scene.detection)
    assert method.config == native.config == {}
    mount = GimbalSample(0., np.eye(4))
    cloud = deck_cloud(scene, mount, scene.cal)
    for timestamp in (0., .1, .2):
        mount = GimbalSample(timestamp, np.eye(4))
        assert method.process_gimbal(mount, timestamp) == native.process_gimbal(mount, timestamp)
        frame = CameraFrame(timestamp, scene.image, mount)
        assert_same_estimate(method.process_camera(frame), native.process_camera(frame))
        # Default mask recovery, refinement, fusion and health all run natively.
        assert "vision_mask" in method.diagnostics
        assert method.last_pose is not None
        lidar = LidarFrame(timestamp, cloud, mount)
        assert_same_estimate(method.process_lidar(lidar), native.process_lidar(lidar))
        assert_same_state(method.state, native.state)
        reference = GimbalReference(timestamp, scene.state.rotation, np.array([.01, .02, .03]))
        method.process_gimbal_reference(reference)
        native.process_gimbal_reference(reference)
        assert_same_state(method.state, native.state)
        assert_same_estimate(method.estimate(), native.estimate())


def test_a1_disables_primary_mask_and_every_recovery_path(scene, monkeypatch):
    monkeypatch.setattr(core, "recover_from_mask", forbidden)
    detector = Detector(scene.detection, scene.detection)
    detector.detect_roi = forbidden
    cfg = dict(mask_geometry=True, roi_recovery=True, mask_fallback_minimum_side_px=16)
    method = build(scene, "A1", config=cfg, detector=detector)
    assert method.process_camera(CameraFrame(0., scene.image)).healthy
    bad = copy.deepcopy(scene.detection)
    bad.keypoints[[0, 1]] = bad.keypoints[[1, 0]]
    detector.full = bad
    method.process_camera(CameraFrame(.1, scene.image))
    assert len(method.diagnostics["vision_attempts"]) == 1
    assert not method.diagnostics["vision_accepted"]
    assert not method.diagnostics["vision_roi_attempted"]


def test_a2_only_disables_reference_feedforward(scene):
    native, disabled = build(scene), build(scene, "A2")
    command = GimbalReference(0., scene.pose[:3, :3], np.array([.1, -.2, .3]))
    native.process_gimbal_reference(command)
    disabled.process_gimbal_reference(command)
    np.testing.assert_allclose(native.state.angular_velocity, scene.state.rotation.T @ command.angular_velocity_deck)
    np.testing.assert_array_equal(disabled.state.angular_velocity, scene.state.angular_velocity)
    np.testing.assert_array_equal(disabled.reference_rate_deck, np.zeros(3))


def test_a3_bypasses_only_recovery_health_without_numeric_tuning(scene):
    cfg = {**scene.config, "reacquisition_confirmations": 2}
    full, ablated = build(scene, config=cfg), build(scene, "A3", config=cfg)
    for method in (full, ablated):
        assert method.process_camera(CameraFrame(0., scene.image)).healthy
        method.process_gimbal(None, 1.1)
        assert not method.estimate().healthy
    assert not full.process_camera(CameraFrame(1.2, scene.image)).healthy
    assert ablated.process_camera(CameraFrame(1.2, scene.image)).healthy
    assert_same_state(full.state, ablated.state)
    assert full.health.max_age == ablated.health.max_age == 1.
    assert full.config == ablated.config == cfg
    assert full.recovery_required == ablated.recovery_required == 2
    assert full.recovering and ablated.recovering
    assert full.recovery_count == ablated.recovery_count == 1
    assert ablated.estimate().diagnostics["vision_recovery_pending"]
    assert not ablated.estimate().diagnostics["vision_recovery_gate_enabled"]
    ablated.state.P[:3, :3] = np.eye(3)*100.
    assert not ablated.estimate().healthy  # Covariance health remains enforced.
    ablated.state.P = full.state.P.copy()
    ablated.config["require_gimbal"] = True
    assert not ablated.estimate().healthy  # Encoder health remains enforced.


def test_raw_pose_is_camera_measurement_before_body_reference_transform(scene):
    mount = GimbalSample(0., transform(Rotation.from_euler("xyz", [.3, -.2, .1]).as_matrix(), [1., .4, .8]),
                         covariance=np.eye(3) * .003)
    point = np.array([.2, -.1, .05])
    lever = mount.T_body_camera[:3, :3].T @ (mount.T_body_camera[:3, 3] - point)
    state = ESKFState(position=scene.pose[:3, 3] - scene.pose[:3, :3] @ lever,
                      rotation=scene.pose[:3, :3])
    method = build(scene, config={**scene.config, "filter_point_body": point, "require_gimbal": True}, state=state)
    result = method.process_camera(CameraFrame(0., scene.image, mount))
    raw = method.raw_pose
    assert result.healthy and raw is not method.last_pose
    np.testing.assert_allclose(raw.T_deck_camera, scene.pose, atol=1e-5)
    np.testing.assert_allclose(raw.covariance, estimate_covariance(scene.points, raw.T_deck_camera, scene.cal))
    np.testing.assert_allclose(result.T_deck_camera, scene.pose, atol=1e-5)
    np.testing.assert_allclose(method.state.position, state.position, atol=1e-5)
    assert not np.allclose(raw.T_deck_camera[:3, 3], method.state.position)
    assert not np.allclose(raw.covariance, result.covariance)


def test_raw_pose_is_retained_on_native_innovation_rejection(scene):
    state = ESKFState(position=[20., 0., 8.], rotation=scene.state.rotation, P=np.eye(12)*1e-8)
    method = build(scene, config={**scene.config, "roi_recovery": False}, state=state)
    method.process_camera(CameraFrame(0., scene.image))
    assert method.diagnostics["vision_rejection"] == "innovation_gate"
    assert method.raw_pose is not None and method.last_pose is None
    np.testing.assert_allclose(method.raw_pose.T_deck_camera, scene.pose, atol=1e-5)
    assert_same_state(method.state, state)


def test_n1_raw_covariance_no_prediction_gates_or_body_transform(scene, monkeypatch):
    for name in ("predict", "update_vision", "update_lidar", "recover_from_mask"):
        monkeypatch.setattr(core, name, forbidden)
    monkeypatch.setattr(methods, "extract_deck_plane", forbidden)
    state = ESKFState(position=[100., -100., 30.], rotation=scene.state.rotation,
                      velocity=[3., 4., 5.], angular_velocity=[.1, .2, .3], P=np.eye(12)*1e-8)
    cfg = dict(require_gimbal=True, filter_point_body=[.2, .1, .3], mask_geometry=True,
               roi_recovery=True, pixel_sigma=2.3)
    detector = Detector(scene.detection)
    detector.detect_roi = forbidden
    method = build(scene, "N1", cfg, state, detector)
    assert method.estimate() is None and method.filter_point_body is None
    mount = GimbalSample(.2, transform(Rotation.from_euler("xyz", [.4, -.3, .2]).as_matrix(), [1, 2, 3]),
                         covariance=np.eye(3)*.1)
    result = method.process_camera(CameraFrame(.2, scene.image, mount))
    assert result.healthy and result.gimbal is mount
    np.testing.assert_allclose(result.T_deck_camera, scene.pose, atol=1e-5)
    np.testing.assert_allclose(result.covariance, estimate_covariance(scene.points, result.T_deck_camera, scene.cal, 2.3))
    np.testing.assert_array_equal(result.velocity, np.zeros(3))
    np.testing.assert_array_equal(result.angular_velocity_camera, np.zeros(3))
    assert result.diagnostics["single_frame"]
    assert not result.diagnostics["velocity_available"]
    assert not result.diagnostics["angular_velocity_available"]
    assert "vision_mahalanobis" not in result.diagnostics
    command = GimbalReference(.2, np.eye(3), np.array([9., 8., 7.]))
    method.process_gimbal_reference(command)
    assert method.process_lidar(LidarFrame(.2, np.zeros((100, 3)), mount)) is not None
    expected_state = copy.deepcopy(state)
    expected_state.timestamp = .2
    assert_same_state(method.state, expected_state)
    assert method.timing["filter_s"] == 0
    # Returned arrays cannot mutate the held measurement or next estimate.
    result.T_deck_camera[0, 3] = 999
    result.covariance[:] = 999
    np.testing.assert_allclose(method.estimate().T_deck_camera, scene.pose, atol=1e-5)


def test_n1_is_current_frame_only_and_rejects_duplicate_or_delayed_packets(scene):
    detector = Detector(scene.detection)
    method = build(scene, "N1", detector=detector)
    assert method.process_camera(CameraFrame(0., scene.image)) is not None
    method.reset_telemetry()
    method.process_gimbal(None, .1)
    assert method.estimate() is None
    detector.full = None
    assert method.process_camera(CameraFrame(.1, scene.image)) is None
    assert method.raw_pose is None and method.last_pose is None
    assert method.diagnostics["vision_primary_rejection"] == "no_target"
    detector.full = scene.detection
    assert method.process_camera(CameraFrame(.2, scene.image)) is not None
    with pytest.raises(ValueError, match="strictly increasing"):
        method.process_camera(CameraFrame(.2, scene.image))
    assert method.process_lidar(LidarFrame(.3, np.empty((0, 3)))) is None
    with pytest.raises(ValueError, match="timestamps must increase"):
        method.process_gimbal(None, .1)


@pytest.mark.parametrize("failure", ["no_target", "unordered_keypoints", "invisible_keypoint", "invalid_quad", "pnp_failed", "reprojection"])
def test_n1_geometry_quality_failures_never_return_held_pose(scene, monkeypatch, failure):
    detector = Detector(copy.deepcopy(scene.detection))
    method = build(scene, "N1", detector=detector)
    assert method.process_camera(CameraFrame(0., scene.image)) is not None
    if failure == "no_target":
        detector.full = None
    elif failure == "unordered_keypoints":
        detector.full.ordered = False
    elif failure == "invisible_keypoint":
        detector.full.visibility[0] = .1
    elif failure == "invalid_quad":
        detector.full.keypoints[0] = [-1, -1]
    else:
        monkeypatch.setattr(methods, "solve_pnp", lambda *a, **kw: [] if failure == "pnp_failed" else [(scene.pose, 100.)])
    assert method.process_camera(CameraFrame(.1, scene.image)) is None
    assert method.diagnostics["vision_rejection"] == failure
    assert method.diagnostics["vision_attempts"][0]["vision_rejection"] == failure
    assert detector.roi_calls == 0


def test_n1_ambiguity_uses_only_nominal_then_previous_accepted_orientation(scene, monkeypatch):
    rotation = scene.state.rotation
    other = rotation @ Rotation.from_euler("z", .5).as_matrix()
    # Deliberately make the correct orientation's position much less plausible.
    far = transform(rotation, [1000., -1000., 8.])
    close = transform(other, scene.state.position)
    candidates = [(close, .1), (far, .1)]
    monkeypatch.setattr(methods, "solve_pnp", lambda *a, **kw: candidates)
    monkeypatch.setattr(methods, "estimate_covariance", lambda *a: np.eye(6))
    method = build(scene, "N1")
    np.testing.assert_array_equal(method.process_camera(CameraFrame(0., scene.image)).T_deck_camera, far)
    accepted = transform(other, [5., 6., 8.])
    candidates[:] = [(accepted, .1)]
    method.process_camera(CameraFrame(.1, scene.image))
    # A rejected orientation must not replace the last accepted one.
    candidates[:] = [(far, 100.)]
    assert method.process_camera(CameraFrame(.2, scene.image)) is None
    candidates[:] = [(far, .1), (accepted, .1)]
    np.testing.assert_array_equal(method.process_camera(CameraFrame(.3, scene.image)).T_deck_camera, accepted)


@pytest.mark.parametrize("name", ["N1", "N3"])
def test_missing_gimbal_clears_full_detection_and_raw_pose(scene, name):
    cfg = {**scene.config, "require_gimbal": True}
    detector = Detector(scene.detection)
    method = build(scene, name, cfg, detector=detector)
    mount = GimbalSample(0., np.eye(4))
    assert method.process_camera(CameraFrame(0., scene.image, mount)) is not None
    result = method.process_camera(CameraFrame(.1, scene.image, mount))
    assert result is None if name == "N1" else not result.healthy
    assert method.last_detection is method.raw_pose is method.last_pose is None
    assert method.diagnostics["vision_rejection"] == "missing_or_stale_gimbal"
    assert method.diagnostics["vision_attempts"] == []
    assert detector.calls == 1


def deck_cloud(scene, mount, calibration):
    x, y = np.meshgrid(np.linspace(-3, 3, 12), np.linspace(-2, 2, 10))
    deck = np.c_[x.ravel(), y.ravel(), .003*np.sin(x.ravel()*4)]
    # Associated outliers, plus points outside the deck association bounds.
    deck = np.vstack([deck, [[0, 0, .5], [1, 1, .7], [99, 99, 0]]])
    return transform_points(invert(scene.pose @ calibration.camera_lidar_at(mount, 0.)), deck)


@pytest.mark.parametrize("lidar_mount", ["camera", "body"])
@pytest.mark.parametrize("body_point", [None, [.1, -.2, .3]])
def test_lidar_matches_native_transforms_and_reuses_one_plane(scene, monkeypatch, lidar_mount, body_point):
    mount = GimbalSample(0., transform(Rotation.from_euler("xyz", [.3, -.4, .2]).as_matrix(), [1., .1, .8]),
                         covariance=np.eye(3)*.001)
    calibration = copy.deepcopy(scene.cal)
    calibration.lidar_mount = lidar_mount
    calibration.T_body_lidar = transform(np.eye(3), [1.2, -.2, .9])
    calibration.T_camera_lidar = transform(Rotation.from_euler("y", .1).as_matrix(), [.1, .2, .3])
    cfg = {**scene.config, "filter_point_body": body_point}
    state = copy.deepcopy(scene.state)
    if body_point is not None:
        lever = mount.T_body_camera[:3, :3].T @ (mount.T_body_camera[:3, 3] - body_point)
        state.position -= state.rotation @ lever
    method = methods.make_method("N3", calibration, scene.points, cfg, state, detector=Detector(scene.detection))
    native = core.NavigationPipeline(calibration, scene.points, cfg, copy.deepcopy(state))
    frame = LidarFrame(0., deck_cloud(scene, mount, calibration), mount)
    original_extract, original_update = methods.extract_deck_plane, methods.update_lidar
    planes, used = [], []

    def extract(points, **kwargs):
        plane = original_extract(points, **kwargs)
        planes.append(plane)
        return plane

    def update(state, plane, **kwargs):
        used.append(plane)
        return original_update(state, plane, **kwargs)

    monkeypatch.setattr(methods, "extract_deck_plane", extract)
    monkeypatch.setattr(methods, "update_lidar", update)
    result = method.process_lidar(frame)
    assert_same_estimate(result, native.process_lidar(frame))
    assert_same_state(method.state, native.state)
    assert len(planes) == len(used) == 1 and planes[0] is used[0] is method.last_plane
    plane, associated = method.last_plane, method.lidar_associated_points
    assert len(associated) == len(frame.points) - 1
    assert method.diagnostics["lidar_associated_points"] == len(associated)
    assert method.diagnostics["lidar_inlier_ratio"] == pytest.approx(plane.inliers.mean())
    assert method.diagnostics["plane_inlier_ratio"] == pytest.approx(plane.inliers.mean())
    residual = np.sqrt(np.mean((associated[plane.inliers] @ plane.normal + plane.offset)**2))
    assert method.diagnostics["lidar_residual_m"] == pytest.approx(residual)
    assert method.diagnostics["lidar_inlier_count"] == int(plane.inliers.sum())
    json.dumps(method.diagnostics, allow_nan=False)
    # A later empty cloud must not report a previous plane's statistics.
    method.process_lidar(LidarFrame(0., np.empty((0, 3)), mount))
    assert method.last_plane is None and method.diagnostics["lidar_residual_m"] is None
    assert method.diagnostics["lidar_associated_points"] == 0
    assert method.diagnostics["lidar_rejection"] == "no_deck_plane"


def test_n2_ignores_lidar_geometry_and_keeps_native_prediction(scene, monkeypatch):
    monkeypatch.setattr(methods, "extract_deck_plane", forbidden)
    monkeypatch.setattr(methods, "associate_deck", forbidden)
    monkeypatch.setattr(methods, "update_lidar", forbidden)
    method = build(scene, "N2")
    reference = build(scene, "N3")
    reference.process_gimbal(None, .2)
    result = method.process_lidar(LidarFrame(.2, np.ones((50, 3))))
    assert_same_state(method.state, reference.state)
    assert not result.diagnostics["lidar_accepted"]
    assert result.diagnostics["lidar_rejection"] == "disabled"
    assert method.timing["geometry_s"] == 0


def test_stage_timing_separates_detector_pnp_and_filter_and_accumulates(scene, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(methods, "perf_counter", lambda: clock[0])
    original_pnp, original_update = core.associate_keypoints, core.update_vision

    class TimedDetector(Detector):
        def __call__(self, image):
            clock[0] += 2.
            return super().__call__(image)

        def detect_roi(self, *args):
            clock[0] += 3.
            return super().detect_roi(*args)

    def pnp(*args, **kwargs):
        clock[0] += 5.
        return original_pnp(*args, **kwargs)

    def update(*args, **kwargs):
        clock[0] += 7.
        return original_update(*args, **kwargs)

    monkeypatch.setattr(core, "associate_keypoints", pnp)
    monkeypatch.setattr(core, "update_vision", update)
    bad = copy.deepcopy(scene.detection)
    bad.keypoints[[0, 1]] = bad.keypoints[[1, 0]]
    method = build(scene, detector=TimedDetector(bad, scene.detection))
    for timestamp in (0., .1):
        method.process_camera(CameraFrame(timestamp, scene.image))
    assert method.timing == dict(inference_s=10., geometry_s=10., filter_s=14., camera_s=34., lidar_s=0.)
    held = method.estimate()
    timing = method.timing
    state, last_measurement = copy.deepcopy(method.state), method.health.last_measurement
    method.reset_telemetry()
    assert method.timing is timing and all(value == 0 for value in timing.values())
    assert method.last_detection is method.raw_pose is method.last_pose is None
    assert set(method.diagnostics) == {"vision_mahalanobis_threshold", "lidar_mahalanobis_threshold"}
    assert_same_state(method.state, state)
    assert method.health.last_measurement == last_measurement
    np.testing.assert_array_equal(method.estimate().T_deck_camera, held.T_deck_camera)


def test_cuda_timing_synchronizes_both_full_and_roi_without_mutating_shared_detector(scene, monkeypatch):
    events, clock = [], [0.]
    monkeypatch.setattr(methods, "perf_counter", lambda: clock[0])

    def synchronize(device):
        assert device == "cuda:1"
        events.append("sync")
        clock[0] += 1.

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)))

    class CudaDetector(Detector):
        device, model = "cuda:1", object()

        def __call__(self, image):
            events.append("full")
            clock[0] += 2.
            return super().__call__(image)

        def detect_roi(self, *args):
            events.append("roi")
            clock[0] += 3.
            return super().detect_roi(*args)

    detector = CudaDetector(None, scene.detection)
    method = build(scene, detector=detector)
    other = build(scene, detector=detector)
    method.process_camera(CameraFrame(0., scene.image))
    assert events == ["sync", "full", "sync", "sync", "roi", "sync"]
    assert method.timing["inference_s"] == 7.  # Includes completion sync, excludes pre-call wait.
    assert method.timing["camera_s"] == 9.
    assert all(value == 0 for value in other.timing.values())
    assert other.last_detection is None
    assert detector.model is CudaDetector.model


def test_timing_finalizes_even_when_detector_raises(scene, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(methods, "perf_counter", lambda: clock[0])

    def failing(image):
        clock[0] += 2.
        raise RuntimeError("inference failed")

    method = build(scene, detector=failing)
    with pytest.raises(RuntimeError, match="inference failed"):
        method.process_camera(CameraFrame(0., scene.image))
    assert method.timing["inference_s"] == method.timing["camera_s"] == 2.
    assert method._stages == [] and method.last_detection is None


@pytest.mark.parametrize("name", methods.METHODS)
def test_threshold_telemetry_matches_actual_native_gates_and_survives_reset(scene, name):
    method = build(scene, name)
    probability = signature(measurement_update).parameters["probability"].default
    for _ in range(2):
        assert method.diagnostics["vision_mahalanobis_threshold"] == (
            None if name == "N1" else pytest.approx(chi2.ppf(probability, 6))
        )
        assert method.diagnostics["lidar_mahalanobis_threshold"] == (
            None if name in ("N1", "N2") else pytest.approx(chi2.ppf(probability, 4))
        )
        method.reset_telemetry()
