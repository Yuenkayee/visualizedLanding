import copy
import json

import cv2
import numpy as np
import pytest

from navigation.common.contracts import CameraCalibration, CameraFrame
from navigation.common.frames import transform
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from navigation.pose.solve_pnp import project_points
from navigation.vision.infer_h import HDetection, HDetector
from navigation.vision.mask_geometry import recover_from_mask, outer_quad
from visual_modeling.blender.build_h_marker import marker_points, marker_polygons


def fixture():
    cal = CameraCalibration(np.array([[300., 0, 320], [0, 300, 240], [0, 0, 1]]), 640, 480)
    pose = transform(np.diag([1., -1., -1.]), [0., 0., 8.])
    pixels = project_points(marker_points(), pose, cal)
    good = HDetection(pixels, np.zeros((480, 640), np.uint8), .99, True,
                      np.ones(4), np.zeros((64, 64), np.uint8), np.ones(2), np.zeros(2))
    cfg = dict(mask_geometry=False, require_gimbal=False, roi_recovery=True)
    pipe = NavigationPipeline(cal, marker_points(), cfg,
                              ESKFState(position=pose[:3, 3], rotation=pose[:3, :3]))
    return cal, pose, good, pipe


class Detector:
    def __init__(self, full, roi=None):
        self.full, self.roi, self.roi_calls = full, roi, 0
    def __call__(self, image):
        return self.full
    def detect_roi(self, image, detection, config):
        self.roi_calls += 1
        return self.roi, dict(status="evaluated", box=[0, 0, 640, 480])


def test_roi_preserves_original_pixel_coordinates_and_native_mask():
    image = np.zeros((960, 1280, 3), np.uint8)
    mask = np.zeros((512, 512), np.uint8)
    mask[180:205, 440:490] = 1
    full = HDetection(np.zeros((4, 2)), image[:, :, 0], 1., True, network_mask=mask)
    class CropDetector(HDetector):
        def __init__(self):
            self.model, self.session = object(), None
        def __call__(self, cropped):
            h, w = cropped.shape[:2]
            return HDetection(np.array([[.3, .3], [.7, .3], [.7, .7], [.3, .7]])*[w, h],
                              np.ones((h, w), np.uint8), .99, True,
                              network_mask=np.ones((512, 512), np.uint8))
    d, info = CropDetector().detect_roi(image, full, {})
    left, top, right, bottom = info["box"]
    assert 0 < left < right <= 1280 and 0 < top < bottom <= 960
    assert left <= 440*2.5 and right >= 490*2.5
    assert d.mask.shape == image.shape[:2] and d.network_mask.shape == (512, 512)
    scale, origin = d.mask_coordinates(image.shape)
    assert np.allclose(d.keypoints[0], origin + [.3, .3]*np.array([right-left, bottom-top]))
    assert np.allclose(scale, np.array([right-left, bottom-top])/512)
    assert d.mask[:top].sum() == 0
    full.network_mask[:, 0] = 1
    # A connected component clipped by the image edge must not invent an ROI.
    full.network_mask[180:205, :490] = 1
    assert CropDetector().detect_roi(image, full, {})[1]["status"] == "truncated_mask"


def test_mask_pose_uses_crop_offset_and_original_intrinsics():
    cal = CameraCalibration(np.array([[600., 0, 640], [0, 450, 480], [0, 0, 1]]), 1280, 960)
    pose = transform(np.diag([1., -1., -1.]), [0., 0., 8.])
    origin, scale = np.array([360., 270.]), np.array([1.1, .82])
    pixels = project_points(marker_points(), pose, cal)
    mask = np.zeros((512, 512), np.uint8)
    for polygon in marker_polygons():
        q = (project_points(polygon, pose, cal)-origin)/scale
        cv2.fillConvexPoly(mask, np.rint(q-.5).astype(np.int32), 1)
    detection = HDetection(pixels+12, np.empty((0, 0), np.uint8), .99, True,
                           network_mask=mask, network_scale=scale, network_origin=origin)
    recovered = recover_from_mask(detection, (960, 1280, 3), marker_points(), cal, pose, {})
    assert recovered is not None
    assert np.max(np.linalg.norm(recovered[0]-pixels, axis=1)) < 2


def test_primary_acceptance_skips_roi_and_does_not_mutate_detection():
    _, _, good, pipe = fixture()
    pipe.detector = Detector(good, good)
    before = copy.deepcopy(good)
    result = pipe.process_camera(CameraFrame(0., np.zeros((480, 640, 3), np.uint8)))
    assert result.healthy and pipe.detector.roi_calls == 0
    assert len(result.diagnostics["vision_attempts"]) == 1
    assert np.array_equal(before.keypoints, good.keypoints)


@pytest.mark.parametrize("rejection", ["invalid_quad", "innovation_gate"])
def test_rejected_primary_then_roi_commits_exactly_one_update(rejection):
    cal, pose, good, pipe = fixture()
    _, _, _, reference = fixture()
    image = np.zeros((480, 640, 3), np.uint8)
    for p in [pipe, reference]:
        p.detector = Detector(good)
        p.process_camera(CameraFrame(0., image))
    bad = copy.deepcopy(good)
    if rejection == "invalid_quad":
        bad.keypoints[[0, 1]] = bad.keypoints[[1, 0]]
    else:
        shifted = pose.copy(); shifted[0, 3] += 3.
        bad.keypoints = project_points(marker_points(), shifted, cal)
    pipe.detector = Detector(bad, good)
    reference.detector = Detector(good)
    result = pipe.process_camera(CameraFrame(.1, image))
    reference.process_camera(CameraFrame(.1, image))
    assert result.diagnostics["vision_primary_rejection"] == rejection
    assert result.diagnostics["vision_path"] == "roi" and result.healthy
    assert sum(a["vision_accepted"] for a in result.diagnostics["vision_attempts"]) == 1
    assert np.allclose(pipe.state.P, reference.state.P)
    assert np.allclose(pipe.state.position, reference.state.position)
    with pytest.raises(ValueError, match="strictly increasing"):
        pipe.process_camera(CameraFrame(.1, image))


def test_reacquisition_requires_consecutive_accepted_poses_and_preserves_timeout():
    _, _, good, pipe = fixture()
    image = np.zeros((480, 640, 3), np.uint8)
    pipe.detector = Detector(good)
    assert pipe.process_camera(CameraFrame(0., image)).healthy
    pipe.advance(1.02)
    assert not pipe.estimate().healthy
    first = pipe.process_camera(CameraFrame(1.2, image))
    assert first.diagnostics["vision_reacquired"] and not first.healthy
    assert first.diagnostics["measurement_age"] == 0
    pipe.detector = Detector(None)
    assert not pipe.process_camera(CameraFrame(1.3, image)).healthy
    pipe.detector = Detector(good)
    assert not pipe.process_camera(CameraFrame(1.4, image)).healthy
    assert pipe.process_camera(CameraFrame(1.5, image)).healthy


def test_small_mask_keeps_shape_gates_and_exposes_quality():
    cal = CameraCalibration(np.array([[90., 0, 64], [0, 90., 64], [0, 0, 1]]), 128, 128)
    pose = transform(np.diag([1., -1., -1.]), [0., 0., 24.])
    pixels = project_points(marker_points(), pose, cal)
    mask = np.zeros((128, 128), np.uint8)
    for polygon in marker_polygons():
        cv2.fillConvexPoly(mask, np.rint(project_points(polygon, pose, cal)-.5).astype(np.int32), 1)
    detail = {}
    assert outer_quad(mask, diagnostics=detail) is None
    assert detail["rejection"] == "minimum_side"
    detection = HDetection(pixels, mask, 1., True, network_mask=mask)
    detail = {}
    result = recover_from_mask(detection, (128, 128, 3), marker_points(), cal, pose,
        dict(mask_minimum_side_px=16, mask_adaptive_edges=True), detail)
    assert result is not None and detail["noise_multiplier"] > 1.
    solid = np.zeros_like(mask)
    cv2.fillConvexPoly(solid, np.rint(pixels).astype(np.int32), 1)
    detection.network_mask = solid
    assert recover_from_mask(detection, (128, 128, 3), marker_points(), cal, pose,
                             dict(mask_minimum_side_px=16, mask_adaptive_edges=True)) is None


def test_recovery_statistics_count_sensor_packets_not_held_diagnostics():
    from main.analyze import recovery_metrics
    rows = []
    for i in range(10):
        rows.append(dict(timestamp=i*.02, next_timestamp=(i+1)*.02, feedback=dict(valid=i<2 or i>=7),
            sensors=dict(camera=i%5==0), estimate=dict(diagnostics=dict(vision_accepted=True,
                vision_path="roi", vision_roi_attempted=True, vision_reacquired=True,
                vision_attempts=[dict(vision_path="full", vision_accepted=False, vision_rejection="invalid_quad"),
                                 dict(vision_path="roi", vision_accepted=True)]))))
    result = recovery_metrics(rows)
    assert result["invalid_duration_s"] == pytest.approx(.1)
    assert result["roi_attempted_frames"] == 2 and result["reacquisitions"] == 2
    assert result["accepted_paths"] == {"roi": 2}
    json.dumps(result, allow_nan=False)


def test_saved_sensor_replay_uses_packets_and_never_initializes_from_truth(tmp_path, monkeypatch):
    from main.replay_sensors import replay
    cal, pose, good, _ = fixture()
    source = tmp_path / "recording"
    (source / "sensors").mkdir(parents=True)
    camera = dict(K=cal.K.tolist(), width=640, height=480, distortion=[0.]*5,
                  T_camera_lidar=np.eye(4).tolist(), lidar_mount="camera", T_body_lidar=None)
    nav = dict(initial_position=pose[:3, 3].tolist(), initial_rotation=pose[:3, :3].tolist(),
               initial_velocity=[0., 0., 0.], filter_point_body=None, require_gimbal=False)
    meta = dict(camera=camera, navigation=nav, geometry=dict(gear_body=[0., 0., 0.]),
                marker=dict(width=6., length=6., stroke=1.2))
    (source / "run.json").write_text(json.dumps(meta))
    rows = []
    for i in range(3):
        t = i*.02
        rows.append(dict(iteration=i, timestamp=t, dt=.02, next_timestamp=t+.02,
            sensors=dict(camera=True, lidar=False), estimate=dict(gimbal=dict(timestamp=t, T_body_camera=np.eye(4).tolist())),
            truth=dict(gear_position_deck=[999., 999., 999.], T_world_deck=np.eye(4).tolist()),
            next_truth=dict(gear_position_deck=[999., 999., 999.])))
        np.savez(source / "sensors" / f"{i:06d}.npz", timestamp=t, image=np.zeros((480, 640, 3), np.uint8))
    (source / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr("navigation.pipeline.HDetector", lambda *a, **k: Detector(good))
    monkeypatch.setattr("main.replay_sensors.analyze", lambda directory: {})
    first, second = tmp_path / "first", tmp_path / "second"
    replay(source, first)
    for row in rows:
        row["truth"]["gear_position_deck"] = [-999., -999., -999.]
    (source / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    replay(source, second)
    a = [json.loads(s) for s in (first / "current/steps.jsonl").read_text().splitlines()]
    b = [json.loads(s) for s in (second / "current/steps.jsonl").read_text().splitlines()]
    assert a[0]["estimate"]["healthy"]
    assert np.linalg.norm(a[0]["gear_estimate_deck_m"]) < 20
    for x, y in zip(a, b):
        assert x["estimate"] == y["estimate"]
    np.savez(source / "sensors/000001.npz", timestamp=3., image=np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(ValueError, match="timestamp mismatch"):
        replay(source, tmp_path / "broken")
    assert json.loads((tmp_path / "broken/current/run.json").read_text())["status"] == "failed"
