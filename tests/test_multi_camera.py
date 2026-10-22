import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from navigation.common.camera_rig import CameraRig, RigCamera, gear_feedback_ned
from navigation.common.contracts import (
    CameraCalibration,
    CameraFrame,
    LidarFrame,
    PoseMeasurement,
    NavigationEstimate,
)
from navigation.common.frames import transform, invert
from navigation.fusion.eskf_state import ESKFState
from navigation.multi_camera_pipeline import MultiCameraPipeline
from navigation.pose.solve_pnp import project_points
from navigation.vision.infer_h import HDetection, HDetector
from visual_modeling.blender.build_h_marker import marker_points
from visual_modeling.sensors.lidar_raycast import beam_directions, raycast_deck
from visual_training.data.generate_multi_camera_dataset import load_config, generate
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss


def rig_fixture():
    cal = CameraCalibration([[500, 0, 320], [0, 500, 240], [0, 0, 1]], 640, 480)
    return CameraRig(
        [
            RigCamera(
                "C0",
                cal,
                transform(
                    Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix(), [2, 0, 1]
                ),
            ),
            RigCamera(
                "C1",
                copy.deepcopy(cal),
                transform(
                    Rotation.from_euler("xyz", [0.12, -0.17, 0.27]).as_matrix(),
                    [4, -0.8, 0.9],
                ),
            ),
        ]
    )


def test_reference_covariance_matches_finite_difference():
    rig = rig_fixture()
    T = transform(Rotation.from_euler("xyz", [0.2, -0.3, 0.1]).as_matrix(), [1, -2, 10])
    rng = np.random.default_rng(4)
    a = rng.normal(size=(6, 6))
    P = a @ a.T * 0.01
    m = rig.measurement_to_reference("C1", PoseMeasurement(0, T, P))
    S = invert(rig.cameras["C1"].T_body_camera) @ rig.reference.T_body_camera
    assert np.allclose(m.T_deck_camera, T @ S)
    J = np.zeros((6, 6))
    for k in range(6):
        perturb = T.copy()
        epsilon = 1e-7
        if k < 3:
            perturb[k, 3] += epsilon
        else:
            axis = np.eye(3)[k - 3] * epsilon
            perturb[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(axis).as_matrix()
        out = perturb @ S
        J[:, k] = (
            np.r_[
                out[:3, 3] - m.T_deck_camera[:3, 3],
                Rotation.from_matrix(
                    m.T_deck_camera[:3, :3].T @ out[:3, :3]
                ).as_rotvec(),
            ]
            / epsilon
        )
    assert np.allclose(m.covariance, J @ P @ J.T, atol=1e-7)
    assert np.linalg.eigvalsh(m.covariance).min() > 0


def test_wheel_ned_feedback_and_rig_validation():
    rig = rig_fixture()
    body_deck = transform(
        Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix(), [1, 2, 8]
    )
    camera = body_deck @ rig.reference.T_body_camera
    estimate = NavigationEstimate(0, camera, np.zeros(3), np.eye(6) * 0.001, True)
    R = Rotation.from_euler("xyz", [0.04, -0.02, 0.6]).as_matrix() @ np.diag(
        [1, -1, -1]
    )
    gear = [0, 0, 2.9]
    feedback = gear_feedback_ned(estimate, rig.reference.T_body_camera, gear, R)
    expected = R @ (body_deck[:3, 3] + body_deck[:3, :3] @ gear)
    assert np.allclose(feedback["gear_relative_estimate_ned"], expected)
    assert feedback["feedback_valid"]
    assert np.linalg.eigvalsh(feedback["relative_covariance_ned"]).min() > 0
    with pytest.raises(ValueError):
        CameraRig([rig.reference, rig.reference])
    with pytest.raises(ValueError):
        CameraRig(list(rig.cameras.values()), "absent")
    bad = np.eye(4)
    bad[3, 0] = 1
    with pytest.raises(ValueError):
        RigCamera("bad", rig.reference.calibration, bad)


def pipeline_fixture():
    rig = rig_fixture()
    pose = transform(np.diag([1, -1, -1]), [0.3, -0.2, 20])
    pipe = MultiCameraPipeline(
        rig,
        marker_points(),
        state=ESKFState(position=pose[:3, 3], rotation=pose[:3, :3]),
    )
    detections = {}
    for index, key in enumerate(rig.cameras):
        pixels = project_points(
            marker_points(),
            pose @ rig.reference_to_camera(key),
            rig.cameras[key].calibration,
        )
        detections[index + 1] = HDetection(
            pixels, np.zeros((480, 640), np.uint8), 0.9, True, np.ones(4, bool)
        )
    pipe.detector = lambda image: detections[int(image[0, 0, 0])]

    def frames(t, ids=("C0", "C1")):
        return {
            key: CameraFrame(
                t, np.full((480, 640, 3), list(rig.cameras).index(key) + 1, np.uint8)
            )
            for key in ids
        }

    return pipe, frames, detections, pose


def test_camera_switch_preserves_reference_and_outlier_fallback():
    pipe, frames, detections, truth = pipeline_fixture()
    first = pipe.process_camera_bundle(frames(0, ("C0",)))
    assert first.diagnostics["vision_accepted"]
    second = pipe.process_camera_bundle(frames(0.05, ("C1",)))
    assert (
        second.diagnostics["camera_switched"]
        and second.diagnostics["selected_camera_id"] == "C1"
    )
    assert np.allclose(second.T_deck_camera, truth, atol=1e-5)
    assert (
        np.linalg.norm(first.T_deck_camera[:3, 3] - second.T_deck_camera[:3, 3]) < 1e-5
    )
    # A geometrically sound, small-reprojection outlier from C0 must not block C1.
    bad = truth.copy()
    bad[0, 3] += 8
    detections[1].keypoints = project_points(
        marker_points(), bad, pipe.rig.reference.calibration
    )
    third = pipe.process_camera_bundle(frames(0.1))
    assert third.diagnostics["vision_accepted"]
    assert third.diagnostics["cameras"]["C0"]["reason"] == "innovation_gate"
    assert third.diagnostics["selected_camera_id"] == "C1"
    detections[1] = detections[2] = None
    lost = pipe.process_camera_bundle(frames(2))
    assert not lost.healthy and not lost.diagnostics["vision_accepted"]


def test_synchronization_and_invisible_keypoints():
    pipe, frames, detections, _ = pipeline_fixture()
    async_frames = frames(0)
    async_frames["C1"].timestamp = 0.01
    with pytest.raises(ValueError, match="timestamp"):
        pipe.process_camera_bundle(async_frames)
    detections[1].keypoint_visibility[0] = False
    result = pipe.process_camera_bundle(frames(0))
    assert result.diagnostics["cameras"]["C0"]["reason"] == "incomplete_keypoints"
    assert result.diagnostics["selected_camera_id"] == "C1"
    with pytest.raises(ValueError, match="stale"):
        pipe.process_camera_bundle(frames(0))
    with pytest.raises(ValueError, match="unknown"):
        pipe.process_camera_bundle({"bad": frames(1)["C0"]})
    wrong = frames(1)
    wrong["C0"] = CameraFrame(1, np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="image size"):
        pipe.process_camera_bundle(wrong)
    assert pipe.state.timestamp == 0


def test_hysteresis_retains_current_until_margin_exceeded():
    pipe, frames, detections, pose = pipeline_fixture()
    # Identical geometry isolates the score margin from different lever arms.
    pipe.rig.cameras["C1"].T_body_camera = pipe.rig.reference.T_body_camera.copy()
    detections[2].keypoints = detections[1].keypoints.copy()
    pipe.process_camera_bundle(frames(0, ("C0",)))
    detections[1].confidence, detections[2].confidence = 0.8, 0.9
    assert (
        pipe.process_camera_bundle(frames(0.05)).diagnostics["selected_camera_id"]
        == "C0"
    )
    detections[1].confidence = 0.5
    result = pipe.process_camera_bundle(frames(0.1))
    assert result.diagnostics["selected_camera_id"] == "C1"
    assert np.allclose(result.T_deck_camera, pose, atol=1e-5)


def test_shared_lidar_uses_reference_extrinsic_after_camera_switch():
    pipe, frames, _, pose = pipeline_fixture()
    extrinsic = transform(Rotation.from_euler("z", 0.2).as_matrix(), [0.4, -0.3, 0.1])
    pipe.calibration.T_camera_lidar = extrinsic
    pipe.process_camera_bundle(frames(0, ("C1",)))
    directions = beam_directions(60, 40, 32, 16)
    cloud = raycast_deck(pose @ extrinsic, directions)
    result = pipe.process_lidar(LidarFrame(0, cloud))
    assert result.diagnostics["lidar_accepted"]
    assert np.allclose(result.T_deck_camera, pose, atol=1e-5)


def test_synchronized_dataset_offsets_and_sortie_disjointness(tmp_path):
    c = load_config("configs/multi_camera_dataset.yaml")
    c["sorties_per_weather"] = 3
    c["weather"] = {"clear": c["weather"]["clear"]}
    c["frames_per_stage"] = 2
    c["camera"].update(width=160, height=120)
    for key in ("approach_duration_s", "hover_duration_s", "descent_duration_s"):
        c["mission"][key] = [1, 1]
    c["touchdown_offset"]["centred_fraction"] = 0
    report = generate(c, tmp_path, "cpu")
    assert report["frames"] == 72 and report["bundles"] == 18
    groups = []
    for split in ("train", "val", "test"):
        labels = [
            json.loads(Path(p).read_text())
            for p in json.loads((tmp_path / "splits" / f"{split}.json").read_text())
        ]
        groups.append({r["sequence_id"] for r in labels})
        assert {r["camera_id"] for r in labels} == set(c["cameras"])
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    for sequence in set.union(*groups):
        views = [
            json.loads(
                (tmp_path / "trajectories" / key / f"{sequence}.json").read_text()
            )
            for key in c["cameras"]
        ]
        actual = CameraRig.from_dict(
            json.loads((tmp_path / "calibration" / f"{sequence}.json").read_text())
        )
        assert len({v["environment_seed"] for v in views}) == 1
        assert len({v["sensor_seed"] for v in views}) == 4
        for i in (0, -1):
            assert all(
                v["rows"][i]["T_world_body"] == views[0]["rows"][i]["T_world_body"]
                for v in views
            )
            assert all(
                v["rows"][i]["timestamp"] == views[0]["rows"][i]["timestamp"]
                for v in views
            )
        final = views[0]["rows"][-1]
        offset = np.array(views[0]["touchdown_offset_deck_m"])
        assert 0 < np.linalg.norm(offset[:2]) and offset[2] == 0
        assert np.allclose(final["gear_relative_deck_m"], offset)
        body, deck = np.array(final["T_world_body"]), np.array(final["T_world_deck"])
        assert np.allclose(
            (invert(deck) @ body)[:3, :3] @ c["geometry"]["gear_body_m"]
            + (invert(deck) @ body)[:3, 3],
            offset,
        )
        for view in views:
            key = view["camera_id"]
            assert np.allclose(view["T_body_camera"], actual.cameras[key].T_body_camera)
            assert np.allclose(
                view["rows"][-1]["T_deck_camera"],
                invert(deck) @ body @ actual.cameras[key].T_body_camera,
            )
        bundles = json.loads((tmp_path / "bundles" / f"{sequence}.json").read_text())
        assert all(set(b["annotations"]) == set(c["cameras"]) for b in bundles)
        with np.load(tmp_path / bundles[0]["lidar"]) as cloud:
            assert cloud["timestamp"] == bundles[0]["timestamp"]
    c["touchdown_offset"]["centred_fraction"] = 1
    from visual_training.data.generate_multi_camera_dataset import offset_landing
    from visual_modeling.trajectories.landing import make_sortie

    centred = offset_landing(
        make_sortie(c, 22, "centred", "clear"), c["touchdown_offset"]
    )
    assert np.allclose(centred["rows"][-1]["gear_relative_deck_m"], 0)


def test_visibility_loss_and_legacy_checkpoint(tmp_path):
    import torch

    torch.set_num_threads(2)
    model = HSegmentationKeypoints(channels=4, predict_visibility=True)
    target = dict(
        image=torch.zeros(2, 3, 32, 32),
        mask=torch.zeros(2, 1, 32, 32),
        keypoints=torch.full((2, 4, 2), 0.5),
        visibility=torch.zeros(2, 4, dtype=torch.bool),
    )
    loss, metrics = segmentation_keypoint_loss(model(target["image"]), target)
    loss.backward()
    assert torch.isfinite(loss) and metrics["visibility"] > 0
    assert torch.isfinite(model.visibility.weight.grad).all()
    assert metrics["keypoint"] == metrics["heatmap"] == 0
    legacy = HSegmentationKeypoints(channels=4)
    path = tmp_path / "old.pt"
    torch.save(
        dict(model=legacy.state_dict(), model_config=dict(channels=4, keypoints=4)),
        path,
    )
    detector = HDetector(path)
    assert detector.model.visibility is None
    assert "visibility_logits" not in detector.model(torch.zeros(1, 3, 32, 32))
