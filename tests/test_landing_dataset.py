import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from navigation.common.frames import invert, ned_to_enu
from visual_modeling.trajectories.landing import (
    make_sortie,
    camera_geometry,
    optical_report,
    cruise_progress,
    STAGES,
)
from visual_training.data.generate_landing_dataset import plan_dataset, render_dataset
from visual_training.data.h_dataset import HDataset


def config():
    c = yaml.safe_load(Path("configs/landing_dataset.yaml").read_text())
    c["sorties_per_weather"] = 3
    c["frames_per_stage"] = 2
    c["trajectory_hz"] = 4
    c["camera"]["width"] = 320
    c["camera"]["height"] = 240
    c["mission"]["approach_duration_s"] = [10, 11]
    c["mission"]["hover_duration_s"] = [2, 3]
    c["mission"]["descent_duration_s"] = [5, 6]
    return c


def test_stages_camera_and_reference_points():
    c = config()
    s = make_sortie(c, 123, "fixture", "clear")
    assert s == make_sortie(c, 123, "fixture", "clear")
    rows = s["rows"]
    assert np.allclose(rows[0]["gear_relative_heading_m"], [-50, 0, 50])
    last_approach = [r for r in rows if r["stage"] == "approach"][-1]
    assert np.allclose(last_approach["gear_relative_heading_m"], [0, 0, 5])
    assert np.allclose(last_approach["gear_relative_ned_m"], [0, 0, -5])
    assert (
        np.allclose(rows[-1]["gear_relative_deck_m"], [0, 0, 0])
        and rows[-1]["touchdown"]
    )
    assert set(r["stage"] for r in rows) == set(STAGES)
    assert all(a["timestamp"] < b["timestamp"] for a, b in zip(rows, rows[1:]))
    mount = np.array(s["T_body_camera"])
    gear = np.array(c["geometry"]["gear_body_m"])
    for r in rows:
        deck = np.array(r["T_world_deck"])
        body = np.array(r["T_world_body"])
        camera = np.array(r["T_deck_camera"])
        assert np.allclose(invert(deck) @ body @ mount, camera)
        gear_world = body[:3, :3] @ gear + body[:3, 3]
        assert np.allclose(
            deck[:3, :3].T @ (gear_world - deck[:3, 3]), r["gear_relative_deck_m"]
        )
        assert np.allclose(
            ned_to_enu()[:3, :3] @ (gear_world - deck[:3, 3]),
            r["gear_relative_ned_m"],
        )
    assert (
        rows[-1]["T_deck_camera"][2][3] > 1.5
    )  # External belly camera stays above deck.


def test_cruise_and_hover_boundaries():
    x = np.linspace(0, 1, 1001)
    p = cruise_progress(x)
    v = np.gradient(p, x)
    assert p[0] == 0 and p[-1] == 1 and (np.diff(p) >= 0).all()
    assert np.ptp(v[200:800]) < 1e-10
    c = config()
    c["motion"]["wake_position_std_m"] = 0
    s = make_sortie(c, 1, "hold", "clear", window_wait=4)
    assert s["window_source"] == "provided_event" and s["stage_durations_s"][1] == 4
    hold = [r for r in s["rows"] if r["stage"] == "window_hold"]
    assert all(np.allclose(r["gear_relative_heading_m"], [0, 0, 5]) for r in hold)
    assert all(np.allclose(r["gear_relative_ned_m"], [0, 0, -5]) for r in hold)
    # The gravity-vertical offset must differ from the rocking deck normal.
    assert any(not np.allclose(r["gear_relative_deck_m"], [0, 0, 5]) for r in hold)
    with pytest.raises(ValueError):
        cruise_progress(0.5, 0)


def test_optical_coverage_and_bad_mount():
    c = config()
    s = make_sortie(c, 42, "view", "clear")
    report = optical_report(s, c["marker"])
    assert all(v["centre_in_frame_fraction"] == 1 for v in report.values())
    assert report["touchdown"]["four_corners_in_frame_fraction"] > 0.9
    assert np.allclose(
        np.asarray(s["T_body_camera"])[:3, 3], c["camera"]["position_body_m"]
    )
    assert all(v["centre_visible_fraction"] == 1 for v in report.values())
    assert all(v["four_corners_visible_fraction"] > 0.95 for v in report.values())
    valid_position = c["camera"]["position_body_m"]
    c["camera"]["position_body_m"] = [0, 0, 0.1]
    with pytest.raises(ValueError, match="inside/on fuselage"):
        camera_geometry(c)
    c["camera"]["position_body_m"] = valid_position
    c["geometry"]["gear_body_m"] = [0, 0, 0]
    with pytest.raises(ValueError):
        camera_geometry(c)


def test_airframe_ray_segments_and_external_mount():
    from visual_modeling.sensors.airframe_visibility import (
        airframe_primitives,
        airframe_occlusion,
        validate_external_camera,
    )

    c = config()
    primitives = airframe_primitives(c["geometry"])
    origin = np.array([0, 0, 1.18])
    validate_external_camera(origin, primitives)
    wheel = np.array([1.6, 1.6, 2.55])
    targets = np.array(
        [
            [0, 0, 4],
            [0, 0, -3],
            origin + 2 * (wheel - origin),
            origin + 0.5 * (wheel - origin),
        ]
    )
    assert airframe_occlusion(origin, targets, primitives).tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert airframe_occlusion([0, 0, 0.1], targets, primitives).all()


def test_cpu_airframe_silhouette():
    from visual_modeling.sensors.airframe_visibility import (
        airframe_primitives,
        projected_airframe_mask,
    )

    c = config()
    c["camera"]["optical_down_deg"] = 90
    c["camera"]["mounting_error_deg"] = 0
    for name in ["ship_roll_deg", "ship_pitch_deg", "ship_heave_m"]:
        c["motion"][name] = [0, 0]
    for name in [
        "wake_position_std_m",
        "wake_attitude_std_deg",
        "camera_vibration_deg",
        "helicopter_pitch_deg",
        "yaw_tracking_error_deg",
    ]:
        c["motion"][name] = 0
    sortie = make_sortie(c, 42, "silhouette", "clear")
    cal, mount = camera_geometry(c)
    pose = np.array(sortie["rows"][-1]["T_deck_camera"])
    mask = projected_airframe_mask(cal, pose, mount, airframe_primitives(c["geometry"]))
    wheel_camera = mount[:3, :3].T @ (np.array([1.6, 1.6, 2.55]) - mount[:3, 3])
    pixel = cal.K @ wheel_camera
    u, v = np.rint(pixel[:2] / pixel[2]).astype(int)
    assert mask[v, u]
    assert not mask[cal.height // 2, cal.width // 2]


def test_stratified_sorties_and_training_contract(tmp_path):
    c = config()
    c["weather"] = {k: c["weather"][k] for k in ["clear", "night"]}
    plan = plan_dataset(c, tmp_path)
    design = json.loads((tmp_path / "calibration" / "design.json").read_text())
    assert np.array(design["K"]).shape == (3, 3)
    assert np.allclose(
        np.array(design["T_body_camera"])[:3, 3], c["camera"]["position_body_m"]
    )
    report = render_dataset(c, plan, tmp_path, "cpu")
    assert report["frames"] == 36
    groups = {}
    for split in ["train", "val", "test"]:
        paths = json.loads((tmp_path / "splits" / f"{split}.json").read_text())
        rows = [json.loads(Path(p).read_text()) for p in paths]
        assert {r["weather"] for r in rows} == {"clear", "night"}
        assert {r["stage"] for r in rows} == set(STAGES)
        groups[split] = {r["sequence_id"] for r in rows}
        sample = HDataset(tmp_path / "splits" / f"{split}.json", tmp_path, 64)[0]
        assert sample["keypoints"].shape == (4, 2) and sample["image"].shape == (
            3,
            64,
            64,
        )
    assert not groups["train"] & groups["test"] and not groups["train"] & groups["val"]
    assert not groups["test"] & groups["val"]
    # Different complete sorties have different poses, not just renamed images.
    a = json.load(open(plan[0]["trajectory"]))
    b = json.load(open(plan[1]["trajectory"]))
    assert a["rows"][0]["T_deck_camera"] != b["rows"][0]["T_deck_camera"]


def test_readout_warp_keeps_label_coordinates_consistent():
    from navigation.common.contracts import CameraCalibration
    from navigation.common.frames import transform
    from scipy.spatial.transform import Rotation
    from visual_training.data.build_dataset import synthetic_frame
    from visual_modeling.blender.build_h_marker import marker_points
    from visual_modeling.blender.export_annotations import make_annotation
    from visual_modeling.sensors.landing_camera import apply_landing_effects

    cal = CameraCalibration([[180, 0, 160], [0, 180, 120], [0, 0, 1]], 320, 240)
    pose = transform(Rotation.from_euler("x", np.pi).as_matrix(), [0, 0, 15])
    following = pose.copy()
    following[0, 3] = 2
    rng = np.random.default_rng(2)
    image, mask = synthetic_frame(cal, pose, rng)
    annotation = make_annotation(
        0, "fixture", 0, "image", "mask", marker_points(), pose, cal
    )
    before = np.array(annotation["keypoints"])
    image, mask = apply_landing_effects(
        image,
        mask,
        annotation,
        dict(glare=0, sensor_noise=0),
        dict(exposure_s=0, readout_s=0.08),
        rng,
        pose,
        following,
        0.1,
        cal,
    )
    after = np.array(annotation["keypoints"])
    assert not np.allclose(before, after)
    for x, y in np.rint(after).astype(int):
        assert mask[y - 2 : y + 3, x - 2 : x + 3].max() > 127
    assert np.isfinite(after).all()
