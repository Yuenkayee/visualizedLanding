"""Host contracts and sensor causality; flight/plant dynamics are not mocked as validation."""

import copy
import json
from pathlib import Path
import numpy as np
import pytest
from main.config import load, validate
from main.sensors import SensorGenerator, python_executable
from main.analyze import analyze, ned_comparison
from simulation.bridge.exchange_contracts import SensorTruth
from navigation.common.frames import transform
from navigation.common.gimbal import Gimbal


def test_training_geometry_and_clock_contract():
    c, landing, cal, g = load()
    validate(c)
    assert c["dt"] == c["plant_step"] == 0.02
    assert round(1 / (c["dt"] * c["camera_hz"])) == 5
    assert round(1 / (c["dt"] * c["lidar_hz"])) == 10
    np.testing.assert_allclose(g["gear_body"], [0, 0, 2.9])
    np.testing.assert_allclose(g["H_ship_body"], [-45, 0, -5.5])
    np.testing.assert_allclose(np.array(g["T_body_camera"])[:3, 3], [2, 0, 1.11])
    assert cal.lidar_mount == "body"
    for field, value in [
        ("dt", 0.031),
        ("dt", 0.05),
        ("duration", 0.111),
        ("camera_hz", 13),
        ("plant_step", 0),
        ("plant_step", 0.01),
        ("weather", "glare"),
    ]:
        bad = copy.deepcopy(c)
        bad[field] = value
        with pytest.raises(ValueError):
            validate(bad)


def test_renderer_preserves_virtual_environment_symlink(tmp_path):
    base = tmp_path / "base_python"
    base.write_text("")
    virtual = tmp_path / "venv/bin/python"
    virtual.parent.mkdir(parents=True)
    virtual.symlink_to(base)
    assert python_executable(virtual) == str(virtual)


def test_sensor_does_not_request_future_plant_state(tmp_path, monkeypatch):
    c, landing, cal, g = load()
    c["renderer"] = "cpu"
    lidar = dict(
        horizontal_fov=50,
        vertical_fov=40,
        columns=8,
        rows=8,
        max_range=150,
        min_range=0.2,
        range_std=0.02,
        dropout=0.05,
    )
    gen = SensorGenerator(c, landing, lidar, tmp_path)
    R = Gimbal(g["T_body_camera"]).reference
    pose = transform(R, [2, 0, 31.79])
    truth = SensorTruth(
        0, np.eye(4), pose, np.array([1.0, 2.0, -3.0]), np.array(g["T_body_camera"])
    )
    calls = []

    def effects(
        image,
        mask,
        annotation,
        weather,
        camera,
        rng,
        current,
        following,
        dt,
        calibration,
    ):
        calls.append((current.copy(), following.copy(), dt))
        return image, mask

    monkeypatch.setattr("main.sensors.apply_landing_effects", effects)
    result = gen.observe(truth, True, True, 0)
    assert set(result) == {"timestamp", "image", "points"}  # no truth labels fed back
    assert result["image"].dtype == np.uint8 and result["points"].shape[1] == 3
    current, following, dt = calls[0]
    np.testing.assert_allclose(
        following[:3, 3], current[:3, 3] + truth.velocity_deck * dt
    )
    np.testing.assert_allclose(following[:3, :3], current[:3, :3])
    np.testing.assert_allclose(truth.T_deck_camera, pose)


def test_analysis_separates_invalid_feedback_and_keeps_last_state(tmp_path):
    rows = []
    for i in range(2):
        rows.append(
            dict(
                timestamp=i * 0.02,
                dt=0.02,
                next_timestamp=(i + 1) * 0.02,
                truth=dict(gear_position_deck=[0, 0, 30]),
                gear_estimate_deck_m=[i, 0, 30],
                feedback=dict(valid=i == 0, gear_relative_ned_m=[i, 0, -30]),
                next_truth=dict(gear_position_deck=[0, 0, 29]),
            )
        )
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path)
    assert result["valid_fraction"] == 0.5
    assert result["valid"]["rmse_3d_m"] == 0
    assert result["all"]["rmse_3d_m"] == pytest.approx(2**-0.5)
    assert result["final_truth_gear_deck_m"] == [0, 0, 29]
    assert (tmp_path / "position_comparison.png").is_file()


def test_ned_analysis_rotates_truth_and_accepts_early_stop(tmp_path):
    from scipy.spatial.transform import Rotation
    deck_to_ned = Rotation.from_euler("xyz", [.2, -.3, .8]).as_matrix()
    world_from_ned = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    position = np.array([2., -1., 8.])
    truth = deck_to_ned @ position
    estimate = truth + [1., 2., 3.]
    row = dict(timestamp=0., dt=.02, next_timestamp=0., plant_stopped=True,
               truth=dict(gear_position_deck=position.tolist(),
                          T_world_deck=transform(world_from_ned @ deck_to_ned).tolist()),
               gear_estimate_deck_m=(position + deck_to_ned.T @ [1., 2., 3.]).tolist(),
               feedback=dict(valid=True, gear_relative_ned_m=estimate.tolist()),
               next_truth=dict(gear_position_deck=position.tolist()))
    np.testing.assert_allclose(ned_comparison([row])[0][0], truth)
    (tmp_path / "steps.jsonl").write_text(json.dumps(row))
    (tmp_path / "run.json").write_text(json.dumps(dict(
        termination_reason="model_stop", navigation_applied_to_control=False)))
    result = analyze(tmp_path)
    assert result["next_state_clock_checked"]
    assert result["duration_s"] == 0
    assert result["navigation_applied_to_control"] is False
    np.testing.assert_allclose(result["ned"]["all"]["rmse_xyz_m"], [1., 2., 3.])
    for name in ("gear_relative_ned_comparison.png", "gear_relative_ned_comparison.pdf", "gear_relative_ned.csv"):
        assert (tmp_path / name).is_file()
    row["gear_relative_truth_ned"] = truth.tolist()
    row["gear_relative_estimate_ned"] = estimate.tolist()
    np.testing.assert_allclose(ned_comparison([row])[0][0], truth)


def test_observer_config_contract():
    c, _, _, _ = load("main/navigation_observer.yaml")
    validate(c)
    assert c["evaluation_mode"] == "observer"
    assert c["outputs"]["helicopter_state"] == c["autonomous_subsystem"] + "/helicopter_state"
    assert c["ship_velocity_frame"] == "body"
    c["input_contract"] = "feedback"
    with pytest.raises(ValueError, match="observer mode"):
        validate(c)
