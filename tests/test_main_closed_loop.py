"""Host contracts and sensor causality; flight/plant dynamics are not mocked as validation."""

import copy
import json
from pathlib import Path
import numpy as np
import pytest
from main.config import load, validate
from main.sensors import SensorGenerator, python_executable
from main.analyze import analyze
from simulation.bridge.exchange_contracts import SensorTruth
from navigation.common.frames import transform
from navigation.common.gimbal import Gimbal


def test_training_geometry_and_clock_contract():
    c, landing, cal, g = load()
    validate(c)
    np.testing.assert_allclose(g["gear_body"], [0, 0, 2.9])
    np.testing.assert_allclose(g["H_ship_body"], [-45, 0, -5.5])
    np.testing.assert_allclose(np.array(g["T_body_camera"])[:3, 3], [2, 0, 1.11])
    assert cal.lidar_mount == "body"
    for field, value in [
        ("dt", 0.031),
        ("duration", 0.111),
        ("camera_hz", 13),
        ("plant_step", 0),
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
                timestamp=i * 0.05,
                dt=0.05,
                next_timestamp=(i + 1) * 0.05,
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
