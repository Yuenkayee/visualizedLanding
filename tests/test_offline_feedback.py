import numpy as np
from navigation.common.contracts import NavigationEstimate
from navigation.common.frames import transform
from simulation.bridge.external_model_adapter import MockExternalModel
from simulation.offline.run_closed_loop import run_closed_loop
from simulation.offline.logger import read_log


def estimate(p, healthy=True):
    return NavigationEstimate(
        0, transform(np.diag([1.0, -1.0, -1.0]), p), np.zeros(3), np.eye(6), healthy
    )


def test_estimate_feedback_changes_plant_not_truth():
    a = MockExternalModel()
    b = MockExternalModel()
    a.write_navigation_estimate(estimate([2, -1, 30]))
    b.write_navigation_estimate(estimate([-2, 1, 0.8]))
    a.advance(0.1)
    b.advance(0.1)
    assert not np.allclose(a.position, b.position)
    assert a.position[2] < b.position[2]


def test_unhealthy_feedback_holds():
    model = MockExternalModel()
    before = model.position.copy()
    model.write_navigation_estimate(estimate([0, 0, 30], False))
    model.advance(0.2)
    assert np.allclose(model.position, before)


def test_closed_loop_sensor_feedback_smoke(tmp_path):
    log = tmp_path / "closed.jsonl"
    result = run_closed_loop(
        dict(duration=1.0, log=str(log), metrics=str(tmp_path / "metrics.json"))
    )
    rows = read_log(log)
    assert result["navigation"]["samples"] == 21
    assert result["navigation"]["healthy_fraction"] > 0.5
    assert (
        rows[-1]["truth"]["T_deck_camera"][2][3]
        < rows[0]["truth"]["T_deck_camera"][2][3]
    )
    assert any(r["estimate"]["diagnostics"].get("vision_accepted") for r in rows)
    assert result["navigation"]["position_rmse"] < 2.0


def test_multi_camera_closed_loop_feedback(tmp_path):
    result = run_closed_loop(
        dict(
            duration=0.3,
            camera_rig_config="configs/multi_camera_dataset.yaml",
            navigation_config="configs/multi_camera_navigation.yaml",
            navigation={"checkpoint": None},
            log=str(tmp_path / "multi.jsonl"),
            metrics=str(tmp_path / "multi.json"),
        )
    )
    rows = read_log(tmp_path / "multi.jsonl")
    assert result["cameras"] == 4 and result["reference_camera_id"] == "C0"
    assert all("feedback" in r for r in rows)
    assert any(r["feedback"]["feedback_valid"] for r in rows)
    assert all(
        r["estimate"]["diagnostics"]["reference_camera_id"] == "C0" for r in rows
    )
