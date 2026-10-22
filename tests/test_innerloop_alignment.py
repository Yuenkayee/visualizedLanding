import copy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from navigation.common.camera_rig import gear_feedback_ned
from navigation.common.contracts import NavigationEstimate
from navigation.common.frames import transform
from simulation.bridge.exchange_contracts import SensorTruth
from simulation.offline.innerloop_alignment import (
    prepare_innerloop_config,
    verify_innerloop_feedback,
)
from simulation.offline.logger import read_log
from simulation.offline.run_closed_loop import run_closed_loop
from visual_training.data.generate_multi_camera_dataset import load_config, nominal_rig


def fixture_config():
    return yaml.safe_load(Path("configs/innerloop_navigation.yaml").read_text())


def test_innerloop_geometry_is_shared_and_conflicts_rejected():
    config = fixture_config()
    rc = load_config(config["camera_rig_config"])
    rig = nominal_rig(rc)
    plant = prepare_innerloop_config(config["plant"], rig, rc)
    assert np.allclose(plant["geometry"]["T_body_camera"], rig.reference.T_body_camera)
    assert plant["reference_camera_id"] == "C0"
    assert "geometry" not in config["plant"]  # caller config was not changed
    bad = copy.deepcopy(config["plant"])
    bad["geometry"] = {"gear_body": [0, 0, 99]}
    with pytest.raises(ValueError, match="conflicts"):
        prepare_innerloop_config(bad, rig, rc)
    bad = copy.deepcopy(plant)
    bad["geometry"]["T_body_camera"][0][3] += 0.1
    with pytest.raises(ValueError, match="reference"):
        prepare_innerloop_config(bad, rig, rc)
    with pytest.raises(ValueError, match="camera_rig_config"):
        prepare_innerloop_config(config["plant"], None, None)


def test_actual_feedback_mismatch_and_stale_time_rejected():
    rc = load_config("configs/multi_camera_dataset.yaml")
    rig = nominal_rig(rc)
    estimate = NavigationEstimate(
        0.2,
        transform(np.diag([1, -1, -1]), [1, 2, 20]),
        np.zeros(3),
        np.eye(6) * 0.01,
        True,
    )
    expected = gear_feedback_ned(
        estimate,
        rig.reference.T_body_camera,
        rc["geometry"]["gear_body_m"],
        np.diag([1, -1, -1]),
    )
    raw = dict(
        timestamp=0.2,
        gear_relative_ned_m=expected["gear_relative_estimate_ned"],
        covariance_ned=expected["relative_covariance_ned"],
        distance_m=expected["relative_distance_estimate_m"],
        valid=True,
    )
    feedback, error = verify_innerloop_feedback(raw, expected, 0.2)
    assert feedback["source"] == "innerLoop.slx" and error == 0
    bad = copy.deepcopy(raw)
    bad["gear_relative_ned_m"][0] += 0.1
    with pytest.raises(ValueError, match="differs"):
        verify_innerloop_feedback(bad, expected, 0.2)
    raw["timestamp"] = 0.1
    with pytest.raises(ValueError, match="timestamp"):
        verify_innerloop_feedback(raw, expected, 0.2)


def test_noncommensurate_events_have_one_current_state_feedback(monkeypatch, tmp_path):
    """A strict transaction fixture catches advance-before-feedback at sensor ticks."""
    records = []

    class StrictInterface:
        initialization = {"interface_contract": {"fixture": True}}

        def __init__(self):
            self.timestamp = 0
            self.phase = "read"
            self.estimate = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read_sensor_truth(self):
            assert self.phase == "read"
            self.phase = "observed"
            records.append(("read", self.timestamp))
            return SensorTruth(
                self.timestamp,
                np.eye(4),
                transform(np.diag([1, -1, -1]), [2, -1, 30]),
                np.zeros(3),
            )

        def read_navigation_context(self):
            return dict(timestamp=self.timestamp, R_ned_deck=np.diag([1, -1, -1]))

        def write_navigation_estimate(self, estimate):
            assert self.phase == "observed" and estimate.timestamp == self.timestamp
            self.phase = "feedback"
            self.estimate = estimate
            records.append(("write", self.timestamp))

        def read_navigation_feedback(self):
            assert self.phase == "feedback"
            f = gear_feedback_ned(
                self.estimate,
                rig.reference.T_body_camera,
                rc["geometry"]["gear_body_m"],
                np.diag([1, -1, -1]),
            )
            return dict(
                timestamp=self.timestamp,
                gear_relative_ned_m=f["gear_relative_estimate_ned"],
                distance_m=f["relative_distance_estimate_m"],
                covariance_ned=f["relative_covariance_ned"],
                valid=f["feedback_valid"],
            )

        def advance(self, dt):
            assert self.phase == "feedback" and dt > 0
            self.timestamp += dt
            self.phase = "read"
            records.append(("advance", self.timestamp))
            return self.read_sensor_truth()

    c = fixture_config()
    rc = load_config(c["camera_rig_config"])
    rig = nominal_rig(rc)
    monkeypatch.setattr(
        "simulation.offline.run_closed_loop.create_adapter",
        lambda *args: StrictInterface(),
    )
    c.update(
        log=str(tmp_path / "control.jsonl"),
        audit_log=str(tmp_path / "audit.jsonl"),
        metrics=str(tmp_path / "metrics.json"),
    )
    result = run_closed_loop(c)
    rows = read_log(c["audit_log"])
    assert len(rows) == 9
    assert result["transactions"]["advances"] == 8
    assert result["transactions"]["feedback_writes"] == 9
    assert result["navigation"]["samples"] == 7
    assert records[:2] == [("read", 0), ("write", 0)]
    assert [kind for kind, _ in records] == ["read", "write"] + [
        kind for _ in range(8) for kind in ("advance", "read", "write")
    ]
    assert any(r["events"] == ["lidar"] for r in rows)
    assert all(r["feedback"]["source"] == "innerLoop.slx" for r in rows)
    for row in rows:
        ops = row["operations"]
        assert (
            ops.index("read_current_state")
            < ops.index("navigation_solution")
            < ops.index("write_navigation_feedback")
        )
    assert (
        json.loads(Path(c["metrics"]).read_text())["closed_loop"]["dynamics_tested"]
        is False
    )
