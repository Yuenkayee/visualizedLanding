import numpy as np
import pytest
from navigation.common.contracts import (
    CameraFrame,
    LidarFrame,
    CameraCalibration,
    PoseMeasurement,
)
from navigation.common.timestamps import require_monotonic
from simulation.offline.sensor_buffer import SensorBuffer
from simulation.offline.scheduler import Scheduler


@pytest.mark.parametrize("timestamp", [-1, float("nan"), float("inf")])
def test_invalid_timestamp(timestamp):
    with pytest.raises(ValueError):
        LidarFrame(timestamp, np.zeros((0, 3)))


def test_reject_invalid_sensor_shapes():
    with pytest.raises(ValueError):
        CameraFrame(0, np.zeros((10, 10, 3)))
    with pytest.raises(ValueError):
        LidarFrame(0, np.zeros((3, 2)))
    with pytest.raises(ValueError):
        CameraCalibration(np.diag([-1, 1, 1]), 640, 480)
    with pytest.raises(ValueError):
        PoseMeasurement(0, np.eye(4), -np.eye(6))


def test_timestamp_and_buffer_order():
    with pytest.raises(ValueError):
        require_monotonic(2, 1)
    b = SensorBuffer()
    b.push("lidar", LidarFrame(2, np.zeros((1, 3))))
    b.push("lidar", LidarFrame(1, np.zeros((1, 3))))
    assert [frame.timestamp for _, frame in b.pop_until(2)] == [1, 2]
    with pytest.raises(ValueError):
        b.push("lidar", LidarFrame(1, np.zeros((1, 3))))


def test_scheduler_no_drift_and_equal_time_order():
    events = list(Scheduler({"camera": 7, "control": 20}, 1))
    assert sum(e.kind == "camera" for e in events) == 8
    assert sum(e.kind == "control" for e in events) == 21
    assert [e.kind for e in events if e.timestamp == 0] == ["camera", "control"]
    assert all(a.timestamp <= b.timestamp for a, b in zip(events, events[1:]))
