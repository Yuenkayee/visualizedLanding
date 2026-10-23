import json
import numpy as np
import pytest
from navigation.common.cg_signal import CgSignal
from main.analyze import analyze
from navigation.common.contracts import NavigationEstimate, GimbalSample
from navigation.common.frames import transform
from navigation.common.cg_signal import relative_cg
from main.cg_replay import CgReplay


def test_cg_geometry_velocity_and_replay_truth_isolation():
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [.12, -.13, .4]).as_matrix()
    Rb = Rotation.from_euler("xyz", [-.2, .1, -.7]).as_matrix()
    mount = transform(Rotation.from_euler("xyz", [.2, -.1, 1.2]).as_matrix(), [2, 0, 1.11])
    p = np.array([3., 4., 20.]); v = np.array([1., 2., -3.]); omega = np.array([.03, -.02, .1])
    w = np.array([.2, -.1, .3]); wg = np.array([.1, .02, -.04])
    Tbody = transform(Rb, p); Tcamera = Tbody @ mount
    camera_v = v + Rb @ np.cross(w, mount[:3, 3])
    camera_w = mount[:3, :3].T @ w + wg
    e = NavigationEstimate(0, Tcamera, camera_v, np.eye(6)*.01, True,
        angular_velocity_camera=camera_w,
        gimbal=GimbalSample(0, mount, wg, np.eye(3)*1e-5, True))
    geometry = dict(H_ship_body=[-45,0,-5.5])
    out = relative_cg(e, geometry, R, omega)
    offset=np.diag([1.,-1.,-1.]) @ geometry["H_ship_body"]
    np.testing.assert_allclose(out["cg_relative_estimate_ned"], R@(p+offset))
    np.testing.assert_allclose(out["navigation_velocity_cg_ned_mps"], R@(v+np.cross(omega,p+offset)))
    A=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
    row=dict(timestamp=0,ship_angular_velocity_deck=omega,
             truth=dict(T_world_deck=transform(A@R), T_deck_camera=Tcamera, T_body_camera=mount))
    cg=CgReplay(geometry); first=cg.update(row,e)
    # Scoring truth can change without changing the estimated signal.
    row["truth"]["T_deck_camera"]=transform(Tcamera[:3,:3],Tcamera[:3,3]+[10,20,30])
    second=CgReplay(geometry).update(row,e)
    assert first["cg_relative_estimate_50hz_ned"]==second["cg_relative_estimate_50hz_ned"]
    assert first["cg_relative_truth_ned"]!=second["cg_relative_truth_ned"]


@pytest.mark.parametrize("sample_stride", [1, 5, 10])
def test_causal_ramp_at_50hz_with_held_packets(sample_stride):
    filt = CgSignal()
    p, v = np.array([-95., 2., -50.]), np.array([4., -1., .2])
    for i in range(101):
        t, sample_t = i*.02, (i//sample_stride)*sample_stride*.02
        out = filt.step(t, p+v*sample_t, v, True, sample_t)
        np.testing.assert_allclose(out["cg_relative_estimate_50hz_ned"], p+v*t, atol=1e-12)
        assert out["cg_signal_valid"]
        assert out["cg_signal_age_s"] == pytest.approx(t-sample_t)


def test_corrections_are_bounded_and_not_reapplied_at_same_timestamp():
    filt = CgSignal()
    filt.step(0, [0, 0, 0], [0, 0, 0], True, 0)
    first = filt.step(.02, [2, 0, 0], [0, 0, 0], True, .02)
    assert first["cg_relative_estimate_50hz_ned"][0] <= .10
    assert not first["cg_signal_valid"]  # large smoothing lag is explicitly invalid
    assert filt.step(.02, [2, 0, 0], [0, 0, 0], True, .02) == first
    previous = first["cg_relative_estimate_50hz_ned"][0]
    for i in range(2, 101):
        result = filt.step(i*.02, [2, 0, 0], [0, 0, 0], True, i*.02)
        current = result["cg_relative_estimate_50hz_ned"][0]
        assert 0 <= current-previous <= .10+1e-12
        previous = current
    assert result["cg_signal_valid"] and abs(current-2)<1e-5


def test_invalid_stale_future_and_recovery_do_not_fabricate_validity():
    filt = CgSignal()
    result = filt.step(0, [10, 0, 0], [1, 0, 0], False, 0)
    assert not result["cg_signal_valid"] and np.isfinite(result["cg_signal_age_s"])
    # Initialization of a Simulink block can precede the first healthy t=0 packet.
    assert filt.step(0, [10, 0, 0], [1, 0, 0], True, 0)["cg_signal_valid"]
    assert not filt.step(.02, [np.nan]*3, [0]*3, False, .02)["cg_signal_valid"]
    for i in range(2, 51):
        result = filt.step(i*.02, [np.nan]*3, [0]*3, False, i*.02)
    np.testing.assert_allclose(result["cg_relative_estimate_50hz_ned"], [10.3, 0, 0])
    assert result["cg_signal_age_s"] == pytest.approx(1)
    result = filt.step(1.02, [100, 0, 0], [1, 0, 0], True, 1.02)
    assert not result["cg_signal_valid"]
    assert result["cg_relative_estimate_50hz_ned"][0] <= 10.40+1e-12
    assert not filt.step(1.04, [11, 0, 0], [1, 0, 0], True, 2)["cg_signal_valid"]
    with pytest.raises(ValueError, match="clock"):
        filt.step(0, [0]*3, [0]*3, True, 0)


def test_held_valid_packet_expires_without_refreshing_its_age():
    filt = CgSignal()
    for i in range(31):
        out = filt.step(i*.02, [0, 0, 0], [1, 0, 0], True, 0)
        assert out["cg_signal_valid"] == (i <= 15)
    np.testing.assert_allclose(out["cg_relative_estimate_50hz_ned"], [.3, 0, 0], atol=1e-12)
    assert out["cg_signal_age_s"] == pytest.approx(.6)


def test_analysis_separates_cg_raw_signal_and_masks(tmp_path):
    rows=[]
    for i in range(3):
        rows.append(dict(timestamp=i*.02, dt=.02, next_timestamp=(i+1)*.02,
            truth=dict(gear_position_deck=[0,0,10]), next_truth=dict(gear_position_deck=[0,0,10]),
            gear_estimate_deck_m=[0,0,10], feedback=dict(valid=True, gear_relative_ned_m=[0,0,-10]),
            cg_relative_truth_ned=[-45,0,-18.4], cg_relative_estimate_ned=[-44,0,-18.4],
            cg_relative_estimate_50hz_ned=[-43,0,-18.4], cg_signal_valid=i>0, cg_signal_age_s=0,
            navigation_velocity_cg_ned_mps=[0,0,0]))
    (tmp_path/"steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result=analyze(tmp_path)["cg_relative"]
    assert result["raw"]["all"]["rmse_3d_m"] == 1
    assert result["signal_50hz"]["valid"]["rmse_3d_m"] == 2
    assert result["signal_50hz"]["valid"]["samples"] == 2
    for name in ("cg_relative_ned_comparison.png", "cg_relative_ned_comparison.pdf",
                 "cg_distance_comparison.png", "cg_relative_ned.csv"):
        assert (tmp_path/name).is_file()
