"""Experiment design/causality regressions, separate from accuracy claims."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from main.config import ROOT, load
from main.navigation_test_data import cases, navigation_config, trajectory
from main.navigation_test_replay import observation, image_quality, fault_active
from main.navigation_tests import validate_config, verify_record
from main.sensors import SensorGenerator
from navigation.common.contracts import GimbalSample
from navigation.common.frames import transform
from navigation.common.gimbal import Gimbal
from simulation.bridge.exchange_contracts import SensorTruth
from visual_modeling.blender.build_h_marker import marker_points


def config():
    return yaml.safe_load((ROOT/"configs/navigation_tests.yaml").read_text())


def test_design_counts_pairing_and_no_algorithm_overrides():
    c=config();validate_config(c)
    plan=cases(c)
    primary=[r for r in plan if r["suite"]=="main"]
    assert len(primary)==240
    assert sum(len(r["methods"]) for r in primary)==1680
    assert len({r["seed"] for r in primary})==240
    assert len({r["record_id"] for r in plan})==480
    for main in primary:
        pair=[r for r in plan if r["pair_id"]==main["pair_id"]]
        assert len({r["seed"] for r in pair})==1
        assert all(r["record_id"]==main["id"] for r in pair if r["gimbal_mode"]!="fixed")
    assert not {"navigation","checkpoint","pixel_sigma","max_position_std"}&c.keys()
    # Adding a method cannot alter independent physical seeds.
    subset=deepcopy(c);subset["methods"]=["N3"]
    assert [(r["id"],r["seed"]) for r in cases(subset)]==[(r["id"],r["seed"]) for r in plan]


def test_timing_validation_and_prior_is_prescribed_not_truth():
    c=config()
    for key,value in (("dt",.01),("camera_hz",11),("seeds_per_cell",0)):
        bad=deepcopy(c);bad[key]=value
        with pytest.raises(ValueError):validate_config(bad)
    base,landing,cal,geometry=load(c["base_config"])
    nav=navigation_config(base,landing,geometry,ROOT/base["checkpoint"],"cpu")
    assert nav["initial_position"]==landing["mission"]["initial_gear_heading_m"]
    original=yaml.safe_load((ROOT/base["navigation_config"]).read_text())
    for key in original:
        if key not in ("checkpoint","filter_point_body","initial_position","initial_velocity","initial_rotation"):
            assert nav[key]==original[key]


def test_trajectory_clock_and_gimbal_pair_keep_identical_body_deck_paths():
    c=config();base,landing,_,_=load(c["base_config"])
    short=deepcopy(landing)
    for key in ("approach_duration_s","hover_duration_s","descent_duration_s"):
        short["mission"][key]=[.2,.2]
    case=dict(seed=125,pair_id="a",weather="clear",motion_scale=1.,gimbal_mode="tracking")
    a=trajectory(short,case,.02)
    b=trajectory(short,{**case,"gimbal_mode":"fixed"},.02)
    np.testing.assert_allclose(np.diff(a[0]),.02,atol=1e-12)
    assert set(a[3])=={"approach","hover","descent"}
    np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])


def test_tick_random_streams_are_independent_and_labels_opt_in(tmp_path):
    c,landing,_,g=load();c.update(renderer="cpu",sensor_seed_by_tick=True,save_sensor_labels=True)
    landing["camera"].update(width=64,height=48)
    lidar=dict(horizontal_fov=50,vertical_fov=40,columns=8,rows=8,max_range=150,min_range=.2,range_std=.02,dropout=.05)
    first=SensorGenerator(c,landing,lidar,tmp_path);second=SensorGenerator(c,landing,lidar,tmp_path)
    truth=SensorTruth(0,np.eye(4),transform(Gimbal(g["T_body_camera"]).reference,[2,0,31.79]),np.zeros(3),np.asarray(g["T_body_camera"]))
    first.observe(truth,True,True,1) # extra branch cannot move tick 7's RNG stream
    a=first.observe(truth,True,True,7);b=second.observe(truth,True,False,7)
    np.testing.assert_array_equal(a["image"],b["image"])
    assert set(a)>={"label_keypoints","label_visibility","label_mask"}
    assert a["label_mask"].shape==(48,64)


def test_faults_change_input_only_and_do_not_depend_on_methods():
    mount=GimbalSample(1.,np.eye(4)).to_dict()
    row=dict(timestamp=1.,iteration=50,dt=.02,stage="descent",gimbal=mount,sensors=dict(camera=True,lidar=True))
    data=dict(image=np.ones((10,10,3),np.uint8),points=np.ones((100,3)))
    case=dict(seed=3,fault=dict(kind="lidar_point_loss",probability=.5))
    a=observation(row,data,case,[row]);b=observation(row,data,case,[row])
    np.testing.assert_array_equal(a[4],b[4]);assert 0<len(a[4])<100
    assert len(data["points"])==100 and row["gimbal"]==mount
    bias=dict(seed=3,fault=dict(kind="mount_bias",offset_m=[.1,0,0],rotation_deg=[1,0,0]))
    assert observation(row,data,bias,[row])[0].T_body_camera[0,3]==.1
    assert row["gimbal"]["T_body_camera"][0][3]==0
    assert not fault_active(dict(stage="hover",duration_s=1),row,[row])


def test_quality_does_not_confuse_geometry_fit_with_true_mask_iou():
    _,landing,cal,g=load();points=marker_points(6,6)
    pose=transform(Gimbal(g["T_body_camera"]).reference,[0,0,30])
    q=image_quality({},None,cal,pose,points)
    assert q["mask_iou"] is None and q["keypoint_error_px"] is None
    assert q["target_scale_px"]>0 and q["edge_margin_px"]>0


def test_record_integrity_rejects_modified_inputs(tmp_path):
    from main.navigation_test_data import write_json,digest
    write_json(tmp_path/"run.json",dict(status="complete"))
    (tmp_path/"steps.jsonl").write_text('{}\n')
    write_json(tmp_path/"sensor_hashes.json",{})
    write_json(tmp_path/"record_hashes.json",{name:digest(tmp_path/name) for name in ("run.json","steps.jsonl","sensor_hashes.json")})
    assert verify_record(tmp_path)
    (tmp_path/"steps.jsonl").write_text('{"tampered":true}\n')
    with pytest.raises(ValueError,match="checksum"):
        verify_record(tmp_path)


def test_source_keeps_exclusions_and_rejects_failed_fragments(tmp_path):
    from main.navigation_test_data import source_provenance,write_json
    rows=[dict(timestamp=i*.02,stage=stage,truth=dict(gear_position_deck=p))
          for i,(stage,p) in enumerate(zip(("approach","hover","descent"),([0,0,50],[0,0,5],[0,0,1])))]
    (tmp_path/"steps.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(tmp_path/"run.json",dict(status="complete",renderer="cpu",case=dict(weather="fog",seed=95,excluded_from_formal=True)))
    p=source_provenance(tmp_path)
    assert p["excluded_from_formal"] and p["source_weather"]=="fog" and p["source_seed"]==95
    write_json(tmp_path/"run.json",dict(status="failed"))
    with pytest.raises(ValueError,match="complete recording"):
        source_provenance(tmp_path)
    write_json(tmp_path/"run.json",dict(status="complete",planned_duration_s=60,completed_duration_s=.06))
    with pytest.raises(ValueError,match="planned duration"):
        source_provenance(tmp_path)


def test_resume_checks_results_not_just_complete_status(tmp_path):
    from main.navigation_tests import verify_result
    from main.navigation_test_data import write_json,digest
    write_json(tmp_path/"run.json",dict(status="complete",planned_duration_s=2,completed_duration_s=2))
    assert not verify_result(tmp_path)
    (tmp_path/"steps.jsonl").write_text('{"timestamp":0}\n')
    write_json(tmp_path/"result_hashes.json",{name:digest(tmp_path/name) for name in ("run.json","steps.jsonl")})
    assert verify_result(tmp_path)
    (tmp_path/"steps.jsonl").unlink()
    assert not verify_result(tmp_path)


@pytest.mark.parametrize("fault", [
    {"kind":"camera_gap","stage":"descent","duration_s":.08},
    {"kind":"joint_gap","stage":"descent","duration_s":.08},
    {"kind":"stale_encoder","stage":"descent","start_fraction":.2,"duration_s":.08,"offset_s":.02},
    {"kind":"delayed_camera","delay_s":.04},
    {"kind":"delayed_lidar","delay_s":.04},
    {"kind":"mount_bias","offset_m":[.05,0,0],"rotation_deg":[.5,0,0]},
    {"kind":"ship_attitude_bias","rotation_deg":[.5,.5,0]},
    {"kind":"initial_position_bias","offset_m":[2,-2,1]},
    {"kind":"initial_attitude_bias","rotation_deg":[2,-2,3]},
])
def test_fault_replay_keeps_clock_and_does_not_crash(tmp_path,fault):
    from main.navigation_test_data import write_json,clean,camera_dict
    from main.navigation_test_replay import replay_record
    from navigation.pose.solve_pnp import project_points
    from navigation.vision.infer_h import HDetection
    c=config();c.update(device="cpu",renderer="cpu")
    base,landing,cal,g=load(c["base_config"])
    nav=navigation_config(base,landing,g,None,"cpu")
    nav["checkpoint"]=None;nav["initial_position"]=[0,0,30]
    pose=transform(Gimbal(g["T_body_camera"]).reference,[2,0,31.79])
    pts=marker_points(6,6)
    class Detector:
        def __call__(self,image):
            return HDetection(project_points(pts,pose,cal),np.zeros(image.shape[:2],np.uint8),1.,True,np.ones(4))
    record=tmp_path/"record";(record/"sensors").mkdir(parents=True)
    write_json(record/"run.json",dict(camera=camera_dict(cal),geometry=g,marker=landing["marker"],
        navigation=nav,cg_signal=base["cg_signal"],planned_duration_s=.2))
    rows=[]
    for i in range(10):
        t=i*.02;encoder=GimbalSample(t,np.asarray(g["T_body_camera"]))
        truth=SensorTruth(t,np.eye(4),pose,np.zeros(3),np.asarray(g["T_body_camera"]),np.array([0,0,30]),np.zeros(3))
        filename=f"sensors/{i:06d}.npz"
        np.savez(record/filename,timestamp=t,image=np.zeros((cal.height,cal.width,3),np.uint8),points=np.zeros((0,3)))
        rows.append(dict(timestamp=t,dt=.02,iteration=i,stage="descent",truth=truth.to_dict(),gimbal=encoder.to_dict(),
            sensor_file=filename,sensors=dict(camera=True,lidar=True)))
    (record/"steps.jsonl").write_text(''.join(json.dumps(clean(r))+'\n' for r in rows))
    target=tmp_path/"result"
    result=replay_record(record,target,dict(id="test",method="N3",seed=4,fault=fault),c,"unused",detector=Detector())
    assert result["status"]=="complete"
    output=[json.loads(s) for s in (target/"steps.jsonl").read_text().splitlines()]
    assert [r["timestamp"] for r in output]==[r["timestamp"] for r in rows]
    if fault["kind"].startswith("delayed"):
        assert sum(r["diagnostics"]["delayed_packets_rejected"] for r in output)==8
    if fault["kind"]=="stale_encoder":
        assert any(not r["estimate_available"] for r in output)
