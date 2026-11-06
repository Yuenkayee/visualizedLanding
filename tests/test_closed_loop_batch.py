"""Scientific denominators, censoring, pairing and batch persistence contracts."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from main.batch.common import ROOT, seeds, write_json
from main.batch.recording import RunawayMonitor
from main.batch.metrics import summarize, motion, nominal_point
from main.batch.statistics import paired, wilson
from main.batch.analysis import tables, balanced_description
from main.batch.__main__ import round_tasks, parser, create, supervise
from main.batch.worker import SessionLease


@pytest.fixture
def policy():
    c=yaml.safe_load((ROOT/"main/batch_closed_loop.yaml").read_text())
    c.update(weather=["clear","fog"],seeds_per_weather=2,bootstrap_samples=100)
    return c


def fake_case(path,policy,weather="clear",sea=5,method="C3",slot=0,candidate=True):
    path.mkdir(parents=True,exist_ok=True);(path/"simulink").mkdir(exist_ok=True)
    geometry=dict(gear_body=[0,0,2.9],H_ship_body=[-45,0,-5.5])
    batch=dict(group=f"{weather}_s{slot:02d}",slot=slot,replacement=0,sea=sea,method=method,
               **seeds(17,weather,slot,0))
    model=dict(nominal_cg_target_ned_m=[-45,0,-8.4],command_limit_mps=[1,1,1],
        wave_component=dict(omega=[1.],waveNumber=[.1],beta=[np.pi],amplitude=[sea/5],phase=[.5]))
    rows=[]
    for i in range(6):
        stage=min(i//2,2)
        state=dict(p=[.1+slot*.05,0,6-i],v=[.1,0,-.2],rd=np.diag([1,-1,-1]).tolist(),db=np.diag([1,-1,-1]).tolist(),H_up_m=5.5+.01*i,H_normal_mps=.02,
                   ship=[i,0,0,30,0,0,.01*i,.02*i,0,0,0,0],heli=[0]*12,velocity_ned_mps=[1,0,0])
        u=dict(stage=min((i+1)//2,2),cg_used=[-45,0,-8.4],target=[-45,0,-8.4],error=[0,0,0],control_valid=i!=2,
               command=[30+i*.01,0,0],relative_command=[i*.01,0,0],feedforward=[30,0,0],wave_command=[30,0,0])
        rows.append(dict(t=i*.02,t1=(i+1)*.02,stage=stage,s=state,s1=state,u=u,wall_s=.2,cg_true=[-45,0,-8.4],
            cg_raw=[-45,0,-8.4],cg_shadow=[-45,0,-8.4],nav_valid=i!=2,shaper_valid=i!=2,
            sampled_camera=i%5==0,sampled_lidar=i==0,diagnostics=dict(vision_accepted=True,lidar_accepted=True),
            stopped=candidate and i==5,stop_reason="StopCommand" if candidate and i==5 else ""))
    metadata=dict(config=dict(batch=batch,weather=weather,seed=batch["sensor_seed"]),geometry=geometry,
                  termination_reason="model_stop" if candidate else "duration_reached",final_timestamp=.12,elapsed_s=2.)
    write_json(path/"run.json",metadata);write_json(path/"requested.json",metadata["config"])
    write_json(path/"simulink/batch_model.json",model)
    with gzip.open(path/"series.jsonl.gz","wt") as f:
        for row in rows:f.write(json.dumps(row)+"\n")
    result=summarize(path,policy);write_json(path/"metrics.json",result)
    return result,rows,model


def test_seed_plan_has_eight_equal_condition_runs_and_stable_replacement(tmp_path,policy):
    manifest=dict(policy=policy,groups=[dict(id="clear_s00",weather="clear",slot=0)])
    tasks=round_tasks(tmp_path,manifest)
    assert len(tasks)==8
    # Simulated runaway: all 8 trials must finish before replacement is queued.
    for i,task in enumerate(tasks):
        write_json(tmp_path/task["case"]/"metrics.json",dict(termination="runaway" if i==0 else "timeout"))
    replacement=round_tasks(tmp_path,manifest)
    assert len(replacement)==8 and {t["attempt"] for t in replacement}=={1}
    assert seeds(42,"clear",0,0)==seeds(42,"clear",0,0)
    assert seeds(42,"clear",0,0)!=seeds(42,"clear",0,1)
    assert json.loads((tmp_path/"groups/clear_s00/attempt_00/attempt.json").read_text())["status"]=="excluded_runaway"
    for task in replacement:write_json(tmp_path/task["case"]/"metrics.json",dict(termination="timeout"))
    assert round_tasks(tmp_path,manifest)==[]
    assert json.loads((tmp_path/"groups/clear_s00/result.json").read_text())["attempt"]==1


def test_runaway_requires_persistent_outward_motion(policy):
    monitor=RunawayMonitor(policy["runaway"])
    assert not any(monitor.update(t,[50+t,0,0]) for t in np.arange(0,30,.5))
    monitor=RunawayMonitor(policy["runaway"])
    triggered=[monitor.update(t,[50+20*t,0,0]) for t in np.arange(0,30,.5)]
    assert any(triggered) and monitor.evidence["net_outward_m"]>30
    monitor=RunawayMonitor(policy["runaway"])
    # Aircraft goes far, then returns well before the 10s far-range window.
    assert not any(monitor.update(t,[50+30*t if t<7 else 260-20*(t-7),0,0]) for t in np.arange(0,17,.5))


def test_timeout_is_not_candidate_and_final_phase_is_censored(tmp_path,policy):
    result,_,_=fake_case(tmp_path/"run",policy,candidate=False)
    assert result["termination"]=="timeout" and result["radial_error_m"] is None
    assert result["phases"][0]["completed"]
    assert result["phases"][1]["completed"]
    assert not result["phases"][2]["completed"]
    assert result["phases"][2]["completed_duration_s"] is None
    assert result["phases"][2]["dwell_s"]==pytest.approx(.04)
    assert result["invalid_wait_s"]==pytest.approx(.02)


def test_stop_uses_terminal_gear_point_and_existing_proxy(tmp_path,policy):
    result,rows,model=fake_case(tmp_path/"run",policy,candidate=True)
    assert result["candidate"] and result["qualified"]
    assert result["touchdown_position"]==rows[-1]["s1"]["p"]
    assert result["radial_error_m"]==pytest.approx(.1)
    assert result["phases"][2]["completed"]
    wave=motion(rows,model)[:,0]
    assert wave==pytest.approx(np.cos(4*np.arange(6)*.02+.5))


def test_nominal_target_is_not_a_ned_projection():
    theta=.2;rd=np.array([[np.cos(theta),0,-np.sin(theta)],[0,-1,0],[-np.sin(theta),0,-np.cos(theta)]])
    s=dict(rd=rd,db=np.diag([1,-1,-1]))
    p=nominal_point(s,dict(nominal_cg_target_ned_m=[-45,0,-8.4]),dict(gear_body=[0,0,2.9],H_ship_body=[-45,0,-5.5]))
    assert abs(p[0])>.1 and abs(p[2])>.1


def test_paired_bootstrap_weather_weights_and_missing_candidates(policy):
    rows=[]
    for w in policy["weather"]:
        for slot in range(2):
            for method in ("C1","C3"):
                rows.append(dict(weather=w,slot=slot,sea=5,method=method,replacement=0,candidate=True,
                    radial_error_m=(1 if method=="C1" else 2 if w=="clear" else 4)))
    result=paired(rows,(5,"C3"),(5,"C1"),"radial_error_m","equal_weather",policy)
    assert result["mean"]==2 and result["pairs"]==4
    for r in rows:
        if r["weather"]=="fog":r["candidate"]=False
    conditional=paired(rows,(5,"C3"),(5,"C1"),"radial_error_m","equal_weather",policy)
    assert conditional["mean"] is None and conditional["pairs"]==2
    assert conditional["missing"]=={"both_candidates_required":2}
    unconditional=paired(rows,(5,"C3"),(5,"C1"),"candidate","equal_weather",policy)
    assert unconditional["mean"]==0 and unconditional["pairs"]==4


def test_wilson_and_equal_weather_description():
    lo,hi=wilson(0,10);assert lo==pytest.approx(0) and hi==pytest.approx(.2775328,abs=1e-6)
    data=[dict(weather="clear",x=0)]*9+[dict(weather="fog",x=10)]
    assert balanced_description(data,"x",["clear","fog"])["mean"]==5


def test_session_lease_only_finalizes():
    class Engine:
        def finalize_external_model(self,**kw):self.finalized=True
    class Session:
        engine=Engine()
        def call(self,*a):return a
    session=Session()
    with SessionLease(session) as lease:assert lease.call("read")==("read",)
    assert session.engine.finalized


def test_full_matrix_and_pilot_validation():
    a=parser().parse_args(["plan","--output","outputs/batch/test"])
    manifest=create(a)
    assert len(manifest["groups"])*8==320
    a.duration=2
    with pytest.raises(ValueError,match="pilot"):create(a)


def test_supervisor_processes_resource_log_and_log_compression(tmp_path,policy,monkeypatch):
    import subprocess
    import sys
    real_popen=subprocess.Popen
    def popen(args,**kwargs):
        if "main.batch.worker" in args:
            args=[sys.executable,"-c","import time; print('worker fixture'); time.sleep(.2)"]
        return real_popen(args,**kwargs)
    monkeypatch.setattr(subprocess,"Popen",popen)
    manifest=dict(policy=policy,groups=[dict(id="clear_s00",weather="clear",slot=0)])
    round_tasks(tmp_path,manifest)
    args=parser().parse_args(["run","--output","outputs/batch/test","--workers","2"])
    supervise(tmp_path,manifest,args)
    assert len(list(tmp_path.glob("worker_*.log.gz")))==2
    resource=json.loads((tmp_path/"resources.jsonl").read_text().splitlines()[0])
    assert len(resource["workers"])==2


def test_all_table_rows_keep_failed_denominators(tmp_path,policy):
    rows=[]
    for weather in policy["weather"]:
        for sea in (5,6):
            for method in ("C0","C1","C2","C3"):
                for slot in range(2):
                    r,_,_=fake_case(tmp_path/f"{weather}_{sea}_{method}_{slot}",policy,weather,sea,method,slot,candidate=slot==0)
                    r["path"]=r["case_id"];rows.append(r)
    t15,t16,t17=tables(rows,policy,tmp_path)
    equal=[r for r in t15 if r["weather"]=="equal_weather"]
    assert all(r["total"]==4 and r["candidates"]==2 and r["rate"]==.5 for r in equal)
    final=[r for r in t17 if r["stage"]==2 and r["weather"]=="equal_weather"]
    assert all(r["entered"]==4 and r["completed"]==2 and r["unfinished"]==2 for r in final)
