"""Sensor lifecycle, crash recovery and preservation of all numeric evidence."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from main import navigation_tests as bench
from main import navigation_test_storage as storage
from main import navigation_test_analysis as analysis
from main.navigation_test_io import digest, write_json


def fixture(root, replicate=1):
    config = dict(sensor_retention="compact", retain_replicates=1, anomaly_context_s=.1,
                  anomaly_windows_per_record=3,
                  evaluation=dict(false_valid_error_m=1., max_outage_s=.5, bootstrap_samples=2))
    common = dict(record_id="shared", pair_id="pair", weather="clear", motion="low", seed=1,
                  gimbal_mode="tracking", replicate=replicate, source_record="shared")
    cases = [dict(common, id="base", methods=["N0", "N3"], suite="main", condition="baseline"),
             dict(common, id="stress", methods=["N3"], suite="stress", condition="camera_gap")]
    job = dict(record_id="shared", cases=cases)
    record = root / "records/shared"
    sensors = record / "sensors"
    sensors.mkdir(parents=True)
    rows = []
    for i in range(100):
        camera, lidar = i % 5 == 0, i % 10 == 0
        np.savez_compressed(sensors / f"{i:06d}.npz", timestamp=i*.02,
                            **({"image": np.full((8, 8, 3), i, dtype=np.uint8)} if camera else {}))
        rows.append(dict(timestamp=i*.02, dt=.02, iteration=i, stage="approach",
                         sensors=dict(camera=camera, lidar=lidar), sensor_file=f"sensors/{i:06d}.npz"))
    (sensors / "current_rgb.png").write_bytes(b"transport")
    (record / "steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    write_json(record / "run.json", dict(status="complete", planned_duration_s=2., completed_duration_s=2.))
    write_json(record / "sensor_hashes.json", {p.name: digest(p) for p in sensors.glob("*.npz")})
    write_json(record / "record_hashes.json", {n: digest(record/n) for n in
                                               ("run.json", "steps.jsonl", "sensor_hashes.json")})
    sig = digest(record / "record_hashes.json")
    for key, folder in storage.result_paths(root, job).items():
        write_json(folder / "run.json", dict(status="complete", planned_duration_s=2., completed_duration_s=2.,
                                             input_record_sha256=sig))
        numerical = []
        for r in rows:
            numerical.append(dict(timestamp=r["timestamp"], dt=.02, iteration=r["iteration"], stage="approach",
                camera_sample=r["sensors"]["camera"], lidar_sample=r["sensors"]["lidar"],
                estimate_available=True, navigation_valid=True, cg_signal_valid=True,
                gear_truth_ned=[0., 0., 0.], gear_estimate_ned=[.1, 0., 0.],
                cg_truth_ned=[0., 0., 0.], cg_estimate_ned=[.1, 0., 0.], cg_shaped_ned=[.1, 0., 0.],
                gear_covariance_ned=np.eye(3).tolist(), cg_covariance_ned=np.eye(3).tolist(),
                diagnostics={}, quality={}, timing={}))
        (folder / "steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in numerical))
        seal_result(folder)
    write_json(root / "record_checks/shared.json", dict(status="passed", input_record_sha256=sig))
    write_json(root / "manifest.json", dict(cases=cases, config=config, evaluation=config["evaluation"]))
    return record, job, config


def seal_result(folder):
    write_json(folder / "result_hashes.json", {n: digest(folder/n) for n in ("run.json", "steps.jsonl")})


def state(root, config):
    return dict(output=root, config=config, resume=True, source=None, source_trajectory=None,
                base={}, landing={}, cal=None, geometry={}, nav={}, duration=2., checkpoint="weights", detector=None)


def test_complete_group_prunes_only_sensors_and_resume_skips_all_methods(tmp_path, monkeypatch, capsys):
    record, job, config = fixture(tmp_path)
    before = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file() and "sensors" not in p.parts}
    receipt = storage.retire_record(record, tmp_path, job, config)
    assert receipt["state"] == "pruned" and receipt["released_sensor_bytes"] > 0
    assert list((record/"sensors").iterdir()) == []
    assert all(digest(p) == h for p, h in before.items())
    monkeypatch.setattr(bench, "generate_record", lambda *a: pytest.fail("unnecessary generation"))
    monkeypatch.setattr(bench, "replay_record", lambda *a, **kw: pytest.fail("unnecessary replay"))
    bench.run_record_group(job, state(tmp_path, config))
    finished = [json.loads(s) for s in capsys.readouterr().out.splitlines() if '"case_finished"' in s]
    assert [e["skipped_methods"] for e in finished] == [["N0", "N3"], ["N3"]]
    assert all(e["status"] == "complete" for e in finished)
    assert storage.storage_summary(tmp_path)["pruned_records"] == 1


@pytest.mark.parametrize("replicate,mode,prune", [(0,"compact",False), (1,"compact",True), (1,"all",False)])
def test_representatives_and_all_mode_preserve_observations(tmp_path, replicate, mode, prune):
    record, job, config = fixture(tmp_path, replicate)
    config["sensor_retention"] = mode
    receipt = storage.retire_record(record, tmp_path, job, config)
    assert len(list((record/"sensors").glob("*.npz"))) == (0 if prune else 100)
    if mode == "compact" and not prune:
        assert receipt["reason"] == "representative_replicate"
        assert bench.verify_record(record)


def test_full_plan_retains_24_prespecified_representative_records():
    c = yaml.safe_load((bench.ROOT/"configs/navigation_tests.yaml").read_text())
    jobs = bench.record_groups(bench.cases(c))
    selected = [j for j in jobs if j["cases"][0]["replicate"] < c["retain_replicates"]]
    assert len(selected) == 24
    assert len({(j["cases"][0]["weather"], j["cases"][0]["motion"], j["cases"][0]["gimbal_mode"]) for j in selected}) == 24


def test_exception_in_any_shared_method_prevents_cleanup(tmp_path, monkeypatch):
    record, job, config = fixture(tmp_path)
    (tmp_path/"results/stress/N3/steps.jsonl").unlink()
    monkeypatch.setattr(bench, "replay_record", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("failure")))
    bench.run_record_group(job, state(tmp_path, config))
    assert not (record/"storage.json").exists()
    assert len(list((record/"sensors").glob("*.npz"))) == 100


def test_cleanup_waits_for_every_shared_replay_and_verifies_results(tmp_path, monkeypatch):
    record, job, config = fixture(tmp_path)
    for folder in storage.result_paths(tmp_path, job).values():
        (folder/"result_hashes.json").unlink()
    seen = []
    def replay(rec, dest, case, *args, **kwargs):
        assert len(list((rec/"sensors").glob("*.npz"))) == 100
        seen.append((case["id"],case["method"]))
        seal_result(dest)
    monkeypatch.setattr(bench,"replay_record",replay)
    bench.run_record_group(job,state(tmp_path,config))
    assert seen == [("base","N0"),("base","N3"),("stress","N3")]
    assert list((record/"sensors").iterdir()) == []


def test_crash_mid_cleanup_is_idempotently_recovered(tmp_path, monkeypatch):
    record, job, config = fixture(tmp_path)
    original = Path.unlink
    count = 0
    def interrupt(p, *a, **kw):
        nonlocal count
        if p.suffix == ".npz":
            count += 1
            if count == 4:
                raise KeyboardInterrupt()
        return original(p,*a,**kw)
    with monkeypatch.context() as m:
        m.setattr(Path,"unlink",interrupt)
        with pytest.raises(KeyboardInterrupt):storage.retire_record(record,tmp_path,job,config)
    assert storage.compact_receipt(record)["state"] == "pruning"
    assert len(list((record/"sensors").glob("*.npz"))) == 97
    monkeypatch.setattr(bench,"generate_record",lambda *a:pytest.fail("regenerated"))
    bench.run_record_group(job,state(tmp_path,config))
    assert storage.compact_receipt(record)["state"] == "pruned"
    assert not list((record/"sensors").iterdir())


@pytest.mark.parametrize("damage",["missing_result","changed_result","replaced_manifest","missing_clip"])
def test_retired_group_damage_requires_rebuilding_every_method(tmp_path, monkeypatch, damage):
    record, job, config = fixture(tmp_path)
    # Force one retained diagnostic observation for the missing-clip scenario.
    if damage == "missing_clip":
        monkeypatch.setattr(storage,"anomaly_windows",lambda *a:[dict(window_start_s=0.,window_end_s=.1)])
    receipt = storage.retire_record(record,tmp_path,job,config)
    result = tmp_path/"results/base/N0"
    if damage == "missing_result":(result/"steps.jsonl").unlink()
    elif damage == "changed_result":(result/"steps.jsonl").write_text("changed")
    elif damage == "replaced_manifest":
        (result/"steps.jsonl").write_text("changed");seal_result(result)
    else:(record/"sensors"/next(iter(receipt["retained_sensors"]))).unlink()
    assert not storage.retired_results_valid(record,tmp_path,job,receipt)
    calls=[]
    def generate(*a):
        calls.append("generate")
        # A new recording identity; all old result files must be overwritten.
        fixture(tmp_path)
        (record/"steps.jsonl").write_text((record/"steps.jsonl").read_text()+"\n")
        write_json(record/"record_hashes.json",{n:digest(record/n) for n in ("run.json","steps.jsonl","sensor_hashes.json")})
    def replay(rec,dest,case,*a,**kw):
        calls.append((case["id"],case["method"]))
        info=json.loads((dest/"run.json").read_text());info["input_record_sha256"]=digest(rec/"record_hashes.json")
        write_json(dest/"run.json",info);seal_result(dest)
    monkeypatch.setattr(bench,"generate_record",generate)
    monkeypatch.setattr(bench,"replay_record",replay)
    bench.run_record_group(job,state(tmp_path,config))
    assert calls == ["generate",("base","N0"),("base","N3"),("stress","N3")]
    assert storage.compact_receipt(record)["input_record_sha256"] != receipt["input_record_sha256"]


def test_anomaly_clips_preserve_images_without_filtering_any_result_rows(tmp_path):
    record, job, config = fixture(tmp_path)
    folder=tmp_path/"results/base/N3"
    rows=[json.loads(s) for s in (folder/"steps.jsonl").read_text().splitlines()]
    for row in rows:
        if .4 <= row["timestamp"] < 1.:
            row["navigation_valid"]=False
        if row["timestamp"]==1.5:
            row["gear_estimate_ned"]=[5.,0.,0.]
    (folder/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows));seal_result(folder)
    before=digest(folder/"steps.jsonl")
    receipt=storage.retire_record(record,tmp_path,job,config)
    assert {w["kind"] for w in receipt["anomaly_windows"]} == {
        "largest_false_valid_error","longest_outage_start","longest_outage_end"}
    kept={p.name for p in (record/"sensors").glob("*.npz")}
    assert {"000020.npz","000050.npz","000075.npz"} <= kept
    assert "000000.npz" not in kept and len(kept)<15
    assert digest(folder/"steps.jsonl")==before
    assert storage.retired_results_valid(record,tmp_path,job,receipt)


def test_external_source_is_never_deleted(tmp_path):
    record,job,config=fixture(tmp_path)
    external=tmp_path/"outside.npz";external.write_bytes(b"external")
    write_json(record/"sensor_hashes.json",{str(external):digest(external)})
    write_json(record/"record_hashes.json",{n:digest(record/n) for n in ("run.json","steps.jsonl","sensor_hashes.json")})
    for p in storage.result_paths(tmp_path,job).values():
        info=json.loads((p/"run.json").read_text());info["input_record_sha256"]=digest(record/"record_hashes.json")
        write_json(p/"run.json",info);seal_result(p)
    receipt=storage.retire_record(record,tmp_path,job,config,external=True)
    assert receipt["state"]=="external" and external.read_bytes()==b"external"
    with pytest.raises(ValueError,match="locally owned"):
        storage.retire_record(record,tmp_path,job,config,external=False)


def test_sensor_symlink_and_tampering_never_authorize_cleanup(tmp_path):
    record,job,config=fixture(tmp_path)
    p=record/"sensors/000000.npz";p.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Sensor changed"):
        storage.retire_record(record,tmp_path,job,config)
    assert len(list((record/"sensors").glob("*.npz")))==100
    p.unlink();p.symlink_to(tmp_path/"outside.npz")
    with pytest.raises(ValueError,match="symlink"):
        storage.retire_record(record,tmp_path,job,config)


@pytest.mark.parametrize("damage",["metadata", "receipt"])
def test_bad_retirement_identity_fails_closed_and_keeps_numeric_results(tmp_path,damage):
    record,job,config=fixture(tmp_path)
    storage.retire_record(record,tmp_path,job,config)
    paths=storage.result_paths(tmp_path,job)
    before={k:digest(p/"steps.jsonl") for k,p in paths.items()}
    if damage=="metadata":(record/"steps.jsonl").write_text("corrupted")
    else:write_json(record/"storage.json",dict(state="unknown"))
    runs,_=analysis._discover(tmp_path,json.loads((tmp_path/"manifest.json").read_text()))
    assert all(r.cohort=="incomplete" for r in runs)
    bench.run_record_group(job,state(tmp_path,config))
    assert json.loads((tmp_path/"record_checks/shared.json").read_text())["status"]=="failed"
    assert all(digest(p/"steps.jsonl")==before[k] for k,p in paths.items())


def test_post_replay_verification_failure_is_not_reported_as_valid_evidence(tmp_path,monkeypatch):
    record,job,config=fixture(tmp_path)
    def fail(*a,**kw):raise ValueError("bad result checksum")
    monkeypatch.setattr(bench,"retire_record",fail)
    with pytest.raises(ValueError,match="checksum"):
        bench.run_record_group(job,state(tmp_path,config))
    runs,_=analysis._discover(tmp_path,json.loads((tmp_path/"manifest.json").read_text()))
    assert all(r.cohort=="incomplete" for r in runs)
    assert len(list((record/"sensors").glob("*.npz")))==100


def test_analysis_is_identical_after_pruning_and_without_sensor_payloads(tmp_path):
    record,job,config=fixture(tmp_path)
    before=analysis.analyze(tmp_path,bootstrap_samples=2)
    hashes={n:digest(tmp_path/n) for n in ("table_5_13.csv","table_5_14.csv","figure_5_8.csv",
        "figure_5_9.csv","figure_5_10.csv","figure_5_10_innovations.csv","flight_metrics.csv","samples.csv","raw_steps.jsonl")}
    storage.retire_record(record,tmp_path,job,config)
    shutil.rmtree(record/"sensors")
    after=analysis.analyze(tmp_path,bootstrap_samples=2)
    assert before["flight_metrics"]==after["flight_metrics"]
    assert not after["failures"] and after["storage"]["pruned_records"]==1
    assert all(digest(tmp_path/n)==h for n,h in hashes.items())
    # A changed numeric result is invalidated individually, not silently used.
    with (tmp_path/"results/base/N0/steps.jsonl").open("a") as f:f.write("{}\n")
    runs,_=analysis._discover(tmp_path,json.loads((tmp_path/"manifest.json").read_text()))
    assert [(r.case["id"],r.method) for r in runs if r.cohort=="incomplete"]==[("base","N0")]


@pytest.mark.parametrize("name,value",[("sensor_retention","bad"),("retain_replicates",-1),
    ("retain_replicates",True),("anomaly_windows_per_record",1.5),("anomaly_context_s",float("nan")),("anomaly_context_s",0)])
def test_invalid_storage_configuration_rejected(name,value):
    c=yaml.safe_load((bench.ROOT/"configs/navigation_tests.yaml").read_text());c[name]=value
    with pytest.raises(ValueError,match=name):bench.validate_config(c)
