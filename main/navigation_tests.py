"""Thesis §5.2.3 navigation experiments, independent of aircraft control.

Run via shell/run_navigation_tests.sh on Ubuntu. A smoke run exercises the
complete recording/replay/report path and is explicitly excluded from science.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from main.config import ROOT, load, path
from main.navigation_test_data import (cases, clean, digest, generate_record, import_record,
                                       navigation_config, source_provenance, write_json)
from main.navigation_test_replay import replay_record
from main.navigation_test_scheduler import experiment_lock, record_groups, run_workers
from main.navigation_test_storage import (compact_receipt, finish_retirement, retire_record,
                                          retired_results_valid, storage_summary, verify_result)

METHODS=("N0","N1","N2","N3","A1","A2","A3")


def validate_config(config):
    if config.get("sensor_retention", "all") not in ("all", "compact"):
        raise ValueError("sensor_retention must be all or compact")
    for name in ("retain_replicates", "anomaly_windows_per_record"):
        value = config.get(name, 1 if name == "retain_replicates" else 3)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    value = config.get("anomaly_context_s", 1.)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError("anomaly_context_s must be positive finite seconds")
    for name in ("workers", "render_threads", "torch_threads"):
        value = config.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if abs(float(config["dt"])-.02)>1e-12:
        raise ValueError("Navigation tests require dt=0.02 s")
    for name in ("camera_hz","lidar_hz"):
        rate=float(config[name]);ratio=1/(config["dt"]*rate) if rate>0 else 0
        if not np.isfinite(ratio) or ratio<1 or abs(ratio-round(ratio))>1e-8:
            raise ValueError(f"{name} must be a positive divisor of 50 Hz")
    if not config["weather"] or set(config["weather"])-{"clear","fog","night","night_fog"}:
        raise ValueError("Only the four trained weather classes are supported")
    if not config["motion_scales"] or any(not np.isfinite(v) or v<=0 for v in config["motion_scales"].values()):
        raise ValueError("Motion scales must be positive")
    if config["seeds_per_cell"]<1 or config["stress_seeds_per_cell"]<0:
        raise ValueError("Invalid number of independent seeds")
    if not config["methods"] or set(config["methods"])-set(METHODS) or "N3" not in config["methods"]:
        raise ValueError("Methods must include N3 and be selected from N0..N3,A1..A3")
    if len(set(config["methods"]))!=len(config["methods"]):
        raise ValueError("Duplicate methods")
    if config["suite"] not in ("main","gimbal","stress","all"):
        raise ValueError("Unknown test suite")
    for name in ("false_valid_error_m","max_outage_s"):
        value=config["evaluation"].get(name)
        if value is not None and (not np.isfinite(value) or value<=0):
            raise ValueError(f"Invalid evaluation {name}")
    if config["evaluation"]["bootstrap_samples"]<1:
        raise ValueError("bootstrap_samples must be positive")
    if config["renderer"] not in ("blender","cpu"):
        raise ValueError("renderer must be blender or cpu")


def freeze(config, base, landing, geometry, checkpoint, source=None, trajectory_source=None):
    files={}
    for folder in ("navigation","main","simulation/bridge","visual_modeling","visual_training/models"):
        for filename in sorted((ROOT/folder).rglob("*.py")):
            files[str(filename.relative_to(ROOT))]=digest(filename)
    for filename in sorted((ROOT/"configs").glob("*.yaml")):
        files[str(filename.relative_to(ROOT))]=digest(filename)
    extra={}
    for name, directory in (("source",source),("trajectory",trajectory_source)):
        if directory:
            directory=Path(directory).resolve()
            extra[name]={"path":str(directory),"run_sha256":digest(directory/"run.json"),
                         "steps_sha256":digest(directory/"steps.jsonl")}
    snapshot=clean(dict(config=config,base_config=base,landing_config=landing,geometry=geometry,
                       checkpoint_sha256=digest(checkpoint),source_files=files,inputs=extra))
    snapshot["fingerprint"]=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
    return snapshot


def verify_record(directory):
    """Do not resume with damaged/replaced sensor files under a frozen manifest."""
    directory=Path(directory)
    meta=json.loads((directory/"run.json").read_text())
    if meta["status"]!="complete":
        return False
    for name,expected in json.loads((directory/"record_hashes.json").read_text()).items():
        if digest(directory/name)!=expected:
            raise ValueError(f"Recorded metadata checksum mismatch: {directory/name}")
    for name,expected in json.loads((directory/"sensor_hashes.json").read_text()).items():
        filename=Path(name)
        if not filename.is_absolute():filename=directory/"sensors"/filename
        if not filename.is_file() or digest(filename)!=expected:
            raise ValueError(f"Recorded sensor checksum mismatch: {filename}")
    return True


def runtime(config):
    import torch
    torch.set_num_threads(config["torch_threads"])
    if config["device"]=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run dependency check first")
    # No autocast/TF32 change relative to inference code; full float32 model.
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                torch=torch.__version__,cuda=torch.version.cuda,device=config["device"],
                gpu=torch.cuda.get_device_name(0) if config["device"]=="cuda" else None,
                torch_threads=torch.get_num_threads(),inference_precision="float32",
                gpu_timing="torch.cuda.synchronize before/after measured navigation blocks",
                renderer=config["renderer"],render_threads=config["render_threads"],
                execution=config.get("execution", {}),
                matlab="not used; navigation does not feed aircraft control")


def initialize_worker(payload):
    import cv2
    from navigation.vision.infer_h import HDetector
    cv2.setNumThreads(1)
    resources=runtime(payload["config"])
    started=time.perf_counter()
    detector=HDetector(payload["checkpoint"],payload["config"]["device"])
    worker_config={**deepcopy(payload["config"]),"attempt_id":payload["attempt_id"],"runtime":resources}
    return dict(payload,detector=detector,config=worker_config), dict(runtime=resources,
        model_load_s=time.perf_counter()-started,network_input_size=detector.image_size)


def run_record_group(job, state):
    """One worker owns a sensor archive and every case that reuses that archive."""
    output=Path(state["output"])
    record=output/"records"/job["record_id"]
    record_error=None
    reused=False
    check_path=output/"record_checks"/(job["record_id"]+".json")
    check=dict(status="checking",record_id=job["record_id"],attempt_id=state.get("attempt_id"))
    write_json(check_path,check)
    try:
        receipt=compact_receipt(record)
        rebuild=False
        if receipt is not None:
            if not state["resume"]:
                raise FileExistsError(f"Retired record already exists: {record}")
            if retired_results_valid(record,output,job,receipt):
                receipt=finish_retirement(record,receipt)
                write_json(check_path,{**check,"status":"passed","input_record_sha256":receipt["input_record_sha256"]})
                print(json.dumps(dict(event="sensor_storage",record_id=job["record_id"],
                    state=receipt["state"],released_sensor_bytes=receipt["released_sensor_bytes"])),flush=True)
                for case in job["cases"]:
                    print(json.dumps(dict(event="case_finished",case=case["id"],record_id=job["record_id"],
                        failures=[],skipped_methods=case["methods"],status="complete")),flush=True)
                return
            # Rebuild all dependent results when retired evidence needs repair.
            rebuild=True
            print(json.dumps(dict(event="retired_record_rebuild",record_id=job["record_id"],
                reason="result or retained observation failed verification; replaying all dependent methods")),flush=True)
        existing=record/"run.json"
        reused=not rebuild and existing.exists() and verify_record(record)
        if not reused:
            if record.exists():
                if not state["resume"]:raise FileExistsError(record)
                shutil.rmtree(record)
            if state["source"]:
                import_record(state["source"],record,state["base"],state["duration"])
            else:
                generate_record(record,job["cases"][0],state["config"],state["base"],state["landing"],
                    state["cal"],state["geometry"],state["nav"],state["duration"],state["source_trajectory"])
        record_signature=digest(record/"record_hashes.json")
        write_json(check_path,{**check,"status":"passed","input_record_sha256":record_signature})
    except Exception as exc:
        record_error=f"{type(exc).__name__}: {exc}"
        write_json(check_path,{**check,"status":"failed","error":record_error})
        print(json.dumps(dict(event="record_failed",record_id=job["record_id"],error=record_error)),flush=True)
    group_failed=record_error is not None
    for case in job["cases"]:
        failures=[];skipped=[]
        if record_error:
            failures.append(dict(case=case["id"],record_id=job["record_id"],phase="record",error=record_error))
        else:
            for method in case["methods"]:
                destination=output/"results"/case["id"]/method
                if state["resume"] and reused and verify_result(destination,record_signature):
                    skipped.append(method)
                    continue
                try:
                    replay_record(record,destination,dict(case,method=method),state["config"],
                                  state["checkpoint"],detector=state["detector"])
                except Exception as exc:
                    failure=dict(case=case["id"],method=method,phase="replay",error=f"{type(exc).__name__}: {exc}")
                    failures.append(failure)
                    group_failed=True
                    print(json.dumps(dict(event="replay_failed",**failure)),flush=True)
        print(json.dumps(dict(event="case_finished",case=case["id"],record_id=job["record_id"],
            failures=failures,skipped_methods=skipped,status="failed" if failures else "complete")),flush=True)
    if not group_failed:
        try:
            receipt=retire_record(record,output,job,state["config"],external=bool(state["source"]))
        except Exception as exc:
            # A post-replay verification failure must not leave a passed marker
            # that could admit damaged numerical evidence to later analysis.
            write_json(check_path,{**check,"status":"failed","error":f"Storage verification/cleanup: {exc}"})
            raise
        if receipt:
            print(json.dumps(dict(event="sensor_storage",record_id=job["record_id"],
                state=receipt["state"],released_sensor_bytes=receipt.get("released_sensor_bytes",0),
                retained_sensor_files=receipt.get("retained_sensor_files"))),flush=True)


def run(args):
    config=yaml.safe_load(path(args.config).read_text())
    config.setdefault("workers",2)
    for key in ("device","renderer","render_device","render_python","seeds_per_cell","suite",
                "workers","render_threads","torch_threads","sensor_retention","retain_replicates"):
        value=getattr(args,key,None)
        if value is not None:config[key]=value
    smoke=args.command=="smoke"
    if args.duration is not None and not smoke:
        raise ValueError("--duration is smoke-only; formal cases must contain complete flights")
    duration=2. if smoke and args.duration is None else args.duration
    if duration is not None and (duration<=0 or abs(duration/.02-round(duration/.02))>1e-7):
        raise ValueError("Smoke duration must be positive and divisible by 0.02")
    if smoke:
        config.update(weather=["clear"],motion_scales={"medium":1.},seeds_per_cell=1,
                      stress_seeds_per_cell=1,suite=args.suite or "main")
    validate_config(config)
    base,landing,cal,geometry=load(config["base_config"])
    checkpoint=path(base["checkpoint"])
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if args.source and args.trajectory:
        raise ValueError("Choose --source fixed sensors OR --trajectory regeneration")
    source=path(args.source) if args.source else None
    source_trajectory=path(args.trajectory) if args.trajectory else None
    provenance=source_provenance(source or source_trajectory) if source or source_trajectory else {}
    if source and args.suite not in (None,"main"):
        raise ValueError("Fixed sensor import supports only --suite main; gimbal studies require regeneration")
    if source:config["suite"]="main"
    excluded=smoke or (config["renderer"]=="cpu" and not source) or provenance.get("excluded_from_formal",False)
    plan=cases(config,excluded)
    for c in plan:
        c["motion_scale"]=config["motion_scales"][c["motion"]]
        c["source_record"]=c["record_id"]
    if source:
        original=json.loads((source/"run.json").read_text())
        weather=provenance["source_weather"]
        plan=[dict(id="recorded",record_id="recorded",pair_id="recorded",suite="main",condition="baseline",
                   weather=weather,motion="recorded",seed=provenance["source_seed"],
                   source_record=str(source),gimbal_mode="recorded",methods=config["methods"],excluded_from_formal=excluded)]
        if original.get("checkpoint_sha256") and original["checkpoint_sha256"]!=digest(checkpoint):
            raise ValueError("Source used different weights; this study fixes the current four-weather checkpoint")
    jobs=record_groups(plan)
    effective=min(config["workers"],len(jobs))
    config["execution"]=dict(workers_requested=config["workers"],workers_effective=effective,
        process_start_method="spawn",scheduling="one job per shared sensor record",
        timing_mode="shared_resources" if effective>1 else "single_worker",
        timing_note="Concurrent timings include CPU/GPU contention; use --workers 1 for isolated latency",
        opencv_threads_per_worker=1,blas_threads_per_worker=1)
    output=path(args.output) if args.output else ROOT/"outputs/navigation_tests"/(
        ("smoke_" if smoke else "run_")+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    snapshot=freeze(config,base,landing,geometry,checkpoint,source,source_trajectory)
    snapshot["smoke_duration_s"]=duration
    snapshot["fingerprint"]=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not args.resume:raise FileExistsError(f"Use a new output directory or --resume: {output}")
    else:
        output.mkdir(parents=True)
    with experiment_lock(output):
        if args.resume:
            previous=json.loads((output/"frozen.json").read_text())
            if previous["fingerprint"]!=snapshot["fingerprint"]:
                raise ValueError("Frozen configuration/code/weights/input changed; use a new experiment directory")
        return execute_experiment(output,snapshot,config,plan,jobs,base,landing,cal,geometry,checkpoint,
                                  source,source_trajectory,duration,provenance,excluded,args.resume)


def execute_experiment(output,snapshot,config,plan,jobs,base,landing,cal,geometry,checkpoint,
                       source,source_trajectory,duration,provenance,excluded,resume):
    from main.navigation_test_analysis import analyze
    write_json(output/"frozen.json",snapshot)
    attempts=[]
    if resume and (output/"manifest.json").exists():
        previous=json.loads((output/"manifest.json").read_text())
        attempts=previous.get("attempts",[dict(attempt_id="legacy",runtime=previous.get("runtime"),
            workers=previous.get("workers",[]),status=previous.get("status"),elapsed_s=previous.get("elapsed_s"))])
    attempt_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt=dict(attempt_id=attempt_id,status="running",started_utc=datetime.now(timezone.utc).isoformat())
    attempts.append(attempt)
    metadata=dict(schema_version=1,status="running",kind="navigation_only",cases=plan,
                  evaluation=config["evaluation"],config=config,excluded_from_formal=excluded,
                  fingerprint=snapshot["fingerprint"],runtime=runtime(config),
                  source_mode="recorded_sensors" if source else "recorded_trajectory" if source_trajectory else "kinematic",
                  source_provenance=provenance,
                  stage_definition="generator mission stages; legacy logs: first horizontal radius<=3m and height<=6m enters hover, then height<4.5m enters descent",
                  failures=[],workers=[],case_status={},completed_cases=0,attempt_id=attempt_id,attempts=attempts)
    attempt.update(runtime=metadata["runtime"],workers=metadata["workers"])
    write_json(output/"manifest.json",metadata)
    recorded_nav=None
    if source_trajectory:
        recorded_nav=json.loads((source_trajectory/"run.json").read_text())["navigation"]
    nav=navigation_config(base,landing,geometry,checkpoint,config["device"],recorded_nav)
    payload=dict(config=config,base=base,landing=landing,cal=cal,geometry=geometry,checkpoint=checkpoint,
        output=output,source=source,source_trajectory=source_trajectory,duration=duration,nav=nav,resume=resume,
        attempt_id=attempt_id)
    start=time.perf_counter()
    print(json.dumps(dict(event="navigation_workers",records=len(jobs),cases=len(plan),
        **config["execution"],render_threads_per_worker=config["render_threads"],
        torch_threads_per_worker=config["torch_threads"])),flush=True)
    try:
        for event in run_workers(jobs,config["workers"],payload,initialize_worker,run_record_group,
                                  output/"workers",config["torch_threads"]):
            if event["event"]=="worker_ready":
                metadata["workers"].append(event)
                metadata["model_load_s"]=sum(w["model_load_s"] for w in metadata["workers"])
                metadata["model_load_note"]="sum across workers, may overlap in wall time"
                metadata["network_input_size"]=event["network_input_size"]
                write_json(output/"manifest.json",metadata)
            elif event["event"]=="case_finished":
                if event["case"] in metadata["case_status"]:
                    raise RuntimeError(f"Duplicate completion: {event['case']}")
                metadata["case_status"][event["case"]]=event["status"]
                metadata["completed_cases"]+=1
                metadata["failures"].extend(event["failures"])
                write_json(output/"manifest.json",metadata)
                event={**event,"event":"case_complete","completed":metadata["completed_cases"],
                    "total":len(plan),"failures":len(metadata["failures"]),"output":str(output),
                    "wall_s":time.perf_counter()-start}
            print(json.dumps({**event,"coordinator_wall_s":time.perf_counter()-start}),flush=True)
        if metadata["completed_cases"]!=len(plan):
            raise RuntimeError("Missing case completion events")
        metadata["status"]="complete" if not metadata["failures"] else "completed_with_failures"
    except BaseException as exc:
        metadata.update(status="interrupted",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_s"]=time.perf_counter()-start
        attempt.update(status=metadata["status"],elapsed_s=metadata["elapsed_s"],
            ended_utc=datetime.now(timezone.utc).isoformat(),failures=metadata["failures"],
            completed_cases=metadata["completed_cases"],error=metadata.get("error"))
        write_json(output/"manifest.json",metadata)
        write_json(output/"storage_summary.json",storage_summary(output))
    analyze(output,bootstrap_samples=config["evaluation"]["bootstrap_samples"])
    print(json.dumps(dict(event="navigation_tests_complete",output=str(output),status=metadata["status"])),flush=True)
    return 1 if metadata["failures"] else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["smoke","run","analyze"])
    p.add_argument("--config",default="configs/navigation_tests.yaml")
    p.add_argument("--output",help="Experiment directory; must be new unless --resume")
    p.add_argument("--source",help="Existing complete run with saved sensors, replay once without MATLAB/rendering")
    p.add_argument("--trajectory",help="Existing run supplying BODY/DECK truth for new independent sensor records")
    p.add_argument("--duration",type=float,help="Smoke only (default 2 seconds)")
    p.add_argument("--device",choices=["cpu","cuda","mps"])
    p.add_argument("--renderer",choices=["cpu","blender"])
    p.add_argument("--render-device",choices=["CPU","CUDA","OPTIX","METAL"])
    p.add_argument("--render-python")
    p.add_argument("--suite",choices=["main","gimbal","stress","all"])
    p.add_argument("--seeds-per-cell",type=int)
    p.add_argument("--workers",type=int,help="concurrent sensor-record jobs (default config: 2)")
    p.add_argument("--render-threads",type=int,help="Blender CPU threads per worker")
    p.add_argument("--torch-threads",type=int,help="PyTorch CPU threads per worker")
    p.add_argument("--sensor-retention",choices=["compact","all"],help="compact retires successful groups after replay")
    p.add_argument("--retain-replicates",type=int,help="compact: full records for first N seeds per weather/motion/gimbal cell")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.command=="analyze":
        if not a.output:p.error("analyze requires --output pointing to an experiment")
        if any(getattr(a,key) is not None for key in ("workers","render_threads","torch_threads","sensor_retention","retain_replicates")):
            p.error("workers/thread/storage options apply only to smoke/run")
        from main.navigation_test_analysis import analyze
        with experiment_lock(path(a.output)):
            analyze(path(a.output))
        return 0
    return run(a)


if __name__=="__main__":
    raise SystemExit(main())
