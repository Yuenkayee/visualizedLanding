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

METHODS=("N0","N1","N2","N3","A1","A2","A3")


def validate_config(config):
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


def verify_result(directory):
    directory=Path(directory)
    try:
        meta=json.loads((directory/"run.json").read_text())
        hashes=json.loads((directory/"result_hashes.json").read_text())
        if meta.get("status")!="complete" or meta.get("completed_duration_s",0)<meta["planned_duration_s"]-1e-8:
            return False
        return all((directory/name).is_file() and digest(directory/name)==expected for name,expected in hashes.items()) and {"run.json","steps.jsonl"}<=hashes.keys()
    except (OSError,KeyError,ValueError):
        return False


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
                matlab="not used; navigation does not feed aircraft control")


def run(args):
    from main.navigation_test_analysis import analyze
    from navigation.vision.infer_h import HDetector
    config=yaml.safe_load(path(args.config).read_text())
    for key in ("device","renderer","render_device","render_python","seeds_per_cell","suite"):
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
    output=path(args.output) if args.output else ROOT/"outputs/navigation_tests"/(
        ("smoke_" if smoke else "run_")+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    snapshot=freeze(config,base,landing,geometry,checkpoint,source,source_trajectory)
    snapshot["smoke_duration_s"]=duration
    snapshot["fingerprint"]=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not args.resume:raise FileExistsError(f"Use a new output directory or --resume: {output}")
        previous=json.loads((output/"frozen.json").read_text())
        if previous["fingerprint"]!=snapshot["fingerprint"]:
            raise ValueError("Frozen configuration/code/weights/input changed; use a new experiment directory")
    else:
        output.mkdir(parents=True)
    write_json(output/"frozen.json",snapshot)
    metadata=dict(schema_version=1,status="running",kind="navigation_only",cases=plan,
                  evaluation=config["evaluation"],config=config,excluded_from_formal=excluded,
                  fingerprint=snapshot["fingerprint"],runtime=runtime(config),
                  source_mode="recorded_sensors" if source else "recorded_trajectory" if source_trajectory else "kinematic",
                  source_provenance=provenance,
                  stage_definition="generator mission stages; legacy logs: first horizontal radius<=3m and height<=6m enters hover, then height<4.5m enters descent",
                  failures=[])
    write_json(output/"manifest.json",metadata)
    load_start=time.perf_counter()
    detector=HDetector(checkpoint,config["device"])
    metadata["model_load_s"]=time.perf_counter()-load_start
    metadata["network_input_size"]=detector.image_size
    write_json(output/"manifest.json",metadata)
    recorded_nav=None
    if source_trajectory:
        recorded_nav=json.loads((source_trajectory/"run.json").read_text())["navigation"]
    nav=navigation_config(base,landing,geometry,checkpoint,config["device"],recorded_nav)
    checked=set();failed_records=set();start=time.perf_counter()
    try:
        for index,case in enumerate(plan):
            record=output/"records"/case["record_id"]
            if case["record_id"] not in checked:
                try:
                    existing=record/"run.json"
                    complete=existing.exists() and verify_record(record)
                    if not complete:
                        if record.exists():
                            # Only an explicitly resumed, incomplete generated record.
                            if not args.resume:raise FileExistsError(record)
                            shutil.rmtree(record)
                        if source:
                            import_record(source,record,base,duration)
                        else:
                            generate_record(record,case,config,base,landing,cal,geometry,nav,duration,source_trajectory)
                    checked.add(case["record_id"])
                except Exception as exc:
                    failed_records.add(case["record_id"]);checked.add(case["record_id"])
                    metadata["failures"].append(dict(case=case["id"],phase="record",error=f"{type(exc).__name__}: {exc}"))
                    print(json.dumps(dict(event="record_failed",**metadata["failures"][-1])),flush=True)
            if case["record_id"] not in failed_records:
                for method in case["methods"]:
                    destination=output/"results"/case["id"]/method
                    existing=destination/"run.json"
                    if args.resume and verify_result(destination):
                        continue
                    try:
                        replay_record(record,destination,dict(case,method=method),config,checkpoint,detector=detector)
                    except Exception as exc:
                        metadata["failures"].append(dict(case=case["id"],method=method,phase="replay",error=f"{type(exc).__name__}: {exc}"))
                        print(json.dumps(dict(event="replay_failed",**metadata["failures"][-1])),flush=True)
            write_json(output/"manifest.json",metadata)
            print(json.dumps(dict(event="case_complete",case=case["id"],completed=index+1,total=len(plan),
                wall_s=time.perf_counter()-start,failures=len(metadata["failures"]),output=str(output))),flush=True)
        metadata["status"]="complete" if not metadata["failures"] else "completed_with_failures"
    except BaseException as exc:
        metadata.update(status="interrupted",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_s"]=time.perf_counter()-start
        write_json(output/"manifest.json",metadata)
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
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.command=="analyze":
        if not a.output:p.error("analyze requires --output pointing to an experiment")
        from main.navigation_test_analysis import analyze
        analyze(path(a.output))
        return 0
    return run(a)


if __name__=="__main__":
    raise SystemExit(main())
