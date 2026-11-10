#!/usr/bin/env python3
"""Forty C3 sorties: four weathers, frozen original seeds, wind 10 m/s and Hs 2 m."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from main.batch.common import WEATHERS, fingerprints, output_path, read_json, seeds, sha, write_json
from main.batch.__main__ import supervise

CONDITION = "custom_w10_h2"
REFERENCE_COMMIT = "ff91248e5275ffd6649b6826e809971d84e7da84"
SEED_KEYS = ("wind_seed", "wave_seed", "sensor_seed")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def script_hashes():
    return {name: sha(ROOT / name) for name in ("main/batch/c3.py", "shell/run_landing_c3.sh")}


def make_plan(a):
    # Committing this launcher advances HEAD without changing the algorithms.
    # Check actual runtime files against the baseline below; record the real HEAD.
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    baseline = output_path(a.baseline)
    out = output_path(a.output)
    require(out != baseline and out not in baseline.parents and baseline not in out.parents,
            "Use a separate output directory outside the baseline.")
    old = read_json(baseline / "manifest.json")
    require(not old.get("pilot") and not old.get("experiment"), "Use the original formal baseline.")
    require(1 <= a.seeds_per_weather <= 10 and 0 < a.duration <= 60, "seeds-per-weather must be 1..10; duration must be in (0,60].")
    require(a.pilot or (a.seeds_per_weather == 10 and a.duration == 60),
            "Short/reduced runs require --pilot; formal C3 is four weathers x ten slots x 60 s.")
    selected=[]
    for weather in WEATHERS:
        subset=sorted((g for g in old["groups"] if g["weather"] == weather), key=lambda g: g["slot"])
        require(len(subset)==10 and [g["slot"] for g in subset]==list(range(10)),
                f"Baseline must contain ten groups for {weather}.")
        selected.extend(subset[:a.seeds_per_weather])
    # Round-robin weather order exercises every renderer condition early in a pilot.
    selected.sort(key=lambda g:(g["slot"],WEATHERS.index(g["weather"])))
    groups, tasks, baseline_hashes, common = [], [], {"manifest.json": sha(baseline / "manifest.json")}, None
    for group in selected:
        result_path = Path("groups") / group["id"] / "result.json"
        result = read_json(baseline / result_path)
        require(result["status"] == "included", f"Baseline group is not included: {group['id']}")
        cases = [c for c in result["cases"] if Path(c).name == "sea5_C3"]
        require(len(cases) == 1, "Expected exactly one original sea5/C3 run per group.")
        case = Path(cases[0])
        require((baseline / case).resolve().is_relative_to(baseline), "Invalid baseline path.")
        request = read_json(baseline / case / "requested.json")
        metrics = read_json(baseline / case / "metrics.json")
        run = read_json(baseline / case / "run.json")
        expected = seeds(old["policy"]["master_seed"], group["weather"], group["slot"], result["attempt"])
        require(run["status"] == "complete" and request["batch"]["method"] == "C3"
                and request["weather"] == group["weather"] and request["batch"]["sea"] == 5,
                f"Invalid baseline run: {case}")
        require(expected == result["seeds"] and all(
            expected[k] == request["batch"][k] == metrics[k] == run["config"]["batch"][k]
            for k in SEED_KEYS) and request["seed"] == run["config"]["seed"] == expected["sensor_seed"],
            f"Baseline seeds do not match the original worker generator: {case}")
        config = {k: copy.deepcopy(v) for k, v in request.items() if k not in ("batch", "seed", "weather")}
        require(common is None or config == common, "Baseline algorithm/configuration varies across cases.")
        common = config
        g = {**group, "attempt": result["attempt"], "seeds": expected, "baseline_case": str(case)}
        relative = str(Path("groups") / g["id"] / f"attempt_{g['attempt']:02d}" / "C3")
        groups.append(g)
        tasks.append(dict(group=g, attempt=g["attempt"], sea=CONDITION, method="C3", case=relative))
        for p in (result_path, *(case / n for n in ("requested.json", "metrics.json", "run.json", "simulink/batch_model.json"))):
            baseline_hashes[str(p)] = sha(baseline / p)
    # Algorithms, weights, stochastic generators and physical source model remain the baseline version.
    exact = {common[k] for k in ("model", "checkpoint", "landing_config", "navigation_config", "lidar_config")}
    runtime = {"main/run.py", "main/sensors.py", "main/config.py", "main/batch/__main__.py",
               "main/batch/worker.py", "main/batch/common.py", "main/batch/metrics.py", "main/batch/recording.py"}
    prefixes = ("navigation/", "configs/", "external/", "main/matlab/", "simulation/bridge/", "simulation/matlab/",
                "visual_modeling/", "visual_training/")
    protected = {p: h for p, h in old["source_sha256"].items()
                 if p in exact or p in runtime or p.startswith(prefixes)}
    require((exact | runtime) <= protected.keys(), "Baseline lacks required source hashes.")
    changed = [p for p, h in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    require(not changed, f"Baseline algorithm/model/weights/sensors changed: {changed}")
    common.update(weather="clear", duration=a.duration, save_sensors=False,
                  torch_threads=old["policy"]["torch_threads"], render_threads=old["policy"]["render_threads"])
    if os.environ.get("RENDER_PYTHON"):
        common["render_python"] = os.environ["RENDER_PYTHON"]
    from main.config import validate
    validate(common)
    policy = copy.deepcopy(old["policy"])
    policy.update(weather=list(WEATHERS), methods=["C3"], seas={CONDITION: dict(wind_mps=10., wave_height_m=2.)},
                  seeds_per_weather=a.seeds_per_weather, duration_s=a.duration, max_replacements=0)
    manifest = dict(schema=1, experiment="c3_four_weather_w10_h2", pilot=a.pilot,
                    condition=dict(wind_mps=10., wave_height_m=2., label="custom wind/wave condition; not sea state 5"),
                    git_commit=git_commit, reference_git_commit=REFERENCE_COMMIT,
                    baseline=str(baseline.relative_to(ROOT)),
                    baseline_sha256=baseline_hashes, launcher_sha256=script_hashes(),
                    policy=policy, base_config=common, groups=groups, source_sha256=fingerprints(common),
                    interpretation="Original four-weather/C3 seeds; all STOP/timeout/runaway outcomes retained; no seed replacement.")
    import yaml
    manifest["landing_snapshot"]=yaml.safe_load((ROOT/common["landing_config"]).read_text())
    # Analysis-only fixes may change without altering the scientific runtime.
    manifest["runtime_sha256"]=protected
    out.mkdir(parents=True, exist_ok=False)
    for task in tasks:
        (out / task["case"]).mkdir(parents=True)
    write_json(out / "manifest.json", manifest)
    write_json(out / "tasks.json", tasks)
    return manifest


def verify(out, manifest):
    require(manifest.get("experiment") == "c3_four_weather_w10_h2", "Wrong experiment directory.")
    require(manifest["launcher_sha256"] == script_hashes(), "Launcher changed since planning.")
    require(manifest["source_sha256"] == fingerprints(manifest["base_config"]), "Source/model/config changed since planning.")
    baseline = output_path(manifest["baseline"])
    require(all((baseline / p).is_file() and sha(baseline / p) == h
                for p, h in manifest["baseline_sha256"].items()), "Baseline evidence changed or is missing.")
    expected = [dict(group=g, attempt=g["attempt"], sea=CONDITION, method="C3",
                     case=str(Path("groups") / g["id"] / f"attempt_{g['attempt']:02d}" / "C3"))
                for g in manifest["groups"]]
    require(read_json(out / "tasks.json") == expected, "Frozen task list changed.")


def read_completed(out, manifest):
    rows, missing = [], []
    for task in read_json(out / "tasks.json"):
        folder = out / task["case"]
        require(folder.resolve().is_relative_to(out.resolve()), "Invalid task path")
        expected = seeds(manifest["policy"]["master_seed"], task["group"]["weather"],
                         task["group"]["slot"], task["attempt"])
        require(expected == task["group"]["seeds"], f"Frozen seeds differ from worker generator: {folder}")
        if not (folder / "metrics.json").exists():
            missing.append(task["group"]["id"])
            continue
        metrics = read_json(folder / "metrics.json")
        requested = read_json(folder / "requested.json")
        model = read_json(folder / "simulink/batch_model.json")
        run = read_json(folder / "run.json")
        require(run["config"] == requested and run["status"] == "complete", f"Run configuration/status mismatch: {folder}")
        require(all(metrics[k] == requested["batch"][k] == task["group"]["seeds"][k] for k in SEED_KEYS),
                f"Recorded seed mismatch: {folder}")
        require(model["wind_mps"] == requested["batch"]["wind_mps"] == 10.
                and model["wave_parameters"]["SignificantWaveHeight"] == requested["batch"]["wave_height_m"] == 2.
                and model["wind_seed"] == metrics["wind_seed"]
                and model["wave_parameters"]["RandomSeed"] == metrics["wave_seed"]
                and metrics["method"] == "C3" and metrics["weather"] == task["group"]["weather"]
                and metrics["group"] == task["group"]["id"] and metrics["sea"] == CONDITION,
                f"Physical condition mismatch: {folder}")
        require(metrics["replacement"] == task["attempt"] and metrics["slot"] == task["group"]["slot"]
                and requested["weather"] == task["group"]["weather"]
                and requested["seed"] == metrics["sensor_seed"]
                and requested["duration"] == manifest["policy"]["duration_s"],
                f"Frozen run identity mismatch: {folder}")
        require(len(model["wave_blocks"]) == 2, f"Missing paired wave blocks: {folder}")
        require((folder / "series.jsonl.gz").is_file(),
                f"Incomplete numeric evidence: {folder}")
        rows.append({**metrics, "path": task["case"]})
    return rows, missing


def finalize(out, manifest):
    rows, missing = read_completed(out, manifest)
    for row in rows:
        write_json(out / "groups" / row["group"] / "result.json", dict(
            status="included", attempt=row["replacement"],
            seeds={k: row[k] for k in SEED_KEYS}, cases=[row["path"]],
            note="One C3 sortie per group; all completed outcomes retained, no replacement."))
    return missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "analyze", "status"))
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", default="outputs/batch/formal_523_w6")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--matlab-sessions")
    p.add_argument("--seeds-per-weather", type=int, default=10)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--analysis-output", help="Write offline reports to a separate outputs/ subdirectory")
    p.add_argument("--duration", type=float, default=60.)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    require(a.workers > 0, "workers must be positive.")
    out = output_path(a.output)
    if a.command == "status":
        manifest=read_json(out/"manifest.json")
        tasks=read_json(out/"tasks.json")
        print(json.dumps(dict(planned=len(tasks),completed=sum((out/t["case"]/"metrics.json").exists() for t in tasks),
            by_weather={w:sum((out/t["case"]/"metrics.json").exists() for t in tasks if t["group"]["weather"]==w) for w in manifest["policy"]["weather"]},
            workers=[read_json(f) for f in sorted(out.glob("worker_*_current.json"))]),ensure_ascii=False,indent=2))
        return
    if a.command == "analyze":
        from .c3_analysis import analyze
        with (out / "coordinator.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            analyze(out, output_path(a.analysis_output) if a.analysis_output else None)
        return
    require(not a.analysis_output,"--analysis-output is only used with analyze")
    if a.command == "run":
        names = a.matlab_sessions.split(",") if a.matlab_sessions else []
        require(len(names) == a.workers and len(set(names)) == len(names) and all(names),
                "Provide one distinct, idle, shared MATLAB session per worker.")
        import matlab.engine
        available = set(matlab.engine.find_matlab())
        require(set(names) <= available, f"Requested sessions missing; available: {sorted(available)}")
        # Exercise the formerly failing route before starting workers.
        from main.run import run
        from scipy.stats import chi2
        require(abs(chi2.ppf(.95, 3) - 7.814727903251179) < 1e-8, "SciPy smoke check failed.")
    out = output_path(a.output)
    manifest = read_json(out / "manifest.json") if a.resume else make_plan(a)
    verify(out, manifest)
    print(f"CUSTOM C3: {len(manifest['groups'])} four-weather runs, wind=10 m/s, Hs=2 m, duration={manifest['policy']['duration_s']} s; {out}", flush=True)
    if a.command == "plan":
        return
    with (out / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.command == "run":
            for claim in (out / "groups").rglob("claim"):
                claim.rmdir()
            missing = finalize(out, manifest)
            if missing:
                # Existing supervisor owns worker groups and leaves shared MATLAB sessions alive.
                def interrupted(*_):
                    raise KeyboardInterrupt("Coordinator received SIGTERM")
                signal.signal(signal.SIGTERM, interrupted)
                supervise(out, manifest, a)
            require(not finalize(out, manifest), "Workers exited with unfinished tasks; inspect logs and --resume")
            verify(out, manifest)
            write_json(out / "source_verification.json", dict(unchanged=True, launcher_unchanged=True))
        from .c3_analysis import analyze
        analyze(out)


if __name__ == "__main__":
    main()
