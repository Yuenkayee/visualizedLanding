#!/usr/bin/env python3
"""Standalone C3 experiment using baseline source hashes, independent of tooling commits."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from main.batch.common import fingerprints, output_path, read_json, seeds, sha, write_json
from main.batch.__main__ import supervise

CONDITION = "custom_w10_h2"
REFERENCE_COMMIT = "ff91248e5275ffd6649b6826e809971d84e7da84"
SEED_KEYS = ("wind_seed", "wave_seed", "sensor_seed")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def script_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in ("run.py", "run.sh")}


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
    require(1 <= a.limit <= 10 and 0 < a.duration <= 60, "limit must be 1..10; duration must be in (0,60].")
    selected = sorted((g for g in old["groups"] if g["weather"] == "clear"), key=lambda g: g["slot"])
    require(len(selected) == 10 and [g["slot"] for g in selected] == list(range(10)),
            "Baseline must contain the ten clear-weather groups.")
    selected = selected[:a.limit]
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
        expected = seeds(old["policy"]["master_seed"], "clear", group["slot"], result["attempt"])
        require(run["status"] == "complete" and request["batch"]["method"] == "C3"
                and request["weather"] == "clear" and request["batch"]["sea"] == 5,
                f"Invalid baseline run: {case}")
        require(expected == result["seeds"] and all(
            expected[k] == request["batch"][k] == metrics[k] == run["config"]["batch"][k]
            for k in SEED_KEYS) and request["seed"] == run["config"]["seed"] == expected["sensor_seed"],
            f"Baseline seeds do not match the original worker generator: {case}")
        config = {k: copy.deepcopy(v) for k, v in request.items() if k not in ("batch", "seed", "weather")}
        require(common is None or config == common, "Baseline algorithm/configuration varies across clear cases.")
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
    policy.update(weather=["clear"], methods=["C3"], seas={CONDITION: dict(wind_mps=10., wave_height_m=2.)},
                  seeds_per_weather=a.limit, duration_s=a.duration, max_replacements=0)
    manifest = dict(schema=1, experiment="c3_clear_w10_h2", pilot=a.duration < 60 or a.limit < 10,
                    condition=dict(wind_mps=10., wave_height_m=2., label="custom wind/wave condition; not sea state 5"),
                    git_commit=git_commit, reference_git_commit=REFERENCE_COMMIT,
                    baseline=str(baseline.relative_to(ROOT)),
                    baseline_sha256=baseline_hashes, launcher_sha256=script_hashes(),
                    policy=policy, base_config=common, groups=groups, source_sha256=fingerprints(common),
                    interpretation="Original clear/C3 seeds; all STOP/timeout/runaway outcomes retained; no seed replacement.")
    out.mkdir(parents=True, exist_ok=False)
    for task in tasks:
        (out / task["case"]).mkdir(parents=True)
    write_json(out / "manifest.json", manifest)
    write_json(out / "tasks.json", tasks)
    return manifest


def verify(out, manifest):
    require(manifest.get("experiment") == "c3_clear_w10_h2", "Wrong experiment directory.")
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
                and metrics["method"] == "C3" and metrics["weather"] == "clear"
                and metrics["group"] == task["group"]["id"] and metrics["sea"] == CONDITION,
                f"Physical condition mismatch: {folder}")
        require((folder / "series.jsonl.gz").is_file(),
                f"Incomplete numeric evidence: {folder}")
        rows.append({**metrics, "path": task["case"]})
    return rows, missing


def analyze(out, manifest):
    verify(out, manifest)
    rows, missing = read_completed(out, manifest)
    dest = out / "analysis"
    dest.mkdir(exist_ok=True)
    paired = []
    for row in rows:
        group = next(g for g in manifest["groups"] if g["id"] == row["group"])
        old = read_json(output_path(manifest["baseline"]) / group["baseline_case"] / "metrics.json")
        item = dict(group=row["group"], **group["seeds"])
        for name, value in (("before", old), ("after", row)):
            for key in ("termination", "candidate", "duration_s", "terminal_stage", "nav_invalid_s", "invalid_wait_s", "radial_error_m"):
                item[f"{key}_{name}"] = value[key]
            for phase in value["phases"]:
                for key in ("entered", "completed", "completed_duration_s", "dwell_s", "invalid_wait_s"):
                    item[f"stage{phase['stage']}_{key}_{name}"] = phase[key]
        paired.append(item)
        write_json(out / "groups" / row["group"] / "result.json",
                   dict(status="included", cases=[row["path"]], seeds=group["seeds"], reason="retain every outcome; no replacement"))
    write_json(dest / "sorties.json", rows)
    write_json(dest / "paired_cases.json", paired)
    if paired:
        with (dest / "paired_cases.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader(); writer.writerows(paired)
    status = dict(complete=not missing, pilot=manifest["pilot"], planned=len(manifest["groups"]), completed=len(rows),
                  missing=missing, physical_conditions_and_seeds_passed=not missing,
                  stop_proxy=sum(r["candidate"] for r in rows), timeout=sum(r["termination"] == "timeout" for r in rows),
                  runaway=sum(r["termination"] == "runaway" for r in rows),
                  phase_completed=[sum(r["phases"][i]["completed"] for r in rows) for i in range(3)])
    write_json(dest / "status.json", status)
    text = ["# 晴天 C3：平均风速 10 m/s，有效波高 2.0 m", "",
            f"已完成 {len(rows)}/{status['planned']} 架次；STOP {status['stop_proxy']}；超时 {status['timeout']}；远离 {status['runaway']}。",
            f"进近/悬停/触舰阶段完成数：{status['phase_completed']}。试运行：{status['pilot']}。", "",
            "paired_cases.csv/json 中 before 为原 12 m/s、3.25 m 基线，after 为本次 10 m/s、2.0 m。",
            "相同原始种子不等于航迹变化后的图像或位置相关尾流完全相同。两个物理参数同时变化，不能分离单项影响。",
            "无效时间是各自完整航程统计；不同终止时间的记录应使用逐步序列作共同时间段比较。",
            "本脚本不修改 ff91248 的指标实现，不将其 Jctrl 用于本次优劣结论；压缩序列保留以便后续复核。",
            "STOP 沿用原模型判据，不意味着三阶段全部完成。所有超时与远离测例均保留，不替换种子。"]
    (dest / "report.md").write_text("\n".join(text) + "\n")
    print(json.dumps(status, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "analyze"))
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", default="outputs/batch/formal_523_w6")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--matlab-sessions")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--duration", type=float, default=60.)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    require(a.workers > 0, "workers must be positive.")
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
    manifest = read_json(out / "manifest.json") if a.resume or a.command == "analyze" else make_plan(a)
    verify(out, manifest)
    print(f"CUSTOM C3: {len(manifest['groups'])} clear runs, wind=10 m/s, Hs=2 m, duration={manifest['policy']['duration_s']} s; {out}", flush=True)
    if a.command == "plan":
        return
    with (out / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.command == "run":
            for claim in (out / "groups").rglob("claim"):
                claim.rmdir()
            _, missing = read_completed(out, manifest)
            if missing:
                # Existing supervisor owns worker groups and leaves shared MATLAB sessions alive.
                def interrupted(*_):
                    raise KeyboardInterrupt("Coordinator received SIGTERM")
                signal.signal(signal.SIGTERM, interrupted)
                supervise(out, manifest, a)
            verify(out, manifest)
            write_json(out / "source_verification.json", dict(unchanged=True, launcher_unchanged=True))
        analyze(out, manifest)


if __name__ == "__main__":
    main()
