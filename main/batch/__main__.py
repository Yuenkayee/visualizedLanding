"""Plan/run/resume/analyze the paired 320-sortie closed-loop experiment."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import gzip
import shutil
import yaml
from .common import ROOT, METHODS, WEATHERS, output_path, fingerprints, write_json, read_json, seeds


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "run", "analyze"])
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="main/batch_closed_loop.yaml")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pilot", action="store_true", help="Mark outputs as development; never pool with formal results")
    p.add_argument("--seeds-per-weather", type=int)
    p.add_argument("--weather", nargs="+", choices=WEATHERS)
    p.add_argument("--duration", type=float)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    p.add_argument("--render-device", choices=["CPU", "CUDA", "OPTIX"], default="OPTIX")
    p.add_argument("--render-python")
    p.add_argument("--renderer", choices=["cpu", "blender"], default="blender")
    p.add_argument("--matlab-sessions", help="Comma-separated dedicated shared names, one per worker; otherwise start owned sessions")
    p.add_argument("--resume", action="store_true")
    return p


def create(a):
    c = yaml.safe_load((ROOT / a.config).read_text())
    base = yaml.safe_load((ROOT / c["base_config"]).read_text())
    if not a.pilot and (a.weather or a.duration is not None or a.seeds_per_weather is not None or a.renderer != "blender"):
        raise ValueError("Matrix/duration/CPU-render overrides require --pilot; formal uses the frozen experiment YAML")
    if a.weather: c["weather"] = a.weather
    if a.seeds_per_weather is not None: c["seeds_per_weather"] = a.seeds_per_weather
    if a.duration is not None: c["duration_s"] = a.duration
    if c["seeds_per_weather"] < 1 or not 0 <= c["representative_slot"] < c["seeds_per_weather"]:
        raise ValueError("Invalid seed count/representative slot")
    if c["methods"] != list(METHODS) or set(c["seas"]) != {5, 6}:
        raise ValueError("C0-C3 and paired seas 5/6 are required")
    if c["touchdown_policy"] != "existing_model_stop_proxy" or c["initial_conditions"] != "unchanged_model_nominal":
        raise ValueError("Unsupported evaluation policy")
    base.update(duration=c["duration_s"], device=a.device, render_device=a.render_device,
                renderer=a.renderer, save_sensors=False, torch_threads=c["torch_threads"],
                render_threads=c["render_threads"])
    if a.render_python: base["render_python"] = a.render_python
    from main.config import validate
    validate(base)
    return dict(schema=1, pilot=a.pilot, policy=c, base_config=base,
                git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                runtime=dict(python=sys.version, executable=sys.executable, platform=sys.platform),
                source_sha256=fingerprints(base),
                groups=[dict(id=f"{w}_s{s:02d}", weather=w, slot=s)
                        for w in c["weather"] for s in range(c["seeds_per_weather"])],
                interpretation=dict(touchdown="existing model STOP proxy, no stable-contact claim",
                    sea_comparison="combined wave-height and user-specified wind-speed change",
                    initial_conditions="fixed nominal model state; no invented perturbation distribution",
                    pairing="same physical-time/channel seeds across all eight sea/method runs",
                    exclusions="any runaway supersedes entire eight-run pair group; ledger and attempted-run sensitivity retained",
                    command_cost="sum squared command increments normalized by existing atan bounds, W=I, divided by dt"))


def supervise(out, manifest, a):
    sessions = a.matlab_sessions.split(",") if a.matlab_sessions else [""]*a.workers
    if len(sessions) != a.workers or (a.matlab_sessions and (len(set(sessions)) != len(sessions) or not all(sessions))):
        raise ValueError("Provide exactly one distinct dedicated MATLAB session per worker")
    env = os.environ.copy()
    env.pop("MATLAB_SHARED_SESSION", None)  # Never inherit the single-run session into all workers.
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    processes, handles = [], []
    owned_descendants = {}
    try:
        for i, session in enumerate(sessions):
            write_json(out/f"worker_{i}_current.json", dict(case="MATLAB startup", started=time.time()))
            log = (out / f"worker_{i}.log").open("a", buffering=1); handles.append(log)
            child_env = {**env, "LANDING_BATCH_MATLAB_SESSION": session}
            processes.append(subprocess.Popen([sys.executable, "-m", "main.batch.worker", str(out), str(i)],
                cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        started, last_message = time.time(), 0
        with (out / "resources.jsonl").open("a", buffering=1) as resource:
            while any(p.poll() is None for p in processes):
                now = time.time()
                # Remember only descendants of worker processes; never signal
                # independently launched shared MATLAB or unrelated jobs.
                raw=subprocess.check_output(["ps","-eo","pid,ppid,lstart,comm"],text=True)
                records=[line.split(None,7) for line in raw.splitlines()[1:]]
                parents={int(v[0]):int(v[1]) for v in records if len(v)==8}
                descendants={p.pid for p in processes}
                for _ in range(12): descendants.update(pid for pid,parent in parents.items() if parent in descendants)
                owned_descendants.update({int(v[0]):" ".join(v[2:]) for v in records if len(v)==8 and int(v[0]) in descendants-{p.pid for p in processes}})
                for i, p in enumerate(processes):
                    if p.poll() not in (None, 0):
                        raise RuntimeError(f"Worker {i} exited {p.returncode}; inspect worker_{i}.log, then --resume")
                    current = out / f"worker_{i}_current.json"
                    if p.poll() is None and current.exists():
                        task = read_json(current)
                        if now-task["started"] > manifest["policy"]["wall_timeout_s"]:
                            raise TimeoutError(f"Worker {i} exceeded per-run wall timeout: {task}")
                if now-last_message >= 30:
                    done = sum((out/"groups"/g["id"]/"result.json").exists() for g in manifest["groups"])
                    record = dict(time=now, elapsed_s=now-started, groups_done=done,
                                  workers=[p.pid for p in processes if p.poll() is None],
                                  load=os.getloadavg())
                    try:
                        record["process_rss_kib"] = subprocess.check_output(["ps", "-eo", "pid,ppid,rss,comm"], text=True)
                        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                        record["gpu_memory_mib_utilization_percent"] = gpu.stdout.strip()
                    except (OSError, subprocess.TimeoutExpired): pass
                    resource.write(__import__("json").dumps(record)+"\n")
                    completed=sum(1 for task in read_json(out/"tasks.json") if (out/task["case"]/"metrics.json").exists())
                    print(f"Completed current-round runs {completed}/{len(read_json(out/'tasks.json'))}; finalized groups {done}/{len(manifest['groups'])}; elapsed {now-started:.0f}s", flush=True)
                    last_message = now
                time.sleep(1)
        if any(p.returncode != 0 for p in processes): raise RuntimeError("Worker failed; inspect logs")
    finally:
        interrupted=any(p.poll() is None for p in processes)
        for p in processes:
            if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
        if interrupted:
            # A detached Engine child can survive a killed Python worker.
            # Verify its command before terminating that previously observed
            # owned process (PID reuse must not target an unrelated command).
            for pid,identity in owned_descendants.items():
                probe=subprocess.run(["ps","-p",str(pid),"-o","lstart=","-o","comm="],capture_output=True,text=True)
                if " ".join(probe.stdout.split())==" ".join(identity.split()) and any(x in probe.stdout.lower() for x in ("matlab","python","blender")):
                    try: os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError: pass
        for log in handles: log.close()
        for i in range(len(handles)):
            source=out/f"worker_{i}.log"
            with source.open("rb") as src, gzip.open(out/f"worker_{i}_{time.time_ns()}.log.gz","wb",compresslevel=5) as dst:
                shutil.copyfileobj(src,dst)
            source.unlink()
        for claim in (out/"groups").rglob("claim"):
            claim.rmdir()


def round_tasks(out, manifest):
    tasks=[]; policy=manifest["policy"]
    for group in manifest["groups"]:
        directory=out/"groups"/group["id"]
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/"result.json").exists(): continue
        for attempt in range(policy["max_replacements"]+1):
            folder=directory/f"attempt_{attempt:02d}"; folder.mkdir(exist_ok=True)
            if (folder/"attempt.json").exists() and read_json(folder/"attempt.json")["status"]=="excluded_runaway": continue
            cases=[]; pending=[]
            for sea in (5,6):
                for method in METHODS:
                    case=folder/f"sea{sea}_{method}"; case.mkdir(exist_ok=True)
                    cases.append(str(case.relative_to(out)))
                    if not (case/"metrics.json").exists():
                        pending.append(dict(group=group,attempt=attempt,sea=sea,method=method,case=cases[-1]))
            if pending:
                tasks.extend(pending); break
            runaway=[c for c in cases if read_json(out/c/"metrics.json")["termination"]=="runaway"]
            status="excluded_runaway" if runaway else "included"
            result=dict(status=status,group=group,attempt=attempt,
                seeds=seeds(policy["master_seed"],group["weather"],group["slot"],attempt),cases=cases,
                runaway_cases=runaway,reason="sustained flyaway; entire paired group superseded" if runaway else None)
            write_json(folder/"attempt.json",result)
            if not runaway:
                write_json(directory/"result.json",result); break
        else:
            raise RuntimeError(f"{group['id']}: replacement limit exhausted; no reduced denominator")
    write_json(out/"tasks.json",tasks)
    return tasks


def main():
    a = parser().parse_args(); out = output_path(a.output)
    if a.command == "analyze":
        from .analysis import analyze
        analyze(out); return
    if a.workers < 1: raise ValueError("workers must be positive")
    if a.resume:
        manifest = read_json(out/"manifest.json")
        if manifest["source_sha256"] != fingerprints(manifest["base_config"]):
            raise RuntimeError("Source/model/weights/config changed since manifest; use a new output directory")
    else:
        manifest = create(a)
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"manifest.json", manifest)
    print(f"{'PILOT' if manifest['pilot'] else 'FORMAL'}: {len(manifest['groups'])*8} runs, {a.workers} workers, {out}")
    if a.command == "plan": return
    # flock is released by the OS after interruption; resume cleans stale claims
    # only while holding this exclusive coordinator lock.
    import fcntl
    with (out/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for claim in (out/"groups").rglob("claim"): claim.rmdir()
        while round_tasks(out,manifest):
            supervise(out, manifest, a)
        current=fingerprints(manifest["base_config"])
        changed=[p for p,h in manifest["source_sha256"].items() if current.get(p)!=h]
        write_json(out/"source_verification.json",dict(unchanged=not changed,changed_paths=changed))
        if changed and not manifest["pilot"]:
            raise RuntimeError("Formal batch source/model/weights changed while running; see source_verification.json")
        from .analysis import analyze
        analyze(out)


if __name__ == "__main__": main()
