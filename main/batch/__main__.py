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
import math
import yaml
from .common import ROOT, METHODS, WEATHERS, output_path, fingerprints, write_json, read_json, seeds
from .recovery import BatchWorkersFailed, RetryLedger, changed_sources, resume_sources


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "run", "analyze", "status"])
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="main/batch_closed_loop.yaml")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pilot", action="store_true", help="Mark outputs as development; never pool with formal results")
    p.add_argument("--c3-only", action="store_true", help="C3 preflight: both seas, four weathers, original paired seeds (80 sorties)")
    p.add_argument("--seeds-per-weather", type=int)
    p.add_argument("--weather", nargs="+", choices=WEATHERS)
    p.add_argument("--duration", type=float)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    p.add_argument("--render-device", choices=["CPU", "CUDA", "OPTIX"], default="OPTIX")
    p.add_argument("--render-python")
    p.add_argument("--renderer", choices=["cpu", "blender"], default="blender")
    p.add_argument("--matlab-sessions", help="Comma-separated dedicated shared names, one per worker; otherwise start owned sessions")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-infra-retries", type=int, default=2,
                   help="Extra worker restarts per case for recognized renderer faults; 0 disables; budget persists on resume")
    p.add_argument("--retry-delay-s", type=float, default=10., help="Initial restart delay; doubles on each retry, capped at 60 s")
    p.add_argument("--allow-retry-upgrade", action="store_true",
                   help="With --resume only: audit an upgrade limited to batch recovery code; keep manifest and scientific inputs unchanged")
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
    if a.c3_only: c["methods"] = ["C3"]
    if c.get("analysis_profile") == "thesis_523_15":
        from .energy import validate_policy
        validate_policy(c)
    if c["touchdown_policy"] != "existing_model_stop_proxy" or c["initial_conditions"] != "unchanged_model_nominal":
        raise ValueError("Unsupported evaluation policy")
    base.update(duration=c["duration_s"], device=a.device, render_device=a.render_device,
                renderer=a.renderer, save_sensors=False, torch_threads=c["torch_threads"],
                render_threads=c["render_threads"])
    if a.render_python: base["render_python"] = a.render_python
    from main.config import validate
    validate(base)
    return dict(schema=2, experiment="c3_preflight" if a.c3_only else "full_closed_loop",
                pilot=a.pilot, policy=c, base_config=base,
                landing_snapshot=yaml.safe_load((ROOT/base["landing_config"]).read_text()),
                git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                runtime=dict(python=sys.version, executable=sys.executable, platform=sys.platform),
                source_sha256=fingerprints(base),
                groups=[dict(id=f"{w}_s{s:02d}", weather=w, slot=s)
                        for w in c["weather"] for s in range(c["seeds_per_weather"])],
                interpretation=dict(touchdown="existing model STOP proxy, no stable-contact claim",
                    sea_comparison="combined wave-height and user-specified wind-speed change",
                    initial_conditions="fixed nominal model state; no invented perturbation distribution",
                    pairing="same physical-time/channel seeds across all selected sea/method runs",
                    exclusions="any runaway supersedes entire matched group; ledger and attempted-run sensitivity retained",
                    command_cost="sum squared command increments normalized by existing atan bounds, W=I, divided by dt"))


def supervise(out, manifest, a):
    if a.max_infra_retries < 0 or not math.isfinite(a.retry_delay_s) or a.retry_delay_s < 0:
        raise ValueError("Retry count and finite retry delay must be nonnegative")
    named = a.matlab_sessions is not None
    sessions = [s.strip() for s in a.matlab_sessions.split(",")] if named else [""]*a.workers
    if len(sessions) != a.workers or (named and (len(set(sessions)) != len(sessions) or not all(sessions))):
        raise ValueError("Provide exactly one distinct dedicated MATLAB session per worker")
    if named:
        import matlab.engine
        available=list(matlab.engine.find_matlab())
        if not set(sessions)<=set(available):
            raise RuntimeError(f"Shared sessions missing; requested {sessions}; available {available}")
    env = os.environ.copy()
    env.pop("MATLAB_SHARED_SESSION", None)  # Never inherit the single-run session into all workers.
    env.pop("LANDING_BATCH_RETRY_CASE", None)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    processes, handles = [], []
    owned_descendants = {}
    pending, handled, failures = {}, set(), []
    ledger = RetryLedger(out)
    task_cases = {t["case"] for t in read_json(out/"tasks.json")}

    def launch(i, retry_case=None):
        (out/f"worker_{i}_matlab.json").unlink(missing_ok=True)
        (out/f"worker_{i}_failure.json").unlink(missing_ok=True)
        write_json(out/f"worker_{i}_current.json", dict(case=retry_case or "MATLAB startup", started=time.time()))
        child_env = {**env, "LANDING_BATCH_MATLAB_SESSION": sessions[i]}
        if retry_case:
            child_env["LANDING_BATCH_RETRY_CASE"] = retry_case
        return subprocess.Popen([sys.executable, "-m", "main.batch.worker", str(out), str(i)],
            cwd=ROOT, env=child_env, stdout=handles[i], stderr=subprocess.STDOUT, start_new_session=True)

    try:
        for i in range(a.workers):
            log = (out / f"worker_{i}.log").open("a", buffering=1); handles.append(log)
            processes.append(launch(i))
        started, last_message = time.time(), 0
        with (out / "resources.jsonl").open("a", buffering=1) as resource:
            while True:
                now = time.time()
                for i in list(pending):
                    case, deadline = pending[i]
                    if now >= deadline:
                        processes[i] = launch(i, case)
                        del pending[i]
                # Remember only descendants of worker processes; never signal
                # independently launched shared MATLAB or unrelated jobs.
                raw=subprocess.check_output(["ps","-eo","pid,ppid,lstart,comm"],text=True)
                records=[line.split(None,7) for line in raw.splitlines()[1:]]
                parents={int(v[0]):int(v[1]) for v in records if len(v)==8}
                descendants={p.pid for p in processes}
                for _ in range(12): descendants.update(pid for pid,parent in parents.items() if parent in descendants)
                owned_descendants.update({int(v[0]):" ".join(v[2:]) for v in records if len(v)==8 and int(v[0]) in descendants-{p.pid for p in processes}})
                for i, p in enumerate(processes):
                    code = p.poll()
                    if code not in (None, 0) and p.pid not in handled:
                        handled.add(p.pid)
                        path = out/f"worker_{i}_failure.json"
                        failure = read_json(path) if path.exists() else {}
                        if failure.get("pid") != p.pid:
                            failure = {}
                        case = failure.get("case")
                        category = failure.get("category")
                        error = failure.get("error", f"worker exited {code} without a classified failure")
                        retry, number = False, 0
                        if case in task_cases and not (out/case/"metrics.json").exists():
                            # The dead process must have retained its exclusive task claim.
                            if not (out/case/"claim").is_dir(): category = None
                            retry, number = ledger.failure(case, i, p.pid, error, category, a.max_infra_retries)
                        if retry:
                            delay = min(60., a.retry_delay_s * 2 ** min(number-1, 10))
                            pending[i] = (case, now+delay)
                            print(f"[retry] worker {i}: {case}; {category}; retry {number}/{a.max_infra_retries} "
                                  f"in {delay:g}s with identical seeds; other workers continue", flush=True)
                        else:
                            failures.append(dict(worker=i, case=case, error=error, exitcode=code))
                            print(f"[failed] worker {i}: {case}; {error}; no automatic retry; draining other workers", flush=True)
                    current = out / f"worker_{i}_current.json"
                    if p.poll() is None and current.exists():
                        task = read_json(current)
                        if now-task["started"] > manifest["policy"]["wall_timeout_s"]:
                            raise TimeoutError(f"Worker {i} exceeded per-run wall timeout: {task}")
                if now-last_message >= 30:
                    done = sum((out/"groups"/g["id"]/"result.json").exists() for g in manifest["groups"])
                    record = dict(time=now, elapsed_s=now-started, groups_done=done,
                                  workers=[p.pid for p in processes if p.poll() is None],
                                  matlab_pids=[read_json(out/f'worker_{i}_matlab.json')['pid']
                                      for i,p in enumerate(processes) if p.poll() is None and (out/f'worker_{i}_matlab.json').exists()],
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
                ledger.refresh()
                if not pending and all(p.poll() is not None and (p.returncode == 0 or p.pid in handled) for p in processes):
                    break
                time.sleep(1)
        ledger.refresh()  # A metrics file may arrive between the last refresh and poll.
        if failures:
            write_json(out/"batch_failure.json", dict(time=time.time(), failures=failures,
                reason="Other workers drained; missing cases are infrastructure failures, not scientific outcomes"))
            raise BatchWorkersFailed(f"{len(failures)} worker(s) failed after bounded recovery; "
                                     "other workers finished. Inspect batch_failure.json and infrastructure_retries.json, then --resume")
        (out/"batch_failure.json").unlink(missing_ok=True)
    finally:
        for p in processes:
            if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
        # All workers are now reaped. A detached Engine/renderer child can survive
        # a native crash as well as an explicit stop. Never target shared sessions
        # or unrelated processes; verify the recorded start time and command.
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
                for method in policy["methods"]:
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
    if a.command == "status":
        print(__import__("json").dumps(status(out), ensure_ascii=False, indent=2)); return
    if a.command == "analyze":
        from .analysis import analyze
        import fcntl
        with (out/"coordinator.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            analyze(out)
        return
    if a.workers < 1: raise ValueError("workers must be positive")
    if a.max_infra_retries < 0 or not math.isfinite(a.retry_delay_s) or a.retry_delay_s < 0:
        raise ValueError("Retry count and finite retry delay must be nonnegative")
    if a.allow_retry_upgrade and (not a.resume or a.command != "run"):
        raise ValueError("--allow-retry-upgrade requires run --resume")
    if a.resume:
        manifest = read_json(out/"manifest.json")
    else:
        manifest = create(a)
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"manifest.json", manifest)
    expected=len(manifest['groups'])*2*len(manifest['policy']['methods'])
    print(f"{'PILOT' if manifest['pilot'] else manifest.get('experiment','FORMAL')}: {expected} runs, {a.workers} workers, {out}")
    if a.command == "plan": return
    # flock is released by the OS after interruption; resume cleans stale claims
    # only while holding this exclusive coordinator lock.
    import fcntl
    with (out/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_sources = (resume_sources(out, manifest, fingerprints(manifest["base_config"]), a.allow_retry_upgrade)
                            if a.resume else manifest["source_sha256"])
        (out/"execution").mkdir(exist_ok=True)
        write_json(out/"execution"/f"{time.time_ns()}.json", dict(time=time.time(),
            workers=a.workers, max_infra_retries=a.max_infra_retries, retry_delay_s=a.retry_delay_s,
            resume=a.resume, retry_upgrade=(out/"retry_source_upgrade.json").exists()))
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"coordinator received signal {signum}; resume with the same manifest")
        signal.signal(signal.SIGTERM, interrupted)
        (out/"coordinator.pid").write_text(str(os.getpid())+"\n")
        for claim in (out/"groups").rglob("claim"): claim.rmdir()
        while round_tasks(out,manifest):
            try:
                supervise(out, manifest, a)
            except BatchWorkersFailed:
                # Commit complete paired groups even when a different case failed.
                # Missing infrastructure cases never acquire a metrics.json or an outcome.
                round_tasks(out, manifest)
                raise
        current=fingerprints(manifest["base_config"])
        changed=changed_sources(expected_sources,current)
        original_changes=changed_sources(manifest["source_sha256"],current)
        write_json(out/"source_verification.json",dict(unchanged=not original_changes,
            changed_paths=original_changes,execution_unchanged=not changed,execution_changed_paths=changed,
            retry_upgrade="retry_source_upgrade.json" if (out/"retry_source_upgrade.json").exists() else None))
        if changed and not manifest["pilot"]:
            raise RuntimeError("Formal batch source/model/weights changed while running; see source_verification.json")
        from .analysis import analyze
        analyze(out)


def status(out):
    manifest=read_json(out/"manifest.json")
    tasks=read_json(out/"tasks.json") if (out/"tasks.json").exists() else []
    return dict(expected_sorties=len(manifest["groups"])*2*len(manifest["policy"]["methods"]),
        finalized_groups=sum((out/"groups"/g["id"]/"result.json").exists() for g in manifest["groups"]),
        planned_groups=len(manifest["groups"]), current_round_tasks=len(tasks),
        current_round_completed=sum((out/t["case"]/"metrics.json").exists() for t in tasks),
        completed_attempts=len(list((out/"groups").glob("*/attempt_*/sea*/metrics.json"))),
        active_workers=[read_json(p) for p in sorted(out.glob("worker_*_current.json"))],
        infrastructure_retries=read_json(out/"infrastructure_retries.json") if (out/"infrastructure_retries.json").exists() else None,
        batch_failure=read_json(out/"batch_failure.json") if (out/"batch_failure.json").exists() else None,
        analysis=read_json(out/"analysis/status.json") if (out/"analysis/status.json").exists() else None)


if __name__ == "__main__": main()
