"""One process, one dedicated MATLAB session, disposable model/cache per run."""
import contextlib
import gzip
import os
from pathlib import Path
import shutil
import sys
import signal
import time
import traceback
from .common import ROOT, read_json, write_json, seeds


def prune(case):
    # Called only after run() closed all handles. Preserve audit + failure
    # evidence, never follow symlinks or touch the source model/checkpoint.
    for name in ("sensors", "simulink/cache"):
        p = case/name
        if p.is_symlink(): raise RuntimeError(f"Refusing to prune symlink: {p}")
        if p.is_dir(): shutil.rmtree(p)
    for p in (case/"simulink").glob("landing_live_*"):
        if p.is_file() and p.suffix in (".slx", ".slxc", ".autosave"): p.unlink()
    for p in case.glob("*.log"):
        with p.open("rb") as src, gzip.open(str(p)+".gz", "wb", compresslevel=5) as dst:
            shutil.copyfileobj(src, dst)
        p.unlink()


class SessionLease:
    """Finalize each plant while leaving this worker's MATLAB engine alive."""
    def __init__(self, session): self.session, self.engine = session, session.engine
    def __enter__(self): return self
    def call(self, *args): return self.session.call(*args)
    def __exit__(self, *exc): self.engine.finalize_external_model(nargout=1)


def run_case(out, task, manifest, worker, session):
    from main.config import load
    from main.run import run
    from .metrics import summarize
    p = manifest["policy"]
    group, attempt, sea, method = (task[k] for k in ("group", "attempt", "sea", "method"))
    case = out/task["case"]
    if (case/"metrics.json").exists(): return
    # Retain interrupted evidence; retry the identical condition on resume.
    if (case/"requested.json").exists():
        archive = case.parent/f"{case.name}_interrupted_{time.time_ns()}"
        archive.mkdir()
        for item in case.iterdir():
            if item.name != "claim": shutil.move(str(item), archive/item.name)
    seed = seeds(p["master_seed"], group["weather"], group["slot"], attempt)
    c = dict(manifest["base_config"])
    c.update(weather=group["weather"], seed=seed["sensor_seed"])
    c["batch"] = dict(method=method, sea=sea, slot=group["slot"], group=group["id"],
        replacement=attempt, **seed, **p["seas"][str(sea)], runaway=p["runaway"],
        sea_state_label=f"sea{sea}", matlab_threads=p["matlab_threads"])
    write_json(case/"requested.json", c)
    write_json(out/f"worker_{worker}_current.json", dict(case=str(case), started=time.time()))
    print(f"{group['id']} attempt={attempt} sea={sea} {method}", flush=True)
    try:
        with (case/"console.log").open("w", buffering=1) as console, contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            resolved, landing, calibration, geometry = load(case/"requested.json")
            run(resolved, landing, calibration, geometry, case, matlab_session=SessionLease(session))
        metrics = summarize(case, p)
        write_json(case/"metrics.json", metrics)
    except BaseException as exc:
        write_json(case/"worker_error.json", dict(error=str(exc), traceback=traceback.format_exc()))
        # Fail fast: infrastructure errors never cause random-seed replacement.
        raise
    finally:
        prune(case)


def main():
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f"batch worker received signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    out = Path(sys.argv[1]); worker = int(sys.argv[2]); manifest = read_json(out/"manifest.json")
    from simulation.bridge.matlab_session import MatlabSession
    session = os.environ.get("LANDING_BATCH_MATLAB_SESSION", "")
    if session:
        os.environ["MATLAB_SHARED_SESSION"] = session
    else:
        os.environ.pop("MATLAB_SHARED_SESSION", None)
    with MatlabSession() as owner:
        for task in read_json(out/"tasks.json"):
            case = out/task["case"]
            if (case/"metrics.json").exists(): continue
            try: (case/"claim").mkdir()
            except FileExistsError: continue
            try: run_case(out, task, manifest, worker, owner)
            finally: (case/"claim").rmdir()
        (out/f"worker_{worker}_current.json").unlink(missing_ok=True)


if __name__ == "__main__": main()
