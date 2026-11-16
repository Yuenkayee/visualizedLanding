"""Bounded infrastructure recovery; never change experiment seeds or outcomes."""
from pathlib import Path
import time

from .common import ROOT, read_json, write_json, sha


# Explicit opt-in for an interrupted pre-recovery batch, not a general hash bypass.
RETRY_UPGRADE_PATHS = frozenset({
    "main/batch/__main__.py", "main/batch/worker.py", "main/batch/recovery.py",
    "tests/test_batch_recovery.py",
})


class BatchWorkersFailed(RuntimeError):
    pass


def renderer_failure(exc):
    """Only errors raised by the isolated renderer RPC may trigger recovery.

    A CUDA-looking exception in the estimator, summaries or MATLAB is not enough.
    Do not follow exception context: a failed cleanup must supersede the RPC error.
    """
    if not isinstance(exc, (RuntimeError, TimeoutError)):
        return None
    tb = exc.__traceback__
    rpc = False
    while tb is not None:
        frame = tb.tb_frame
        rpc |= (Path(frame.f_code.co_filename).resolve() == ROOT / "main/sensors.py"
                and frame.f_code.co_name == "_request")
        tb = tb.tb_next
    if not rpc:
        return None
    message = str(exc).lower()
    if "illegal address in cuda queue" in message or "cuda_error_illegal_address" in message:
        return "renderer_gpu_illegal_address"
    if "cuda" in message and ("out of memory" in message or "cuda_error_out_of_memory" in message):
        return "renderer_gpu_out_of_memory"
    if isinstance(exc, TimeoutError) and message == "renderer timed out; inspect renderer.log":
        return "renderer_timeout"
    return None


class RetryLedger:
    """Coordinator is the sole writer; budgets persist across --resume."""
    def __init__(self, out):
        self.out = Path(out)
        self.path = self.out / "infrastructure_retries.json"
        self.data = read_json(self.path) if self.path.exists() else dict(schema=1, cases={})

    def failure(self, case, worker, pid, error, category, limit):
        entry = self.data["cases"].setdefault(case, dict(retries_started=0, failures=[]))
        retry = category is not None and entry["retries_started"] < limit
        entry["failures"].append(dict(time=time.time(), worker=worker, pid=pid,
            error=error, category=category, retry_scheduled=retry))
        entry["status"] = "retry_pending" if retry else "failed"
        if retry:
            # Reserve the budget durably BEFORE launching the replacement process.
            entry["retries_started"] += 1
        write_json(self.path, self.data)
        return retry, entry["retries_started"]

    def refresh(self):
        changed = False
        for case, entry in self.data["cases"].items():
            if entry["status"] != "completed" and (self.out / case / "metrics.json").exists():
                entry["status"] = "completed"
                changed = True
        if changed:
            write_json(self.path, self.data)


def changed_sources(old, new):
    return sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))


def resume_sources(out, manifest, current, allow_retry_upgrade=False):
    """Called with the coordinator lock held; preserve the original manifest."""
    out = Path(out)
    record = out / "retry_source_upgrade.json"
    original = manifest["source_sha256"]
    if record.exists():
        saved = read_json(record)
        if saved["manifest_sha256"] != sha(out / "manifest.json"):
            raise RuntimeError("Original manifest changed since retry upgrade")
        if saved["execution_source_sha256"] != current:
            raise RuntimeError("Source changed since retry upgrade; use a new output directory")
        return current
    changed = changed_sources(original, current)
    if not changed:
        return current
    if not allow_retry_upgrade:
        raise RuntimeError("Source/model/weights/config changed since manifest; use a new output directory. "
                           "Only the recovery-only patch supports --allow-retry-upgrade. "
                           f"Changed: {changed}")
    forbidden = set(changed) - RETRY_UPGRADE_PATHS
    removed = set(original) - set(current)
    if forbidden or removed:
        raise RuntimeError(f"Retry-only upgrade rejected: unrelated or removed sources {sorted(forbidden | removed)}")
    write_json(record, dict(schema=1, time=time.time(), manifest_sha256=sha(out / "manifest.json"),
        changed_paths=changed, original_source_sha256=original,
        execution_source_sha256=current,
        scope="Only batch scheduling/recovery and its tests; experiment inputs unchanged"))
    return current
