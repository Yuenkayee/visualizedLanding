"""Spawned navigation workers with one writer per sensor record and manifest.

The coordinator owns the manifest. A job owns a whole record_id and all cases
that reuse it. Native CUDA/Blender state never crosses a process boundary.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback


def record_groups(plan):
    groups = {}
    identifiers = set()
    for case in plan:
        for name in ("id", "record_id"):
            value=case[name]
            if not isinstance(value,str) or Path(value).name!=value or value in (".","..",""):
                raise ValueError(f"Invalid {name}: {value!r}")
        if case["id"] in identifiers:
            raise ValueError(f"Duplicate case id: {case['id']}")
        identifiers.add(case["id"])
        groups.setdefault(case["record_id"], []).append(case)
    return [dict(record_id=key, cases=value) for key, value in groups.items()]


@contextmanager
def experiment_lock(output):
    """Reject simultaneous launches/resumes into the same output directory."""
    with (Path(output) / ".navigation.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Navigation experiment already active: {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class EventLog:
    """Keep full worker stdout locally; forward complete JSON progress lines."""
    def __init__(self, stream, emit):
        self.stream, self.emit, self.pending = stream, emit, ""

    def write(self, text):
        self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and "event" in event:
                self.emit(event)
        return len(text)

    def flush(self):
        self.stream.flush()


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt("navigation run interrupted")


def _worker(index, tasks, events, payload, initialize, execute, log_directory):
    # Each worker and its renderer have their own process group. Cleanup can
    # terminate a failed worker's renderer without touching the user's terminal.
    os.setsid()
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    pid = os.getpid()
    logfile = Path(log_directory) / f"worker_{index:02d}_{pid}.log"

    def emit(event):
        events.send({**event, "worker": index, "pid": pid})

    emit(dict(event="worker_started", log=str(logfile)))
    try:
        with logfile.open("a", buffering=1) as stream, redirect_stderr(stream), redirect_stdout(EventLog(stream, emit)):
            state, details = initialize(payload)
            emit(dict(event="worker_ready", **details))
            while True:
                job = tasks.get()
                if job is None:
                    break
                emit(dict(event="job_started", record_id=job["record_id"]))
                execute(job, state)
                emit(dict(event="job_complete", record_id=job["record_id"]))
    except BaseException as exc:
        with logfile.open("a") as stream:
            stream.write(traceback.format_exc())
        emit(dict(event="worker_failed", error=f"{type(exc).__name__}: {exc}", log=str(logfile)))
        raise
    finally:
        emit(dict(event="worker_stopped"))


def _stop(processes, groups):
    def send(process, sig):
        try:
            if process.pid in groups:
                os.killpg(process.pid, sig)
            elif process.is_alive():
                os.kill(process.pid, sig)
        except ProcessLookupError:
            pass

    # Bounded grace period for Sensors.__exit__ and partial-result bookkeeping.
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        for process in processes:
            send(process, sig)
        deadline = time.monotonic() + timeout
        for process in processes:
            process.join(timeout=max(0., deadline-time.monotonic()))
        # Even after a worker exits, its process group may contain a renderer.
        if sig == signal.SIGKILL:
            break


def _forward_events(connection, events, index):
    """A broken/truncated worker message cannot block coordinator liveness checks.

    Each pipe has exactly one worker writer. A separate daemon reader handles
    its framing; the coordinator waits on a thread-safe, in-process queue.
    """
    error = None
    try:
        while True:
            events.put(connection.recv())
    except EOFError:
        pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        events.put(dict(event="worker_channel_closed", worker=index, error=error))
        connection.close()


def run_workers(jobs, workers, payload, initialize, execute, log_directory, torch_threads=2):
    """Yield serialized progress; fail fast on worker crashes, preserving files.

    initialize/execute are top-level callables, so spawn also works on macOS.
    Application failures belong in execute's case events; a process failure
    aborts the experiment and never produces a successful completion event.
    """
    if not jobs:
        return
    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    tasks, events = context.Queue(), queue.Queue()
    processes, groups = [], set()
    channels, readers = [], []
    completed, stopped = set(), set()
    count = min(workers, len(jobs))
    env = dict(OMP_NUM_THREADS=str(torch_threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    saved = {key: os.environ.get(key) for key in env}
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    success = False
    try:
        # tmux shutdown/kill as well as Ctrl-C must reach worker/renderer cleanup.
        for sig in handlers:
            signal.signal(sig, _interrupt)
        # Set before spawn/imports, including NumPy's native thread pools.
        os.environ.update(env)
        try:
            for index in range(count):
                receiver, sender = context.Pipe(duplex=False)
                channels.append(receiver)
                process = context.Process(target=_worker, args=(index, tasks, sender, payload,
                    initialize, execute, log_directory), name=f"navigation-{index}")
                try:
                    process.start()
                finally:
                    sender.close()
                processes.append(process)
                reader = threading.Thread(target=_forward_events, args=(receiver, events, index), daemon=True)
                reader.start()
                readers.append(reader)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        for job in jobs:
            tasks.put(job)
        for _ in processes:
            tasks.put(None)
        heartbeat = time.monotonic()
        while len(completed) < len(jobs) or len(stopped) < count:
            try:
                event = events.get(timeout=1)
            except queue.Empty:
                crashed = [(p.pid, p.exitcode) for p in processes if p.exitcode not in (None, 0)]
                if crashed:
                    raise RuntimeError(f"Navigation worker exited unexpectedly (pid, exitcode): {crashed}")
                if all(p.exitcode is not None for p in processes):
                    raise RuntimeError("Navigation workers exited before all completion events were received")
                if time.monotonic()-heartbeat >= 10:
                    yield dict(event="workers_waiting", completed_records=len(completed), total_records=len(jobs))
                    heartbeat = time.monotonic()
                continue
            if event["event"] == "worker_started":
                groups.add(event["pid"])
            elif event["event"] == "worker_channel_closed":
                if event["worker"] not in stopped:
                    process = processes[event["worker"]]
                    process.join(timeout=.1)
                    raise RuntimeError(f"Navigation worker exited unexpectedly: pid={process.pid}, "
                        f"exitcode={process.exitcode}, channel={event['error']}")
                continue
            elif event["event"] == "worker_failed":
                yield event
                raise RuntimeError(f"Worker {event['worker']} failed: {event['error']}; inspect {event['log']}")
            elif event["event"] == "job_complete":
                completed.add(event["record_id"])
            elif event["event"] == "worker_stopped":
                stopped.add(event["worker"])
            yield event
        for process in processes:
            process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError(f"Navigation worker {process.pid} did not exit cleanly: {process.exitcode}")
        success = True
    finally:
        if not success:
            # A child can die after setsid but before its first queue message.
            for process in processes:
                try:
                    if os.getpgid(process.pid) == process.pid:
                        groups.add(process.pid)
                except ProcessLookupError:
                    pass
            _stop(processes, groups)
        tasks.cancel_join_thread()
        tasks.close()
        for reader in readers:
            reader.join(timeout=1)
        for connection in channels:
            connection.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
