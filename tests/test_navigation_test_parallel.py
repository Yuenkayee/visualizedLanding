"""Concurrency, ownership, recovery and bounded cancellation regressions."""
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import pytest
import yaml

from main import navigation_tests as benchmark
from main.navigation_test_data import cases, write_json
from main.navigation_test_scheduler import experiment_lock, record_groups, run_workers
from main import navigation_test_scheduler as scheduler


def _initialize(payload):
    return payload, {"initialized": True}


def _execute_overlap(job, state):
    directory = Path(state["directory"])
    started = time.time()
    (directory / (job["record_id"] + ".started")).write_text(str(os.getpid()))
    deadline = time.monotonic() + 10
    while len(list(directory.glob("*.started"))) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError("No second concurrent worker")
        time.sleep(.02)
    time.sleep(.1)
    print(json.dumps(dict(event="test_result", record=job["record_id"], start=started,
                         end=time.time(), cases=[c["id"] for c in job["cases"]])), flush=True)


def _execute_crash(job, state):
    os._exit(17)


def _truncated_writer(index, tasks, channel, payload, initialize, execute, log_directory):
    os.setsid()
    channel.send(dict(event="worker_started",worker=index,pid=os.getpid()))
    # Abrupt native crash halfway through a multiprocessing Connection frame.
    os.write(channel.fileno(),struct.pack("!i",1000000)+b"partial")
    os._exit(17)


def _execute_child(job, state):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        print(json.dumps(dict(event="child_started", child=child.pid)), flush=True)
        time.sleep(300)
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_record_groups_are_exclusive_and_preserve_method_order():
    config = yaml.safe_load((benchmark.ROOT / "configs/navigation_tests.yaml").read_text())
    plan = cases(config)
    jobs = record_groups(plan)
    assert len(jobs) == 480 and sum(len(j["cases"]) for j in jobs) == 1224
    for job in jobs:
        assert {c["record_id"] for c in job["cases"]} == {job["record_id"]}
        assert job["cases"] == [c for c in plan if c["record_id"] == job["record_id"]]
    with pytest.raises(ValueError, match="Duplicate"):
        record_groups([plan[0], plan[0]])


def test_spawned_workers_actually_overlap_and_forward_progress(tmp_path):
    jobs = record_groups([dict(id="a", record_id="a"), dict(id="a_again", record_id="a"),
                          dict(id="b", record_id="b")])
    events = list(run_workers(jobs, 2, dict(directory=str(tmp_path)), _initialize,
                              _execute_overlap, tmp_path / "logs"))
    results = [e for e in events if e["event"] == "test_result"]
    assert len(results) == 2 and len({e["pid"] for e in results}) == 2
    assert max(e["start"] for e in results) < min(e["end"] for e in results)
    assert sorted(c for e in results for c in e["cases"]) == ["a", "a_again", "b"]
    assert sum(e["event"] == "job_complete" for e in events) == 2
    assert sum(e["event"] == "worker_stopped" for e in events) == 2
    assert len(list((tmp_path / "logs").glob("worker_*.log"))) == 2


def test_worker_crash_fails_without_hanging(tmp_path):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited unexpectedly.*17"):
        list(run_workers([dict(record_id="crash", cases=[])], 1, {}, _initialize,
                         _execute_crash, tmp_path / "logs"))
    assert time.monotonic() - started < 20


def test_truncated_worker_message_does_not_block_liveness(tmp_path,monkeypatch):
    monkeypatch.setattr(scheduler,"_worker",_truncated_writer)
    started=time.monotonic()
    with pytest.raises(RuntimeError,match="exited unexpectedly"):
        list(run_workers([dict(record_id="crash",cases=[])],1,{},_initialize,
                         _execute_crash,tmp_path/"logs"))
    assert time.monotonic()-started < 15


def test_cancellation_cleans_worker_and_its_child(tmp_path):
    events = run_workers([dict(record_id="child", cases=[])], 1, {}, _initialize,
                         _execute_child, tmp_path / "logs")
    try:
        for event in events:
            if event["event"] == "child_started":
                worker, child = event["pid"], event["child"]
                break
        else:
            pytest.fail("Child never started")
    finally:
        events.close()
    for pid in (worker, child):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_terminal_shutdown_cleans_descendants(tmp_path, sig):
    script = tmp_path / "launch.py"
    script.write_text(f'''import sys, json, subprocess, time, os
from pathlib import Path
sys.path.insert(0, {str(benchmark.ROOT)!r})
from main.navigation_test_scheduler import run_workers
def initialize(payload):
    return payload, {{}}
def execute(job, state):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        print(json.dumps(dict(event="child_started", child=child.pid)), flush=True)
        time.sleep(300)
    finally:
        if child.poll() is None: child.terminate()
        child.wait(timeout=5)
if __name__ == "__main__":
    root = Path(sys.argv[1])
    for event in run_workers([dict(record_id="child", cases=[])], 1, {{}}, initialize, execute, root/"logs"):
        if event["event"] == "child_started":
            (root/"ready.json").write_text(json.dumps(event))
''')
    with (tmp_path / "console.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(script), str(tmp_path)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / "ready.json").exists() and process.poll() is None:
                assert time.monotonic() < deadline, "worker startup timed out"
                time.sleep(.05)
            event = json.loads((tmp_path / "ready.json").read_text())
            process.send_signal(sig)
            assert process.wait(timeout=15) != 0
            for pid in (event["pid"], event["child"]):
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


def test_output_lock_prevents_second_writer_and_releases(tmp_path):
    with experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already active"):
            with experiment_lock(tmp_path):
                pytest.fail("second writer entered")
    with experiment_lock(tmp_path):
        pass


def state(tmp_path):
    return dict(output=tmp_path, resume=False, source=None, source_trajectory=None,
                base={}, landing={}, cal=None, geometry={}, nav={}, duration=.1,
                config={}, checkpoint="weights", detector=None)


def group():
    return dict(record_id="shared", cases=[dict(id="main", methods=["N0", "N3"]),
                                          dict(id="stress", methods=["N3"])])


def test_group_generates_shared_record_once_and_serializes_methods(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(benchmark, "generate_record", lambda *a: calls.append("generate"))
    monkeypatch.setattr(benchmark, "digest", lambda *a: "record-v1")
    monkeypatch.setattr(benchmark, "replay_record", lambda r, t, c, *a, **k:
                        calls.append((c["id"], c["method"])))
    benchmark.run_record_group(group(), state(tmp_path))
    assert calls == ["generate", ("main", "N0"), ("main", "N3"), ("stress", "N3")]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(events) == 2 and all(e["status"] == "complete" for e in events)


def test_generation_failure_accounts_for_every_dependent_case(tmp_path, monkeypatch, capsys):
    def fail(*args):
        raise ValueError("bad record")
    monkeypatch.setattr(benchmark, "generate_record", fail)
    monkeypatch.setattr(benchmark, "replay_record", lambda *a, **kw: pytest.fail("replayed failed record"))
    benchmark.run_record_group(group(), state(tmp_path))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    finished = [e for e in events if e["event"] == "case_finished"]
    assert [e["case"] for e in finished] == ["main", "stress"]
    assert all(e["failures"][0]["phase"] == "record" and e["status"] == "failed" for e in finished)


@pytest.mark.parametrize("intact", [True, False])
def test_resume_only_skips_results_when_record_is_reused(tmp_path, monkeypatch, intact):
    payload = state(tmp_path)
    payload["resume"] = True
    write_json(tmp_path / "records/shared/run.json", dict(status="complete"))
    calls = []
    monkeypatch.setattr(benchmark, "verify_record", lambda *a: intact)
    monkeypatch.setattr(benchmark, "digest", lambda *a: "record-v1")
    monkeypatch.setattr(benchmark, "verify_result", lambda *a: True)
    monkeypatch.setattr(benchmark, "generate_record", lambda *a: calls.append("generate"))
    monkeypatch.setattr(benchmark, "replay_record", lambda *a, **k: calls.append("replay"))
    benchmark.run_record_group(group(), payload)
    assert calls == ([] if intact else ["generate", "replay", "replay", "replay"])


def test_result_input_binding_survives_interrupted_regeneration(tmp_path):
    write_json(tmp_path/"run.json",dict(status="complete",planned_duration_s=1,
        completed_duration_s=1,input_record_sha256="old-record"))
    (tmp_path/"steps.jsonl").write_text("{}\n")
    write_json(tmp_path/"result_hashes.json",{n:benchmark.digest(tmp_path/n) for n in ("run.json","steps.jsonl")})
    assert benchmark.verify_result(tmp_path,"old-record")
    assert not benchmark.verify_result(tmp_path,"regenerated-record")


def test_interrupted_record_validation_keeps_durable_checking_marker(tmp_path,monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(benchmark,"generate_record",interrupt)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_record_group(group(),state(tmp_path))
    assert json.loads((tmp_path/"record_checks/shared.json").read_text())["status"] == "checking"


def test_resume_retains_previous_attempt_runtime(tmp_path,monkeypatch):
    from main import navigation_test_analysis as analysis
    previous=dict(attempt_id="first",status="interrupted",runtime=dict(gpu="old GPU"),workers=[])
    write_json(tmp_path/"manifest.json",dict(attempts=[previous]))
    config=dict(evaluation=dict(bootstrap_samples=1),workers=2,torch_threads=2,render_threads=4,
                device="cpu",execution=dict(workers_effective=2))
    plan=[dict(id="case",record_id="rec")]
    monkeypatch.setattr(benchmark,"runtime",lambda *a:dict(gpu="new GPU"))
    monkeypatch.setattr(benchmark,"navigation_config",lambda *a:{})
    monkeypatch.setattr(analysis,"analyze",lambda *a,**kw:{})
    monkeypatch.setattr(benchmark,"run_workers",lambda *a:iter([
        dict(event="worker_ready",runtime=dict(gpu="new GPU"),model_load_s=.1,network_input_size=512),
        dict(event="case_finished",case="case",status="complete",failures=[],skipped_methods=["N3"])]))
    assert benchmark.execute_experiment(tmp_path,dict(fingerprint="fixed"),config,plan,[],{},{},None,{},
        Path("weights"),None,None,None,{},False,True)==0
    metadata=json.loads((tmp_path/"manifest.json").read_text())
    assert metadata["attempts"][0]==previous
    assert metadata["attempts"][1]["runtime"]["gpu"]=="new GPU"
    assert metadata["attempts"][1]["status"]=="complete"


@pytest.mark.parametrize("name", ["workers", "render_threads", "torch_threads"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_concurrency_config_rejects_invalid_values(name, value):
    config = yaml.safe_load((benchmark.ROOT / "configs/navigation_tests.yaml").read_text())
    config[name] = value
    with pytest.raises(ValueError, match=name):
        benchmark.validate_config(config)
