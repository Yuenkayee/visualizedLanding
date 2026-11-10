"""Tooling commits may advance HEAD; experiment runtime content stays protected."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from main.batch import common


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "custom_c3_launcher", repository / "outputs/tools/c3_w10_h2/run.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    def commit(message):
        git("add", ".")
        git("-c", "user.name=Launcher Test", "-c", "user.email=launcher@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", message)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    config = yaml.safe_load((repository / "main/navigation_feedback.yaml").read_text())
    policy = yaml.safe_load((repository / "main/batch_closed_loop.yaml").read_text())
    protected = {config[k] for k in ("model", "checkpoint", "landing_config", "navigation_config", "lidar_config")}
    protected.update(("main/run.py", "main/sensors.py", "main/config.py", "main/batch/__main__.py",
                      "main/batch/worker.py", "main/batch/common.py", "main/batch/metrics.py",
                      "main/batch/recording.py", "main/matlab/landing_batch_configure.m"))
    for name in protected:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("baseline fixture\n")
    first_commit = commit("baseline runtime")
    baseline = tmp_path / "outputs/batch/baseline"
    baseline.mkdir(parents=True)
    groups = [dict(id=f"clear_s{i:02d}", weather="clear", slot=i) for i in range(10)]
    launcher.write_json(baseline / "manifest.json", dict(pilot=False, policy=policy, groups=groups,
        source_sha256={p: launcher.sha(tmp_path / p) for p in protected}))
    for group in groups:
        seed = launcher.seeds(policy["master_seed"], "clear", group["slot"], 0)
        relative = Path("groups") / group["id"] / "attempt_00/sea5_C3"
        case = baseline / relative
        (case / "simulink").mkdir(parents=True)
        request = copy.deepcopy(config)
        request.update(seed=seed["sensor_seed"], batch=dict(method="C3", sea=5, **seed))
        launcher.write_json(case / "requested.json", request)
        launcher.write_json(case / "metrics.json", seed)
        launcher.write_json(case / "run.json", dict(status="complete", config=request))
        launcher.write_json(case / "simulink/batch_model.json", {})
        launcher.write_json(baseline / "groups" / group["id"] / "result.json",
                            dict(status="included", attempt=0, seeds=seed, cases=[str(relative)]))
    # Reproduce the user's workflow: only the tooling commit advances HEAD.
    tool = tmp_path / "outputs/tools/example.txt"
    tool.parent.mkdir(parents=True)
    tool.write_text("auxiliary launcher\n")
    current_commit = commit("add launcher")
    assert current_commit != first_commit
    args = SimpleNamespace(baseline=str(baseline), output=str(tmp_path / "outputs/batch/new"),
                           limit=6, duration=2.)
    return launcher, args, current_commit, tmp_path


def test_new_tooling_commit_can_plan_and_records_actual_head(experiment):
    launcher, args, head, root = experiment
    manifest = launcher.make_plan(args)
    assert manifest["git_commit"] == head
    assert manifest["reference_git_commit"] == launcher.REFERENCE_COMMIT
    assert manifest["pilot"] and len(manifest["groups"]) == 6
    tasks = json.loads((Path(args.output) / "tasks.json").read_text())
    assert len(tasks) == 6 and {t["method"] for t in tasks} == {"C3"}
    assert manifest["policy"]["seas"] == {"custom_w10_h2": {"wind_mps": 10., "wave_height_m": 2.}}
    launcher.verify(Path(args.output), manifest)
    # The pre-existing freeze protection still rejects changes after planning.
    (root / "main/run.py").write_text("changed after planning\n")
    with pytest.raises(RuntimeError, match="Source/model/config changed since planning"):
        launcher.verify(Path(args.output), manifest)


@pytest.mark.parametrize("changed", ["main/run.py", "main/batch/worker.py",
                                    "main/matlab/landing_batch_configure.m",
                                    "external/innerloop/simu/innerLoop_landing.slx"])
def test_changed_runtime_is_rejected_before_creating_output(experiment, changed):
    launcher, args, _, root = experiment
    (root / changed).write_text("changed runtime\n")
    with pytest.raises(RuntimeError, match="Baseline algorithm/model/weights/sensors changed"):
        launcher.make_plan(args)
    assert not Path(args.output).exists()
