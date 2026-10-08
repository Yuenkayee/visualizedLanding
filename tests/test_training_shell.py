"""Run shell entry points with isolated Python/pip/tmux stand-ins; no downloads."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "server project"
    (root / "shell").mkdir(parents=True)
    for name in (
        "check_dependencies.sh",
        "train_rtx5090.sh",
        "python_environment.sh",
        "generate_multi_camera_dataset.sh",
    ):
        shutil.copy(Path("shell") / name, root / "shell" / name)
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in (
        "bash",
        "cat",
        "dirname",
        "mkdir",
        "tee",
        "date",
        "mktemp",
        "mv",
        "env",
        "printenv",
    ):
        (binary / name).symlink_to(shutil.which(name))
    log = tmp_path / "calls.jsonl"
    python_code = r"""
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['MOCK_ROOT'])
with open(os.environ['MOCK_CALLS'], 'a') as stream:
    stream.write(json.dumps([os.path.abspath(sys.argv[0]), args]) + '\n')
if args[:1] == ['-c'] and 'os.path.abspath(sys.executable)' in args[1]:
    print(os.path.abspath(sys.argv[0]))
    sys.exit(0)
if args[:1] == ['-c'] and 'expected = tuple' in args[1]:
    actual = os.environ.get('MOCK_PYTHON_VERSION', args[2])
    if actual != args[2]:
        print(f'expected Python {args[2]}, got {actual}; use --python', file=sys.stderr)
        sys.exit(2)
if args[:1] == ['-']:
    source = sys.stdin.read()
    if 'missing=[]' in source:
        sys.exit(0 if (root / '.installed').exists() else 2)
    if 'selected=set()' in source and os.environ.get('MOCK_TRANSITIVE_FAIL'):
        sys.exit(2)
if args == ['-m', 'pip', '--version']:
    sys.exit(1 if os.environ.get('MOCK_NO_PIP') and not (root / '.pip_ready').exists() else 0)
if args[:2] == ['-m', 'ensurepip']:
    if os.environ.get('MOCK_ENSUREPIP_FAIL'):
        sys.exit(1)
    (root / '.pip_ready').touch()
if args[:3] == ['-m', 'pip', 'install']:
    if os.environ.get('MOCK_INSTALL_FAIL'):
        print('pip download/permission error', file=sys.stderr)
        sys.exit(1)
    if not os.environ.get('MOCK_INSTALL_NO_EFFECT'):
        (root / '.installed').touch()
if args[:1] == ['-c'] and 'import numpy,scipy' in args[1] and os.environ.get('MOCK_IMPORT_FAIL'):
    sys.exit(1)
if 'visual_training.check_gpu' in args and os.environ.get('MOCK_GPU_FAIL'):
    sys.exit(1)
if 'visual_training.train' in args:
    for name in ('HTTPS_PROXY', 'CUDA_VISIBLE_DEVICES', 'CONDA_PREFIX', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'PIP_REQUIRE_VIRTUALENV'):
        assert os.environ.get(name) != 'stale', name
    print('TRAINING_STARTED')
    sys.exit(int(os.environ.get('MOCK_TRAIN_EXIT', '0')))
"""
    for name in ("python", "python3"):
        executable(binary / name, python_code)
    executable(
        binary / "uname",
        "import sys\nprint('Linux' if sys.argv[1] == '-s' else 'x86_64')\n",
    )
    executable(
        binary / "tmux",
        r"""
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ['MOCK_CALLS'], 'a') as stream:
    stream.write(json.dumps(['tmux', args]) + '\n')
if args[0] == 'has-session':
    sys.exit(0 if os.environ.get('MOCK_SESSION_EXISTS') else 1)
if args[0] == 'new-session':
    # An old tmux server can have a different interpreter and environment.
    child_env = dict(os.environ, PATH='/invalid/stale/path', LANDING_PYTHON='/invalid/python')
    for name in ('HTTPS_PROXY', 'CUDA_VISIBLE_DEVICES', 'CONDA_PREFIX', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'PIP_REQUIRE_VIRTUALENV'):
        child_env[name] = 'stale'
    command = args[args.index('env'):]
    command[0] = os.path.join(os.environ['PATH'], 'env')
    subprocess.run(command, env=child_env, check=False)
""",
    )
    environment = dict(
        os.environ, PATH=str(binary), MOCK_ROOT=str(root), MOCK_CALLS=str(log)
    )
    for name in (
        "LANDING_PYTHON",
        "CUDA_VISIBLE_DEVICES",
        "HTTPS_PROXY",
        "CONDA_PREFIX",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "PIP_REQUIRE_VIRTUALENV",
    ):
        environment.pop(name, None)
    return root, environment, log


def run(server, script, *args, cwd=None, **overrides):
    root, env, _ = server
    return subprocess.run(
        [shutil.which("bash"), str(root / "shell" / script), *args],
        cwd=cwd,
        env=dict(env, **overrides),
        text=True,
        capture_output=True,
        timeout=20,
    )


def calls(server):
    return [json.loads(line) for line in server[2].read_text().splitlines()]


def installs(server):
    return [
        (name, args)
        for name, args in calls(server)
        if args[:3] == ["-m", "pip", "install"]
    ]


@pytest.mark.parametrize("explicit", [False, True])
def test_dataset_launcher_uses_current_python_without_venv(server, explicit):
    args = ["--renderer", "cpu", "--output", "/data/images with spaces"]
    chosen = str(Path(server[1]["PATH"]) / "python")
    if explicit:
        chosen = str(server[0].parent / "selected python")
        shutil.copy(Path(server[1]["PATH"]) / "python", chosen)
        args += ["--python", "./selected python"]
    result = run(
        server, "generate_multi_camera_dataset.sh", *args, cwd=server[0].parent
    )
    assert result.returncode == 0, result.stderr
    launch = calls(server)[-1]
    assert launch[0] == chosen
    assert launch[1] == [
        "-u",
        "-m",
        "visual_training.data.blender_runtime",
        "--renderer",
        "cpu",
        "--output",
        "/data/images with spaces",
    ]
    assert not (server[0] / ".venv").exists()


def test_dataset_default_launch_with_no_options(server):
    result = run(server, "generate_multi_camera_dataset.sh")
    assert result.returncode == 0, result.stderr
    assert calls(server)[-1][1] == ["-u", "-m", "visual_training.data.blender_runtime"]


def test_installs_into_active_environment_then_reuses_it(server):
    result = run(server, "check_dependencies.sh", "--rtx5090")
    assert result.returncode == 0, result.stderr
    commands = installs(server)
    assert len(commands) == 1
    name, args = commands[0]
    assert name == str(Path(server[1]["PATH"]) / "python")
    assert args[args.index("--requirement") + 1] == "requirements-rtx5090.txt"
    assert "--retries" in args and "--timeout" in args
    assert not (server[0] / ".venv").exists()
    assert not (server[0] / ".tools").exists()
    server[2].write_text("")
    result = run(server, "check_dependencies.sh", "--rtx5090")
    assert result.returncode == 0, result.stderr
    assert not installs(server)
    assert any("visual_training.check_gpu" in args for _, args in calls(server))


def test_check_only_does_not_install_missing_packages_or_pip(server):
    result = run(
        server, "check_dependencies.sh", "--rtx5090", "--check", MOCK_NO_PIP="1"
    )
    assert result.returncode == 2
    assert "Rerun without --check" in result.stderr
    assert not installs(server)
    assert not any("ensurepip" in args for _, args in calls(server))
    assert not (server[0] / ".venv").exists()


def test_check_only_validates_imports_and_gpu(server):
    (server[0] / ".installed").touch()
    result = run(server, "check_dependencies.sh", "--rtx5090", "--check")
    assert result.returncode == 0, result.stderr
    assert any(
        args[:1] == ["-c"] and "import numpy,scipy" in args[1]
        for _, args in calls(server)
    )
    assert any("visual_training.check_gpu" in args for _, args in calls(server))
    assert not installs(server)


def test_bootstraps_pip_only_in_selected_interpreter(server):
    result = run(server, "check_dependencies.sh", "--rtx5090", MOCK_NO_PIP="1")
    assert result.returncode == 0, result.stderr
    bootstrap = [(name, args) for name, args in calls(server) if "ensurepip" in args]
    assert len(bootstrap) == 1
    assert bootstrap[0][0] == installs(server)[0][0]
    assert not (server[0] / ".venv").exists()


@pytest.mark.parametrize(
    "failure",
    [
        {"MOCK_INSTALL_FAIL": "1"},
        {"MOCK_NO_PIP": "1", "MOCK_ENSUREPIP_FAIL": "1"},
        {"MOCK_INSTALL_NO_EFFECT": "1"},
        {"MOCK_IMPORT_FAIL": "1"},
        {"MOCK_TRANSITIVE_FAIL": "1"},
        {"MOCK_GPU_FAIL": "1"},
        {"MOCK_PYTHON_VERSION": "3.11"},
    ],
)
def test_failed_prerequisite_never_starts_training_and_records_exit(server, failure):
    log = server[0] / "logs/failed.log"
    result = run(
        server, "train_rtx5090.sh", "--foreground", "--log", str(log), **failure
    )
    assert result.returncode != 0
    assert "TRAINING_STARTED" not in log.read_text()
    assert not any("visual_training.train" in args for _, args in calls(server))
    assert int(Path(str(log) + ".exit_code").read_text()) == result.returncode
    assert not (server[0] / ".venv").exists()


def test_tmux_job_uses_same_interpreter_and_orders_install_check_train(server):
    chosen = server[0].parent / "conda environment/bin/python"
    chosen.parent.mkdir(parents=True)
    shutil.copy(Path(server[1]["PATH"]) / "python", chosen)
    log = server[0] / "logs/training.log"
    result = run(
        server,
        "train_rtx5090.sh",
        "--python",
        str(chosen),
        "--log",
        str(log),
        "--",
        "--data-root",
        "/data/landing images",
        "--epochs",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert "TRAINING_STARTED" in log.read_text()
    assert Path(str(log) + ".exit_code").read_text() == "0\n"
    commands = calls(server)
    install = next(
        i
        for i, (_, args) in enumerate(commands)
        if args[:3] == ["-m", "pip", "install"]
    )
    check = next(
        i for i, (_, args) in enumerate(commands) if "visual_training.check_gpu" in args
    )
    training = next(
        i for i, (_, args) in enumerate(commands) if "visual_training.train" in args
    )
    assert install < check < training
    assert (
        commands[install][0]
        == commands[check][0]
        == commands[training][0]
        == str(chosen)
    )
    assert commands[training][1][-4:] == [
        "--data-root",
        "/data/landing images",
        "--epochs",
        "1",
    ]


@pytest.mark.parametrize("selector", ["option", "environment", "python3_fallback"])
def test_interpreter_selection_and_relative_paths(server, selector):
    binary = Path(server[1]["PATH"])
    args, env = [], {}
    if selector == "option":
        args = ["--python", "bin/python3"]
        env = {"LANDING_PYTHON": "/invalid/should-not-be-used"}
    elif selector == "environment":
        env = {"LANDING_PYTHON": str(binary / "python3")}
    else:
        (binary / "python").unlink()
    result = run(
        server, "check_dependencies.sh", "--rtx5090", *args, cwd=binary.parent, **env
    )
    assert result.returncode == 0, result.stderr
    assert installs(server)[0][0] == str(binary / "python3")


def test_unused_project_venv_is_never_selected_or_modified(server):
    old = server[0] / ".venv/bin/python"
    executable(old, "raise SystemExit('Old project Python must not run')\n")
    before = old.read_bytes()
    result = run(server, "train_rtx5090.sh", "--foreground")
    assert result.returncode == 0, result.stderr
    assert old.read_bytes() == before
    assert all(name != str(old) for name, _ in calls(server))


@pytest.mark.parametrize(
    "profile,version",
    [(["--rtx5090"], "3.12"), (["--without-blender", "--without-matlab"], "3.11")],
)
def test_profile_checks_python_without_creating_environment(server, profile, version):
    result = run(server, "check_dependencies.sh", *profile)
    assert result.returncode == 0, result.stderr
    check = next(
        args
        for _, args in calls(server)
        if args[:1] == ["-c"] and "expected = tuple" in args[1]
    )
    assert check[-1] == version
    assert not (server[0] / ".venv").exists()


def test_duplicate_tmux_session_is_not_started(server):
    result = run(server, "train_rtx5090.sh", MOCK_SESSION_EXISTS="1")
    assert result.returncode == 2
    assert "Session already exists" in result.stderr
    assert not installs(server)
    assert not (server[0] / "outputs/logs").exists()


def test_default_logs_are_unique_and_record_training_failure(server):
    for _ in range(2):
        result = run(server, "train_rtx5090.sh", MOCK_TRAIN_EXIT="7")
        assert result.returncode == 0, result.stderr  # tmux launch succeeded
    logs = list((server[0] / "outputs/logs").glob("rtx5090_*.log"))
    assert len(logs) == 2
    for log in logs:
        assert "TRAINING_STARTED" in log.read_text()
        assert Path(str(log) + ".exit_code").read_text() == "7\n"
