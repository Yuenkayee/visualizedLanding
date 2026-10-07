"""Exercise installer failures and job ordering without downloads or a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "server project"
    (root / "shell").mkdir(parents=True)
    for name in ("check_dependencies.sh", "train_rtx5090.sh"):
        shutil.copy(Path("shell") / name, root / "shell" / name)
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in (
        "bash",
        "cat",
        "dirname",
        "mkdir",
        "tar",
        "tee",
        "date",
        "mktemp",
        "mv",
        "env",
        "printenv",
    ):
        (binary / name).symlink_to(shutil.which(name))
    log = tmp_path / "calls.jsonl"
    common = """
import json, os, pathlib, shutil, subprocess, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['MOCK_ROOT'])
with open(os.environ['MOCK_CALLS'], 'a') as stream:
    stream.write(json.dumps([pathlib.Path(sys.argv[0]).name, args]) + '\\n')
"""
    interpreter = tmp_path / "mock-python"
    executable(
        interpreter,
        common
        + """
if args[:1] == ['-'] and len(args) > 1:
    sys.exit(0 if (root / '.installed').exists() else 2)
if 'visual_training.train' in args:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != 'stale'
    assert os.environ.get('HTTPS_PROXY') != 'stale'
    print('TRAINING_STARTED')
sys.exit(0)
""",
    )
    uv = tmp_path / "mock-uv"
    executable(
        uv,
        common
        + """
if args[0] == '--version':
    print('uv 0.9.5')
elif args[0] == 'venv':
    if os.environ.get('MOCK_PYTHON_FAIL'):
        print('Python download failed', file=sys.stderr)
        sys.exit(1)
    dest = root / '.venv/bin/python'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(os.environ['MOCK_PYTHON'], dest)
elif args[:2] == ['pip', 'install']:
    (root / '.installed').touch()
""",
    )
    executable(
        binary / "python3",
        common
        + """
if args == ['-m', 'pip', '--version']:
    sys.exit(0)
if os.environ.get('MOCK_PIP_FAIL'):
    sys.exit(1)
target = pathlib.Path(args[args.index('--target') + 1]) / 'bin/uv'
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(os.environ['MOCK_UV'], target)
""",
    )
    executable(
        binary / "uname",
        "import sys\nprint('Linux' if sys.argv[1] == '-s' else 'x86_64')\n",
    )
    archive = tmp_path / "uv.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(uv, arcname="uv-x86_64-unknown-linux-gnu/uv")
    executable(
        binary / "curl",
        common
        + """
if os.environ.get('MOCK_CURL_FAIL'):
    print('curl: (56) unexpected eof while reading', file=sys.stderr)
    sys.exit(56)
shutil.copy(os.environ['MOCK_ARCHIVE'], args[args.index('-o') + 1])
""",
    )
    executable(
        binary / "tmux",
        common
        + """
if args[0] == 'has-session':
    sys.exit(0 if os.environ.get('MOCK_SESSION_EXISTS') else 1)
if args[0] == 'new-session':
    # Execute the detached payload synchronously to inspect its result in tests.
    child_env = dict(os.environ, HTTPS_PROXY='stale', CUDA_VISIBLE_DEVICES='stale')
    subprocess.run(args[args.index('env'):], env=child_env, check=False)
""",
    )
    environment = dict(
        os.environ,
        PATH=str(binary),
        MOCK_ROOT=str(root),
        MOCK_CALLS=str(log),
        MOCK_PYTHON=str(interpreter),
        MOCK_UV=str(uv),
        MOCK_ARCHIVE=str(archive),
    )
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("HTTPS_PROXY", None)
    return root, environment, log


def run(server, script, *args, **overrides):
    root, env, _ = server
    return subprocess.run(
        [shutil.which("bash"), str(root / "shell" / script), *args],
        env=dict(env, **overrides),
        text=True,
        capture_output=True,
        timeout=20,
    )


def calls(server):
    return [json.loads(line) for line in server[2].read_text().splitlines()]


def test_pypi_bootstrap_avoids_github_and_reuses_local_uv(server):
    result = run(server, "check_dependencies.sh", "--rtx5090", MOCK_CURL_FAIL="1")
    assert result.returncode == 0, result.stderr
    commands = calls(server)
    assert not any(name == "curl" for name, _ in commands)
    install = next(
        args for name, args in commands if name == "python3" and "install" in args
    )
    assert install[install.index("--target") + 1] == str(
        server[0] / ".tools/uv-bootstrap"
    )
    assert any("visual_training.check_gpu" in args for _, args in commands)
    server[2].write_text("")
    result = run(server, "check_dependencies.sh", "--rtx5090")
    assert result.returncode == 0, result.stderr
    assert not any(name in ("python3", "curl") for name, _ in calls(server))


def test_failed_pypi_download_falls_back_to_retried_github(server):
    result = run(server, "check_dependencies.sh", "--rtx5090", MOCK_PIP_FAIL="1")
    assert result.returncode == 0, result.stderr
    curl = next(args for name, args in calls(server) if name == "curl")
    assert "--retry-all-errors" in curl and "--connect-timeout" in curl
    assert (server[0] / ".venv/bin/python").is_file()


@pytest.mark.parametrize(
    "failure",
    [{"MOCK_PIP_FAIL": "1", "MOCK_CURL_FAIL": "1"}, {"MOCK_PYTHON_FAIL": "1"}],
)
def test_failed_setup_never_starts_training_and_records_exit(server, failure):
    log = server[0] / "logs/failed.log"
    result = run(
        server, "train_rtx5090.sh", "--foreground", "--log", str(log), **failure
    )
    assert result.returncode != 0
    assert "TRAINING_STARTED" not in log.read_text()
    assert not any("visual_training.train" in args for _, args in calls(server))
    assert int(Path(str(log) + ".exit_code").read_text()) == result.returncode
    if "MOCK_CURL_FAIL" in failure:
        assert "unexpected eof" in log.read_text()
        assert "python3-pip" in log.read_text()
    else:
        assert "LANDING_PYTHON" in log.read_text()


def test_tmux_job_checks_before_training_preserves_arguments_and_environment(server):
    log = server[0] / "logs/training.log"
    result = run(
        server,
        "train_rtx5090.sh",
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
    check = next(
        i for i, (_, args) in enumerate(commands) if "visual_training.check_gpu" in args
    )
    training = next(
        i for i, (_, args) in enumerate(commands) if "visual_training.train" in args
    )
    assert check < training
    assert commands[training][1][-4:] == [
        "--data-root",
        "/data/landing images",
        "--epochs",
        "1",
    ]


def test_duplicate_tmux_session_is_not_started(server):
    result = run(server, "train_rtx5090.sh", MOCK_SESSION_EXISTS="1")
    assert result.returncode == 2
    assert "Session already exists" in result.stderr
    assert not any(name == "python3" for name, _ in calls(server))
    assert not (server[0] / "outputs/logs").exists()


def test_default_log_names_are_unique_and_persist_after_completion(server):
    for _ in range(2):
        result = run(server, "train_rtx5090.sh")
        assert result.returncode == 0, result.stderr
    logs = list((server[0] / "outputs/logs").glob("rtx5090_*.log"))
    assert len(logs) == 2
    for log in logs:
        assert "TRAINING_STARTED" in log.read_text()
        assert Path(str(log) + ".exit_code").read_text() == "0\n"
