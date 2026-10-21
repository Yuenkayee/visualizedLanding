"""Deployment contracts; never install system packages or MATLAB in tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shell import closed_loop_dependencies as deploy


def test_read_only_apt_reports_missing_without_mutation(monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr(deploy, "run", lambda *a, **k: pytest.fail("read-only check mutated system"))
    with pytest.raises(RuntimeError, match="libssl-dev"):
        deploy.ensure_apt(["libssl-dev"], check=True)


def test_installed_apt_packages_do_not_need_root(monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="install ok installed"))
    monkeypatch.setattr(deploy, "privilege", lambda: pytest.fail("unnecessary sudo"))
    deploy.ensure_apt(["curl"])


def test_read_only_missing_python_has_no_download(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "ensure_uv", lambda: pytest.fail("download attempted"))
    with pytest.raises(RuntimeError, match="rerun without --check"):
        deploy.ensure_venv(tmp_path / ".venv", "3.12", check=True)
    assert not (tmp_path / ".venv").exists()


def test_explicit_bad_python_never_falls_back(monkeypatch):
    monkeypatch.setenv("PYTHON_BIN", "/missing/python")
    monkeypatch.setattr(deploy, "ensure_venv", lambda *a, **k: pytest.fail("explicit choice ignored"))
    with pytest.raises(RuntimeError, match="unset PYTHON_BIN"):
        deploy.host_python()


def test_venv_symlink_is_preserved(tmp_path):
    binary = tmp_path / "venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(sys.executable)
    assert deploy.absolute_executable(binary) == str(binary)


def test_github_failure_builds_from_python_org(tmp_path, monkeypatch):
    folder = tmp_path / ".venv"
    folder.mkdir()
    (folder / "keep.txt").write_text("previous incomplete environment")
    calls = []
    ready = False

    def matches(binary, versions):
        return ready and str(binary) == str(folder / "bin/python")

    def run(command, **kwargs):
        nonlocal ready
        calls.append(list(map(str, command)))
        if command[0] == "uv-test":
            raise subprocess.CalledProcessError(1, command)
        ready = True

    monkeypatch.setattr(deploy, "python_matches", matches)
    monkeypatch.setattr(deploy, "ensure_uv", lambda: ["uv-test"])
    monkeypatch.setattr(deploy, "source_python", lambda version: Path("/source/python3.12"))
    monkeypatch.setattr(deploy, "run", run)
    assert deploy.ensure_venv(folder, "3.12") == str(folder / "bin/python")
    assert calls[-1] == ["/source/python3.12", "-m", "venv", str(folder)]
    assert len(list(tmp_path.glob(".venv.backup.*/keep.txt"))) == 1


def test_existing_python_seed_failure_does_not_trigger_source_build(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "python_matches", lambda p, v: str(p) == "python3.12")
    monkeypatch.setattr(deploy, "ensure_uv", lambda: ["uv-test"])
    monkeypatch.setattr(deploy, "source_python", lambda v: pytest.fail("unnecessary build"))

    def fail(command, **kwargs):
        assert "--managed-python" not in command
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(deploy, "run", fail)
    with pytest.raises(RuntimeError, match="pip/index connectivity"):
        deploy.ensure_venv(tmp_path / "env", "3.12")


def test_matlab_install_preserves_update_and_passes_distinct_products(tmp_path, monkeypatch):
    root = tmp_path / "MATLAB"
    root.mkdir()
    (root / "VersionInfo.xml").write_text("<root><release>R2024b</release><description>Update 5</description></root>")
    monkeypatch.setattr(deploy, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(deploy.platform, "system", lambda: "Linux")
    calls = []

    def download(url, target):
        assert url == "https://www.mathworks.com/mpm/glnxa64/mpm"
        target.parent.mkdir(parents=True)
        target.write_text("installer")

    def run(command, **kwargs):
        calls.append(list(map(str, command)))
        for marker in deploy.PRODUCTS.values():
            file = root / marker
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("installed")

    monkeypatch.setattr(deploy, "download", download)
    monkeypatch.setattr(deploy, "run", run)
    deploy.ensure_matlab(root)
    assert "--release=R2024bU5" in calls[0]
    assert calls[0][-4:] == ["--products=MATLAB", "Simulink", "Control_System_Toolbox", "Robust_Control_Toolbox"]
    deploy.ensure_matlab(root)
    assert len(calls) == 1


def test_matlab_read_only_and_wrong_release_do_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "download", lambda *a: pytest.fail("unexpected download"))
    with pytest.raises(RuntimeError, match="Missing MATLAB"):
        deploy.ensure_matlab(tmp_path, check=True)
    (tmp_path / "VersionInfo.xml").write_text("<root><release>R2025a</release></root>")
    with pytest.raises(RuntimeError, match="must point to R2024b"):
        deploy.ensure_matlab(tmp_path)


def test_explicit_matlab_destination_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("MATLAB_ROOT", str(tmp_path / "custom"))
    assert deploy.matlab_destination() == tmp_path / "custom"


@pytest.mark.parametrize("fail_at", ["bootstrap", "environment", None])
def test_launcher_stops_after_failure_and_forwards_environment(tmp_path, fail_at):
    """Execute the real shell wrapper with harmless bootstrap/worker stubs."""
    shell = tmp_path / "shell"
    shell.mkdir()
    script = shell / "run_landing_closed_loop.sh"
    script.write_text((deploy.ROOT / "shell/run_landing_closed_loop.sh").read_text())
    (shell / "closed_loop_dependencies.py").write_text('''
import argparse, os, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--report-dir'); a=p.parse_args()
Path('bootstrap_called').touch()
if os.environ.get('TEST_FAIL') == 'bootstrap': sys.exit(17)
root=Path.cwd(); report=Path(a.report_dir)
(report/'python.txt').write_text(sys.executable+'\\n')
(report/'matlab-root.txt').write_text(str(root/'MATLAB')+'\\n')
''')
    main = tmp_path / "main"
    main.mkdir()
    (main / "__init__.py").touch()
    (main / "check_environment.py").write_text('''
import os,sys,json
from pathlib import Path
Path('environment_called').write_text(json.dumps(sys.argv))
assert os.environ['MATLAB_ROOT'] == str(Path.cwd()/'MATLAB')
if os.environ.get('TEST_FAIL') == 'environment': sys.exit(19)
''')
    (main / "run.py").write_text("from pathlib import Path; Path('run_called').touch()\n")
    env = {**os.environ, "PYTHON_BIN": sys.executable, "TEST_FAIL": fail_at or ""}
    env.pop("MATLAB_ROOT", None)
    result = subprocess.run(["bash", str(script), "run", "--duration", "0.2", "--device=cuda",
                             "--checkpoint", "/weights/best.pt"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == {"bootstrap": 17, "environment": 19, None: 0}[fail_at], result.stderr + result.stdout
    assert (tmp_path / "run_called").exists() == (fail_at is None)
    if fail_at != "bootstrap":
        args = json.loads((tmp_path / "environment_called").read_text())
        assert "--duration" not in args
        assert "--device=cuda" in args
        assert "/weights/best.pt" in args
        assert "main/navigation_observer.yaml" in args
    assert len(list(tmp_path.glob("outputs/logs/closed_loop/*/console.log"))) == 1


def test_shipped_checkpoint_is_real_and_ignored_weights_stay_ignored():
    import hashlib
    checkpoint = deploy.ROOT / "visual_training/checkpoints/four_weather_v1/best.pt"
    assert checkpoint.stat().st_size == 1977791
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "1605b4a258414d931f08c630dc1bd83c3946f1a91bfa595614ee0e54df186ab3"
    for path, ignored in [(checkpoint, False), (checkpoint.with_name("last.pt"), True),
                          (checkpoint.parent / "metrics/analysis.md", True)]:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", str(path)], cwd=deploy.ROOT)
        assert (result.returncode == 0) == ignored
