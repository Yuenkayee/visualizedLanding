"""Keep account login and both Engine entry points on the same license mode."""

import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shell import matlab_startup as startup


@pytest.fixture
def preference(tmp_path, monkeypatch):
    path = tmp_path / "tools/matlab-startup.json"
    monkeypatch.setattr(startup, "PREFERENCE", path)
    monkeypatch.delenv("MATLAB_LICENSE_MODE", raising=False)
    monkeypatch.delenv("MATLAB_SHARED_SESSION", raising=False)
    monkeypatch.setenv("MATLAB_ROOT", str(tmp_path / "MATLAB"))
    return path


@pytest.fixture
def engine_api(monkeypatch):
    calls = []
    engine = SimpleNamespace(addpath=lambda *a, **k: None, setup_paths=lambda *a, **k: None,
                             finalize_external_model=lambda **k: None, quit=lambda: None)
    api = SimpleNamespace(start_matlab=lambda options: calls.append(options) or engine)
    monkeypatch.setitem(sys.modules, "matlab", SimpleNamespace(engine=api))
    monkeypatch.setitem(sys.modules, "matlab.engine", api)
    return api, engine, calls


def test_default_keeps_installed_or_network_license_discovery(preference):
    assert startup.license_mode() == "auto"
    assert "-licmode" not in startup.engine_options(startup.license_mode())
    assert startup.license_mode(login=True) == "online"
    assert not preference.exists()


@pytest.mark.parametrize("mode", ["online", "onlinelicensing", "file", "auto"])
def test_explicit_mode_reaches_engine_and_report(preference, engine_api, monkeypatch, mode):
    monkeypatch.setenv("MATLAB_LICENSE_MODE", mode)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    api, engine, calls = engine_api
    report = preference.parent / "log/startup.json"
    assert startup.start_matlab(report_path=report) is engine
    assert "-nodisplay" in calls[0]
    assert (f"-licmode {mode}" in calls[0]) if mode != "auto" else ("-licmode" not in calls[0])
    assert json.loads(report.read_text()) == {"status": "started", "license_mode": mode, "options": calls[0],
                                            "connection": "new", "session_name": None}


def test_macos_keeps_previous_headless_options(preference, monkeypatch):
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    assert startup.engine_options("auto") == "-nodesktop -nosplash"


def test_login_mode_is_remembered_without_secrets_and_can_be_overridden(preference, monkeypatch):
    root = preference.parent / "MATLAB"
    startup.remember_login_mode(root, "online")
    stored = json.loads(preference.read_text())
    assert set(stored) == {"mode", "matlab_root", "uid", "home"}
    assert startup.license_mode(root) == "online"
    monkeypatch.setenv("MATLAB_LICENSE_MODE", "file")
    assert startup.license_mode(root) == "file"


def test_other_install_or_user_does_not_inherit_login_mode(preference):
    root = preference.parent / "MATLAB"
    startup.remember_login_mode(root, "online")
    assert startup.license_mode(root / "other") == "auto"
    saved = json.loads(preference.read_text())
    saved["uid"] += 1
    preference.write_text(json.dumps(saved))
    assert startup.license_mode(root) == "auto"


def test_invalid_mode_fails_before_launch(preference, engine_api, monkeypatch):
    monkeypatch.setenv("MATLAB_LICENSE_MODE", "online -r unexpected")
    with pytest.raises(ValueError, match="MATLAB_LICENSE_MODE"):
        startup.start_matlab()
    assert not engine_api[2]


def test_file_mode_does_not_enter_account_login(preference, monkeypatch):
    from shell import closed_loop_dependencies as deploy
    monkeypatch.setenv("MATLAB_LICENSE_MODE", "file")
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: pytest.fail("unexpected login"))
    with pytest.raises(ValueError, match="--login requires"):
        deploy.matlab_login(preference.parent / "MATLAB")


@pytest.mark.parametrize("success", [True, False])
def test_interactive_login_persists_only_after_success(preference, monkeypatch, success):
    from shell import closed_loop_dependencies as deploy
    monkeypatch.setenv("MATLAB_LICENSE_MODE", "onlinelicensing")
    monkeypatch.setattr(deploy, "open", lambda *a, **k: io.BytesIO(), raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["stdin"] is not None and kwargs["stdout"] is kwargs["stderr"]
        if not success:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(deploy.subprocess, "run", run)
    root = preference.parent / "MATLAB"
    if success:
        deploy.matlab_login(root)
        monkeypatch.delenv("MATLAB_LICENSE_MODE")
        assert startup.license_mode(root) == "onlinelicensing"
    else:
        with pytest.raises(subprocess.CalledProcessError):
            deploy.matlab_login(root)
        assert not preference.exists()
    assert calls[0][1:] == ["-licmode", "onlinelicensing", "-nodisplay"]


def test_failure_report_preserves_underlying_licensing_error(preference, engine_api, monkeypatch):
    monkeypatch.setenv("MATLAB_LICENSE_MODE", "online")

    def fail(options):
        raise RuntimeError("Unable to launch MVM server: License Error: Licensing shutdown")

    engine_api[0].start_matlab = fail
    report = preference.parent / "log/startup.json"
    with pytest.raises(RuntimeError, match="license_mode=online"):
        startup.start_matlab(report_path=report)
    saved = json.loads(report.read_text())
    assert saved["status"] == "failed" and saved["license_mode"] == "online"
    assert saved["error"].endswith("Licensing shutdown")


def test_actual_simulation_session_reuses_saved_login_mode(preference, engine_api, monkeypatch):
    from simulation.bridge.matlab_session import MatlabSession
    root = preference.parent / "MATLAB"
    monkeypatch.setenv("MATLAB_ROOT", str(root))
    startup.remember_login_mode(root, "online")
    with MatlabSession() as session:
        assert session.engine is engine_api[1]
    assert "-licmode online" in engine_api[2][0]


def test_check_uses_shared_starter_and_records_options():
    # The runtime call itself is covered above; prevent an unconfigured second
    # start_matlab call from being reintroduced in the environment checker.
    from main import check_environment
    assert check_environment.start_matlab is startup.start_matlab


@pytest.fixture
def shared_api(preference, engine_api, monkeypatch):
    api, engine, calls = engine_api
    attached = []
    disconnected = []
    monkeypatch.setenv("MATLAB_SHARED_SESSION", "LandingMATLAB")
    monkeypatch.setenv("MATLAB_ROOT", str(preference.parent / "MATLAB"))
    engine.matlabroot = lambda: str(preference.parent / "MATLAB")
    engine.version = lambda flag: "2024b"
    engine.quit = lambda: disconnected.append(True)
    api.find_matlab = lambda: ("LandingMATLAB",)
    api.connect_matlab = lambda name: attached.append(name) or engine
    return api, engine, calls, attached, disconnected


def test_shared_session_attaches_by_name_and_reports_existing_authorization(preference, shared_api):
    api, engine, launched, attached, disconnected = shared_api
    report = preference.parent / "shared.json"
    assert startup.start_matlab(report_path=report) is engine
    assert attached == ["LandingMATLAB"] and not launched and not disconnected
    saved = json.loads(report.read_text())
    assert saved["status"] == "connected" and saved["connection"] == "shared"
    assert saved["options"] is None and saved["license_mode"] == "existing_session"
    assert saved["release"] == "2024b"


def test_missing_shared_session_never_falls_back_to_new_process(preference, shared_api):
    api, _, launched, attached, _ = shared_api
    api.find_matlab = lambda: ("AnotherMATLAB",)
    report = preference.parent / "shared.json"
    with pytest.raises(RuntimeError, match="not found"):
        startup.start_matlab(report_path=report)
    assert not launched and not attached
    saved = json.loads(report.read_text())
    assert saved["available_sessions"] == ["AnotherMATLAB"] and saved["status"] == "failed"


@pytest.mark.parametrize("mismatch", ["release", "root"])
def test_shared_session_rejects_wrong_installation(preference, shared_api, mismatch):
    _, engine, launched, _, disconnected = shared_api
    if mismatch == "release":
        engine.version = lambda flag: "2025b"
    else:
        engine.matlabroot = lambda: "/another/MATLAB"
    with pytest.raises(RuntimeError, match="Shared MATLAB"):
        startup.start_matlab()
    assert disconnected == [True] and not launched


def test_shared_attach_failure_is_recorded_without_new_start(preference, shared_api):
    api, _, launched, _, _ = shared_api

    def fail(name):
        raise RuntimeError("Connection closed")

    api.connect_matlab = fail
    report = preference.parent / "shared.json"
    with pytest.raises(RuntimeError, match="Connection closed"):
        startup.start_matlab(report_path=report)
    assert json.loads(report.read_text())["status"] == "failed" and not launched


def test_actual_simulation_uses_shared_session_and_disconnects(shared_api):
    from simulation.bridge.matlab_session import MatlabSession
    _, engine, launched, attached, disconnected = shared_api
    with MatlabSession() as session:
        assert session.engine is engine
    assert attached == ["LandingMATLAB"] and disconnected == [True] and not launched


def test_invalid_shared_name_fails_before_engine_call(preference, engine_api, monkeypatch):
    monkeypatch.setenv("MATLAB_SHARED_SESSION", "bad name")
    with pytest.raises(ValueError, match="MATLAB_SHARED_SESSION"):
        startup.start_matlab()
    assert not engine_api[2]
