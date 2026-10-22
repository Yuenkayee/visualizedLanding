"""Use the same explicit licensing mode for login, checks and simulation.

Only a mode selection is persisted; credentials remain managed by MathWorks.
Without a selection, keep MATLAB's default license discovery (auto).
"""

import json
import os
from pathlib import Path
import platform
import re

PREFERENCE = Path(__file__).resolve().parents[1] / ".tools/closed-loop/matlab-startup.json"
MODES = {"auto", "online", "onlinelicensing", "file"}


def shared_session_name():
    name = os.environ.get("MATLAB_SHARED_SESSION", "").strip()
    if name and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", name):
        raise ValueError("MATLAB_SHARED_SESSION must be a MATLAB identifier (letters, digits, underscore; max 63 characters)")
    return name


def identity(root):
    return {"matlab_root": str(Path(root).expanduser().resolve()),
            "home": str(Path.home().resolve()), "uid": os.getuid()}


def license_mode(root=None, login=False):
    mode = os.environ.get("MATLAB_LICENSE_MODE", "").strip()
    root = root or os.environ.get("MATLAB_ROOT")
    if not mode and root and PREFERENCE.is_file():
        saved = json.loads(PREFERENCE.read_text())
        if all(saved.get(key) == value for key, value in identity(root).items()):
            mode = saved.get("mode", "")
    mode = mode or ("online" if login else "auto")
    if mode not in MODES:
        raise ValueError("MATLAB_LICENSE_MODE must be auto, online, onlinelicensing or file")
    if login and mode not in ("online", "onlinelicensing"):
        raise ValueError("--login requires MATLAB_LICENSE_MODE=online or onlinelicensing; file/auto licenses do not use account login")
    return mode


def remember_login_mode(root, mode):
    if mode not in ("online", "onlinelicensing"):
        raise ValueError("Only account login modes can be remembered")
    PREFERENCE.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCE.write_text(json.dumps({**identity(root), "mode": mode}, indent=2) + "\n")


def engine_options(mode):
    if mode not in MODES:
        raise ValueError(f"Unsupported MATLAB license mode: {mode}")
    options = ["-nodesktop", "-nosplash"]
    if platform.system() == "Linux":
        options.append("-nodisplay")
    if mode != "auto":
        options.extend(["-licmode", mode])
    return " ".join(options)


def start_matlab(root=None, report_path=None):
    import matlab.engine

    mode = license_mode(root)
    name = shared_session_name()
    options = None if name else engine_options(mode)
    report = {"status": "connecting" if name else "starting",
              "connection": "shared" if name else "new",
              "license_mode": "existing_session" if name else mode,
              "options": options, "session_name": name or None}

    def write_report():
        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")

    write_report()
    if name:
        print(f"[matlab] Connecting to shared session: {name} (existing MATLAB authorization)", flush=True)
        engine = None
        try:
            available = list(matlab.engine.find_matlab())
            report["available_sessions"] = available
            if name not in available:
                raise RuntimeError(f"Shared MATLAB session {name!r} not found; available: {available}. "
                                   "Keep the named MATLAB session running as the same user on this host/container")
            # Always pass the name: connect_matlab() without one can silently
            # launch a new process, returning to the failing startup path.
            engine = matlab.engine.connect_matlab(name)
            actual_root = str(engine.matlabroot())
            release = str(engine.version("-release"))
            report.update(matlab_root=actual_root, release=release)
            if release != "2024b":
                raise RuntimeError(f"Shared MATLAB must be R2024b, found {release}")
            expected_root = root or os.environ.get("MATLAB_ROOT")
            if expected_root and Path(actual_root).resolve() != Path(expected_root).expanduser().resolve():
                raise RuntimeError(f"Shared MATLAB root {actual_root} differs from MATLAB_ROOT={expected_root}")
        except Exception as exc:
            # For a connect_matlab attachment quit() disconnects this client;
            # it does not terminate the independently launched shared MATLAB.
            if engine is not None:
                try:
                    engine.quit()
                except Exception as cleanup:
                    report["disconnect_error"] = str(cleanup)
            report.update(status="failed", error=str(exc))
            write_report()
            raise RuntimeError(f"MATLAB shared-session connection failed: {exc}. "
                               "No new MATLAB process was started. See main/README.md for shared-session setup") from exc
        report["status"] = "connected"
        write_report()
        return engine
    print(f"[matlab] Starting Engine: {options} (license_mode={mode})", flush=True)
    try:
        engine = matlab.engine.start_matlab(options)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        write_report()
        raise RuntimeError(
            f"MATLAB Engine startup failed (license_mode={mode}): {exc}. "
            "For account licensing, select MATLAB_LICENSE_MODE=online (or onlinelicensing for your license) "
            "and run bash shell/run_landing_closed_loop.sh check --login in an interactive terminal as the same user. "
            "If interactive MATLAB also fails, inspect its licensing error and MathWorks Service Host/connectivity; "
            "this error alone does not identify an expired license."
        ) from exc
    report["status"] = "started"
    write_report()
    return engine
