"""Native-library isolation for Blender children of the MATLAB coordinator.

Use this environment at process creation: changing os.environ after a native
library has been loaded cannot change the loader's existing link map.
"""

import json
import os
from pathlib import Path
import platform
import re
import subprocess


def renderer_environment(environ=None):
    env = dict(os.environ if environ is None else environ)
    if platform.system() != "Linux":
        return env
    root_value = env.get("MATLAB_ROOT")
    root = Path(root_value).expanduser().resolve() if root_value else None

    def matlab_path(value):
        path = Path(value).expanduser().resolve()
        if root is not None and (path == root or root in path.parents):
            return True
        # Also recognize a different MATLAB installation or its symlinked paths.
        return any((parent / "VersionInfo.xml").is_file() and
                   (parent / "bin/matlab").is_file() for parent in (path, *path.parents))

    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        if name not in env:
            continue
        entries = re.split(r"[\s:]+", env[name]) if name == "LD_PRELOAD" else env[name].split(os.pathsep)
        kept = [entry for entry in entries if entry and not matlab_path(entry)]
        if kept:
            env[name] = os.pathsep.join(kept)
        else:
            env.pop(name, None)
    # Preserve CUDA/NVIDIA paths, user libraries and the system libstdc++ preload
    # used for Conda compatibility. The parent's environment is never mutated.
    return env


_PROBE = r'''
import importlib.util, json, os, platform, subprocess, sys, traceback
from pathlib import Path
report = {"python": sys.executable, "platform": platform.system(),
          "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
          "LD_PRELOAD": os.environ.get("LD_PRELOAD", "")}
try:
    import bpy, cv2, numpy, scipy, yaml
    report.update(status="ready", blender=bpy.app.version_string)
except Exception as exc:
    report.update(status="failed", error=str(exc))
    traceback.print_exc()
    if platform.system() == "Linux":
        spec = importlib.util.find_spec("bpy")
        if spec and spec.origin:
            usd = Path(spec.origin).parent / "lib/libusd_ms.so"
            if usd.is_file():
                try:
                    result = subprocess.run(["ldd", str(usd)], capture_output=True, text=True, timeout=20)
                    report["usd_ldd"] = result.stdout + result.stderr
                except (OSError, subprocess.TimeoutExpired) as diagnostic:
                    report["ldd_error"] = str(diagnostic)
finally:
    maps = Path("/proc/self/maps")
    if maps.is_file():
        report["loaded_libraries"] = sorted({line.split()[-1] for line in maps.read_text().splitlines()
            if any(name in line for name in ("libvulkan", "libstdc++", "libusd", "libGL", "libEGL"))})
    print("LANDING_RENDERER_PROBE=" + json.dumps(report), flush=True)
sys.exit(0 if report.get("status") == "ready" else 1)
'''


def probe_renderer(python, report_path=None):
    result = subprocess.run([str(python), "-c", _PROBE], env=renderer_environment(),
                            capture_output=True, text=True)
    prefix = "LANDING_RENDERER_PROBE="
    reports = [line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)]
    report = json.loads(reports[-1]) if reports else {
        "status": "failed", "python": str(python), "error": "Renderer exited before writing its import report"}
    report["returncode"] = result.returncode
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    if result.returncode or report["status"] != "ready":
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        location = f"; diagnostic report: {report_path}" if report_path else ""
        raise RuntimeError(f"Blender import failed in isolated renderer environment: {report.get('error', '')}{location}")
    print(f"[environment] Renderer: {report['blender']} (isolated library paths)", flush=True)
    return report
