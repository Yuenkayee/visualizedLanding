"""Verify Python packages, MATLAB licenses, checkpoint, CUDA and Cycles.

Use shell/run_landing_closed_loop.sh check for the complete Ubuntu installation,
including the OS libraries, Python runtimes and MATLAB products.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from shell.server_dependencies import (
    ROOT,
    ensure_requirements,
    ensure_torch,
    ensure_libraries,
    run,
)
from shell.closed_loop_dependencies import (
    absolute_executable, ensure_venv, matlab_destination, matlab_engine_environment,
    python_matches,
)
from shell.render_environment import probe_renderer, renderer_environment
from shell.matlab_startup import license_mode, start_matlab


def matlab_root():
    root = matlab_destination()
    if (root / "bin/matlab").is_file():
        return root
    raise RuntimeError(
        "MATLAB R2024b not found. Run bash shell/run_landing_closed_loop.sh check to install on Ubuntu."
    )


def renderer(check, python=None, report_path=None):
    if platform.system() == "Linux":
        ensure_libraries(check)
        default = ROOT / ".venv-render-server/bin/python"
        if python and absolute_executable(python) != str(default):
            python = absolute_executable(python)
            if not python_matches(python, ["3.11"]):
                raise RuntimeError("RENDER_PYTHON/--render-python must be an existing Python 3.11 interpreter")
        else:
            python = ensure_venv(default.parent.parent, "3.11", check)
        ensure_requirements(python, ROOT / "requirements-render-server.txt", check)
        probe_renderer(python, report_path)
        return python
    if python is None:
        python = sys.executable
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                "import bpy,cv2,numpy,scipy,yaml;print(bpy.app.version_string)",
            ],
            capture_output=True,
            text=True,
            env=renderer_environment(),
        )
        missing = bool(result.returncode)
    except FileNotFoundError:
        missing = True
    if missing:
        if check:
            raise RuntimeError(
                f"Renderer missing dependencies: {python}; rerun without --check"
            )
        if not python_matches(python, ["3.11"]):
            raise RuntimeError(
                "Local bpy profile requires Python 3.11; use .venv/bin/python or set RENDER_PYTHON"
            )
        ensure_requirements(python, ROOT / "requirements-data.txt", False)
        run([python, "-m", "pip", "install", "bpy==4.2.0"])
    probe_renderer(python, report_path)
    return python


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Do not install anything")
    p.add_argument("--config", default="main/navigation_observer.yaml")
    p.add_argument("--renderer", choices=["cpu", "blender"])
    p.add_argument("--render-python")
    p.add_argument("--report")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"])
    p.add_argument("--render-device", choices=["CPU", "CUDA", "OPTIX"])
    p.add_argument("--checkpoint")
    a = p.parse_args()
    if platform.system() not in ("Darwin", "Linux"):
        raise RuntimeError("Supported hosts: macOS and Ubuntu 22.04")
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise RuntimeError(
            "Use local Python 3.11 or server Python 3.12 for this repository"
        )
    root = matlab_root()
    release = ET.parse(root / "VersionInfo.xml").findtext("release")
    if release != "R2024b":
        raise RuntimeError(
            f"This dependency profile is tested against R2024b; found {release}"
        )
    # Also support direct python -m main.check_environment calls: the isolated
    # Engine build inherits this environment. The wrapper sets it at startup.
    os.environ.update(matlab_engine_environment(root))
    if (
        platform.system() == "Linux"
        and platform.machine() == "x86_64"
    ):
        ensure_torch(a.check)
    ensure_requirements(sys.executable, ROOT / "requirements-closed-loop.txt", a.check)
    ensure_requirements(sys.executable, ROOT / "requirements-matlab.txt", a.check)
    import yaml
    import torch
    import numpy, scipy, cv2, matplotlib  # check native imports, not only metadata
    import matlab.engine

    config = yaml.safe_load((ROOT / a.config).read_text())
    selected = a.renderer or config["renderer"]
    device = a.device or config["device"]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable to PyTorch; check NVIDIA driver/container GPU access"
            )
        assert (torch.ones(8, device="cuda") @ torch.ones(8, device="cuda")).item() == 8
    elif device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable to PyTorch")
    checkpoint = a.checkpoint or config.get("checkpoint")
    checkpoint_info = None
    if checkpoint:
        checkpoint = (ROOT / Path(checkpoint).expanduser()).absolute()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint}; restore tracked four_weather_v1/best.pt or pass --checkpoint")
        from navigation.vision.infer_h import HDetector
        detector = HDetector(checkpoint, device)
        del detector
        checkpoint_info = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                               sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    render_python = (
        renderer(
            a.check,
            a.render_python
            or os.environ.get("RENDER_PYTHON")
            or config.get("render_python"),
            Path(a.report).with_name("renderer-environment.json") if a.report else None,
        )
        if selected == "blender"
        else None
    )
    render_device = a.render_device or config["render_device"]
    if render_python:
        run(
            [
                render_python,
                "-m",
                "visual_modeling.blender.cycles_device",
                "--device",
                render_device,
                "--threads",
                str(config.get("render_threads", 8)),
            ],
            cwd=ROOT,
            env=renderer_environment(),
        )
    print("[environment] Starting MATLAB and checking toolbox licenses", flush=True)
    engine = start_matlab(root, Path(a.report).with_name("matlab-startup.json") if a.report else None)
    try:
        features = sorted(set(["Simulink", *config.get("required_toolboxes", [])]))
        licenses = {name: bool(engine.license("checkout", name)) for name in features}
        if not all(licenses.values()):
            raise RuntimeError(f"Required toolbox license unavailable: {licenses}; authorize with check --login or configure your license server")
        # License testing alone is not enough: load the actual installed block library.
        engine.load_system("simulink", nargout=0)
        for feature, function in [
            ("Control_Toolbox", "ss"),
            ("Robust_Toolbox", "hinfsyn"),
        ]:
            if feature in features and not engine.which(function):
                raise RuntimeError(
                    f"{feature} is licensed but not installed; install it with MATLAB products"
                )
        report = dict(
            status="ready",
            os=platform.platform(),
            python=sys.version,
            matlab=engine.version(),
            matlab_root=str(root),
            matlab_license_mode=license_mode(root),
            matlab_platform=engine.computer(),
            licenses=licenses,
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            device=device,
            renderer=selected,
            render_python=render_python,
            render_device=render_device,
            checkpoint=checkpoint_info,
            note="Environment check does not certify the supplied aircraft model or its toolbox dependencies.",
        )
    finally:
        engine.quit()
    if a.report:
        dest = Path(a.report)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
