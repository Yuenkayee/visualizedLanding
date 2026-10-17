"""Check/install Python dependencies; check licensed MATLAB/Simulink separately.

MATLAB/Simulink and a license must already exist. MATLAB Runtime is insufficient
for editing a model or controlling a live Simulink simulation through Engine.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
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


def matlab_root():
    candidates = []
    if os.environ.get("MATLAB_ROOT"):
        candidates.append(Path(os.environ["MATLAB_ROOT"]))
    binary = shutil.which("matlab")
    if binary:
        candidates.append(Path(binary).resolve().parent.parent)
    candidates += [
        Path("/Applications/MATLAB_R2024b.app"),
        Path("/usr/local/MATLAB/R2024b"),
        Path("/opt/MATLAB/R2024b"),
    ]
    for p in candidates:
        if (p / "bin/matlab").exists():
            return p
    raise RuntimeError(
        "MATLAB R2024b not found. Install licensed MATLAB + Simulink and set MATLAB_ROOT; pip and MATLAB Runtime cannot supply Simulink. Local macOS is supported."
    )


def renderer(check, python=None):
    requested = python
    if python is None:
        candidate = ROOT / ".venv-render-server/bin/python"
        python = (
            str(candidate)
            if platform.system() == "Linux" and candidate.exists()
            else sys.executable
        )
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                "import bpy,cv2,numpy,scipy,yaml;print(bpy.app.version_string)",
            ],
            capture_output=True,
            text=True,
        )
        missing = bool(result.returncode)
    except FileNotFoundError:
        missing = True
    if missing:
        if check:
            raise RuntimeError(
                f"Renderer missing dependencies: {python}; rerun without --check"
            )
        if platform.system() == "Linux":
            ensure_libraries(False)
            candidate = ROOT / ".venv-render-server/bin/python"
            if requested and Path(requested).resolve() != candidate.resolve():
                run(
                    [
                        requested,
                        "-c",
                        'import sys; assert sys.version_info[:2]==(3,11), "RENDER_PYTHON requires Python 3.11"',
                    ]
                )
                candidate = Path(requested)
            if not candidate.exists():
                run([sys.executable, "-m", "pip", "install", "uv"])
                run(
                    [
                        sys.executable,
                        "-m",
                        "uv",
                        "venv",
                        "--python",
                        "3.11",
                        "--managed-python",
                        "--seed",
                        candidate.parent.parent,
                    ]
                )
            python = str(candidate)
            ensure_requirements(python, ROOT / "requirements-render-server.txt", False)
        else:
            if sys.version_info[:2] != (3, 11):
                raise RuntimeError(
                    "Local bpy profile requires Python 3.11; use .venv/bin/python or set RENDER_PYTHON"
                )
            ensure_requirements(python, ROOT / "requirements-data.txt", False)
            run([python, "-m", "pip", "install", "bpy==4.2.0"])
    if platform.system() == "Linux":
        ensure_libraries(check)
    run(
        [
            python,
            "-c",
            'import bpy,cv2,numpy,scipy,yaml;print("Renderer:",bpy.app.version_string)',
        ]
    )
    return python


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Do not install anything")
    p.add_argument("--config", default="main/closed_loop.yaml")
    p.add_argument("--renderer", choices=["cpu", "blender"])
    p.add_argument("--render-python")
    p.add_argument("--report")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"])
    p.add_argument("--render-device", choices=["CPU", "CUDA", "OPTIX"])
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
    os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ["PATH"]
    os.environ["MATLAB_ROOT"] = str(root)
    if (
        platform.system() == "Linux"
        and platform.machine() == "x86_64"
        and sys.version_info[:2] == (3, 12)
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
    render_python = (
        renderer(
            a.check,
            a.render_python
            or os.environ.get("RENDER_PYTHON")
            or config.get("render_python"),
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
        )
    engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    try:
        features = sorted(set(["Simulink", *config.get("required_toolboxes", [])]))
        licenses = {name: bool(engine.license("test", name)) for name in features}
        if not all(licenses.values()):
            raise RuntimeError(f"Required toolbox license missing: {licenses}")
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
            matlab_platform=engine.computer(),
            licenses=licenses,
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            device=device,
            renderer=selected,
            render_python=render_python,
            render_device=render_device,
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
