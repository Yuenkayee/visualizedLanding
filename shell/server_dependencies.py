"""Bootstrap the Ubuntu/Python 3.12 coordinator and isolated cp311 bpy runtime.

Uses only the standard library until pip's requirement parser is available.
Never installs the repository's Python 3.11 / Torch 2.5 / cu124 lock files.
"""

import argparse
import ctypes.util
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

if __package__:
    from .closed_loop_dependencies import apt_command
else:
    from closed_loop_dependencies import apt_command

ROOT = Path(__file__).resolve().parents[1]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
RENDER_PYTHON = ROOT / ".venv-render-server/bin/python"


def run(command, **kwargs):
    print("[environment] " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def requirements(path):
    """Read our pinned profiles, including shared -r files."""
    from pip._vendor.packaging.requirements import Requirement

    path = Path(path)
    result = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            result.extend(requirements(path.parent / line[3:].strip()))
        elif line:
            result.append(Requirement(line))
    return result


def audit(path):
    """Check versions and the declared transitive dependency closure only.

    An unrelated preinstalled notebook package must not force a torch downgrade.
    No module imports here: broken binary imports are checked separately.
    """
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name

    pending = requirements(path)
    visited = set()
    problems = []
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            dist = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            problems.append(f"missing {requirement}")
            continue
        if dist.version not in requirement.specifier:
            problems.append(f"need {requirement}, found {dist.version}")
        key = canonicalize_name(requirement.name)
        if key not in visited:
            visited.add(key)
            pending.extend(Requirement(r) for r in dist.requires or [])
    return sorted(set(problems))


def ensure_requirements(python, profile, check):
    def inspect():
        result = subprocess.run(
            [str(python), str(Path(__file__).resolve()), "--audit", str(profile)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    problems = inspect()
    if problems:
        print("\n".join(problems), flush=True)
        if check:
            raise RuntimeError(
                f"Dependency check failed: {profile.name}; rerun without --check"
            )
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--upgrade-strategy",
                "only-if-needed",
                "-r",
                profile,
            ]
        )
        problems = inspect()
        if problems:
            raise RuntimeError(
                "Unresolved required dependencies: " + "; ".join(problems)
            )
    print(f"[environment] {profile.name}: OK", flush=True)


def ensure_torch(check):
    # A 2.8.0 CPU/cu126 wheel also satisfies ==2.8.0: inspect CUDA separately.
    code = "import torch; print(torch.__version__.split('+')[0], torch.version.cuda)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    if result.returncode or result.stdout.strip() != "2.8.0 12.8":
        if check:
            raise RuntimeError("Need an importable PyTorch 2.8.0 CUDA 12.8 build")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "torch==2.8.0+cu128",
                "--index-url",
                os.environ.get("LANDING_TORCH_INDEX") or TORCH_INDEX,
            ]
        )


def ensure_libraries(check):
    # bpy loads Blender's native libraries even in a headless process. No X
    # server/Xvfb is required. NVIDIA driver libraries come from the host.
    packages = {
        "X11": "libx11-6",
        "Xi": "libxi6",
        "Xxf86vm": "libxxf86vm1",
        "Xfixes": "libxfixes3",
        "Xrender": "libxrender1",
        "Xext": "libxext6",
        "SM": "libsm6",
        "ICE": "libice6",
        "GL": "libgl1",
        "EGL": "libegl1",
        "xkbcommon": "libxkbcommon0",
        "gomp": "libgomp1",
    }
    missing = [
        package
        for lib, package in packages.items()
        if not ctypes.util.find_library(lib)
    ]
    if not missing:
        return
    arguments = ["install", "-y", "--no-install-recommends", *missing]
    if check:
        raise RuntimeError("Missing system libraries; run: sudo apt-get " + " ".join(arguments))
    prefix = []
    if os.geteuid() != 0:
        if (
            not shutil.which("sudo")
            or subprocess.run(["sudo", "-n", "true"]).returncode
        ):
            raise RuntimeError(
                "Install the missing libraries then rerun: sudo apt-get " + " ".join(arguments)
            )
        prefix = ["sudo", "-n"]
    run([*prefix, *apt_command("update")])
    run([*prefix, *apt_command(*arguments)], env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})


def setup_training(check=False, need_tmux=False):
    """Training needs neither bpy nor a second Python runtime."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Training server profile requires Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the image's Python 3.12 via PYTHON_BIN")
    run([sys.executable, "-m", "pip", "--version"])
    ensure_torch(check)
    ensure_requirements(
        sys.executable, ROOT / "requirements-training-server.txt", check
    )
    run(
        [
            sys.executable,
            "-c",
            "import torch, cv2, numpy, scipy, yaml; "
            "assert torch.__version__.split('+')[0] == '2.8.0' and torch.version.cuda == '12.8'; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "x=torch.ones(16, device='cuda'); assert (x @ x).item() == 16; "
            "assert torch.cuda.is_bf16_supported(), 'BF16 unsupported on selected GPU'; "
            "print('Training GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__)",
        ]
    )
    if need_tmux and not shutil.which("tmux"):
        arguments = ["install", "-y", "--no-install-recommends", "tmux"]
        prefix = []
        if check:
            raise RuntimeError("Missing tmux; run: sudo apt-get " + " ".join(arguments))
        if os.geteuid() != 0:
            if (
                not shutil.which("sudo")
                or subprocess.run(["sudo", "-n", "true"]).returncode
            ):
                raise RuntimeError("Install tmux then rerun: sudo apt-get " + " ".join(arguments))
            prefix = ["sudo", "-n"]
        run([*prefix, *apt_command("update")])
        run(
            [*prefix, *apt_command(*arguments)], env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
        )
    if need_tmux:
        run(["tmux", "-V"])


def setup(check=False, device="OPTIX", gpu_index=0, threads=8):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Server profile requires Linux x86_64 (Ubuntu 22.04)")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Use the image's Python 3.12 via PYTHON_BIN; got {sys.version}"
        )
    run([sys.executable, "-m", "pip", "--version"])
    ensure_torch(check)
    ensure_requirements(sys.executable, ROOT / "requirements-server.txt", check)
    ensure_libraries(check)
    if not RENDER_PYTHON.exists():
        if check:
            raise RuntimeError("Missing .venv-render-server; rerun without --check")
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
                RENDER_PYTHON.parent.parent,
            ]
        )
    run([RENDER_PYTHON, "-c", "import sys; assert sys.version_info[:2] == (3, 11)"])
    ensure_requirements(RENDER_PYTHON, ROOT / "requirements-render-server.txt", check)
    run(
        [
            sys.executable,
            "-c",
            "import cv2, numpy, scipy, yaml, PIL, matplotlib, onnx, onnxruntime, torch; "
            "assert torch.__version__.split('+')[0] == '2.8.0' and torch.version.cuda == '12.8'; "
            "assert torch.cuda.is_available(), 'CUDA unavailable: check host driver and container GPU access'; "
            "x=torch.ones(16, device='cuda'); assert (x @ x).item() == 16; "
            "print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, "
            "'GPU:', torch.cuda.get_device_name(0))",
        ]
    )
    run(
        [
            RENDER_PYTHON,
            "-c",
            "import bpy, cv2, numpy, scipy, yaml; print('bpy:', bpy.app.version_string)",
        ]
    )
    run(
        [
            RENDER_PYTHON,
            "-m",
            "visual_modeling.blender.cycles_device",
            "--device",
            device,
            "--gpu-index",
            gpu_index,
            "--threads",
            threads,
        ],
        cwd=ROOT,
    )
    print(
        "[environment] Dependencies, CUDA tensor operation and Cycles render: OK",
        flush=True,
    )
    return str(RENDER_PYTHON)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="check only; do not install"
    )
    parser.add_argument("--device", choices=["OPTIX", "CUDA"], default="OPTIX")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--training",
        action="store_true",
        help="check/install training dependencies only",
    )
    parser.add_argument(
        "--with-tmux",
        action="store_true",
        help="also check/install tmux (training mode)",
    )
    parser.add_argument("--audit", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(args.audit)))
        return
    if args.threads < 1 or args.gpu_index < 0:
        parser.error("--threads must be positive and --gpu-index must be nonnegative")
    try:
        if args.training:
            setup_training(args.check, args.with_tmux)
        else:
            setup(args.check, args.device, args.gpu_index, args.threads)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Environment check failed: {exc}\n")


if __name__ == "__main__":
    main()
