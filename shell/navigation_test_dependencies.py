"""Navigation-only bootstrap and preflight; no MATLAB installation or startup.

The entry point uses stdlib until the selected interpreter's pinned requirements
are ready. --check-only permits reports/probes but never installs dependencies.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shell import closed_loop_dependencies as bootstrap
from shell import server_dependencies as dependencies
from shell.render_environment import probe_renderer, renderer_environment

PROFILE = ROOT / "requirements-navigation-test.txt"
RENDER_PROFILE = ROOT / "requirements-render-server.txt"


def project_path(value):
    return (ROOT / Path(value).expanduser()).absolute()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(filename):
    """Resolve the benchmark's base_config without invoking MATLAB validation."""
    import yaml

    path = project_path(filename)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Navigation config must be a YAML mapping")
    hashes = {str(path): sha256(path)}
    if config.get("base_config"):
        base_path = project_path(config["base_config"])
        base = yaml.safe_load(base_path.read_text())
        if not isinstance(base, dict):
            raise ValueError("base_config must be a YAML mapping")
        hashes[str(base_path)] = sha256(base_path)
        config = {**base, **config}
        # main.navigation_tests always takes weights from the frozen base.
        config["checkpoint"] = base.get("checkpoint")
    return config, hashes


def select_python(check=False, device="cuda"):
    """Reuse compatible environments without resolving away venv symlinks."""
    system = platform.system()
    if system not in ("Linux", "Darwin"):
        raise RuntimeError("Supported hosts: Ubuntu 22.04 and macOS (local CPU checks)")
    versions = ["3.12"] if system == "Linux" else ["3.11", "3.12"]
    if device == "cuda" and (system != "Linux" or platform.machine() != "x86_64"):
        raise RuntimeError("CUDA benchmark requires Linux x86_64; use --device cpu for local checks")
    requested = os.environ.get("PYTHON_BIN")
    if requested:
        if not bootstrap.python_matches(requested, versions):
            raise RuntimeError(f"PYTHON_BIN must be an existing Python {'/'.join(versions)} interpreter")
        return bootstrap.absolute_executable(requested)
    for candidate in (ROOT / ".venv-navigation/bin/python", ROOT / ".venv/bin/python",
                      sys.executable, "python3.12", "python3.11", "python3", "python"):
        if bootstrap.python_matches(candidate, versions):
            return bootstrap.absolute_executable(candidate)
    if system == "Linux":
        if not check:
            bootstrap.ubuntu()
            bootstrap.ensure_apt(["python3-venv", "curl", "ca-certificates"])
        return bootstrap.ensure_venv(ROOT / ".venv-navigation", "3.12", check)
    raise RuntimeError("Install Python 3.11/3.12 and set PYTHON_BIN for local CPU checks")


def ensure_pip(python, check):
    result = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True, text=True)
    if result.returncode:
        if check:
            raise RuntimeError(f"pip is missing in {python}; rerun without --check-only")
        dependencies.run([python, "-m", "ensurepip", "--upgrade"])


def ensure_tmux(check):
    if not shutil.which("tmux"):
        if check or platform.system() != "Linux":
            raise RuntimeError("tmux is required only for --background; install tmux or run in foreground")
        bootstrap.ubuntu()
        bootstrap.ensure_apt(["tmux"])
    version = dependencies.run(["tmux", "-V"], capture_output=True, text=True).stdout.strip()
    match = re.search(r"\btmux\s+(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (3, 2):
        raise RuntimeError(f"Need tmux >=3.2 for argument-safe background launch; found {version!r}")
    print(f"[navigation] {version}", flush=True)


def setup_renderer(check, python, render_device, report_dir=None, threads=8):
    if platform.system() == "Linux":
        if not check:
            bootstrap.ubuntu()
        dependencies.ensure_libraries(check)
    if python:
        python = bootstrap.absolute_executable(python)
        if not bootstrap.python_matches(python, ["3.11"]):
            raise RuntimeError("RENDER_PYTHON/--render-python must be an existing Python 3.11 interpreter")
    else:
        python = bootstrap.ensure_venv(ROOT / ".venv-render-server", "3.11", check)
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        raise RuntimeError("bpy must use an isolated interpreter, separate from the navigation coordinator")
    ensure_pip(python, check)
    dependencies.ensure_requirements(python, RENDER_PROFILE, check)
    report = probe_renderer(python, report_dir / "renderer-environment.json" if report_dir else None)
    # A successful bpy import is not proof that the requested GPU can render.
    dependencies.run([python, "-m", "visual_modeling.blender.cycles_device",
                      "--device", render_device, "--threads", threads],
                     cwd=ROOT, env=renderer_environment())
    return python, report


def check_inference(checkpoint, device):
    """Exercise HDetector's actual preprocessing/forward path with loaded weights.

    A blank frame can legitimately yield no detection. Inspect the model outputs
    via a temporary hook so NaNs cannot masquerade as a successful empty result.
    This is an execution check, not an accuracy or flight-performance test.
    """
    import numpy as np
    import torch
    from navigation.vision.infer_h import HDetector

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the NVIDIA driver and container GPU access")
        value = torch.ones(16, device=device)
        if (value @ value).item() != 16:
            raise RuntimeError("CUDA tensor operation failed")
    detector = HDetector(checkpoint, device)
    if detector.model is None:
        raise RuntimeError("Benchmark requires a learned PyTorch checkpoint, not a fallback detector/ONNX CPU session")
    outputs = {}

    def record_outputs(_model, _inputs, result):
        if not isinstance(result, dict) or not {"mask_logits", "keypoints"} <= result.keys():
            raise RuntimeError("Checkpoint inference did not return mask_logits and keypoints")
        for name, value in result.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0 or not torch.isfinite(value).all().item():
                raise RuntimeError(f"Non-finite or empty network output: {name}")
            if value.device.type != device:
                raise RuntimeError(f"Inference ran on {value.device}, requested {device}")
            outputs[name] = list(value.shape)

    hook = detector.model.register_forward_hook(record_outputs)
    try:
        detection = detector(np.zeros((detector.image_size, detector.image_size, 3), dtype=np.uint8))
    finally:
        hook.remove()
    if not outputs:
        raise RuntimeError("Checkpoint inference did not execute the network")
    if device == "cuda":
        torch.cuda.synchronize()
    return dict(path=str(checkpoint), bytes=checkpoint.stat().st_size, sha256=sha256(checkpoint),
                inference="passed", input_size=detector.image_size, output_shapes=outputs,
                detected_on_probe=detection is not None)


def setup(args):
    ensure_pip(sys.executable, args.check_only)
    # Bootstrap YAML/numerical pins before reading config defaults; installing
    # the full profile first could fetch the wrong Torch build from PyPI.
    dependencies.ensure_requirements(sys.executable, ROOT / "requirements-data.txt", args.check_only)
    config, config_hashes = ({}, {}) if args.analysis_only else load_config(args.config)
    for name, fallback, choices in (
        ("device", "cuda", ("cpu", "cuda")),
        ("renderer", "blender", ("cpu", "blender")),
        ("render_device", "OPTIX", ("CPU", "CUDA", "OPTIX")),
    ):
        value = getattr(args, name) or config.get(name, fallback)
        if value not in choices:
            raise ValueError(f"Unsupported {name}: {value!r}; choose {choices}")
        setattr(args, name, value)
    if not args.analysis_only and args.device == "cuda":
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("CUDA benchmark requires Linux x86_64; select --device cpu locally")
        dependencies.ensure_torch(args.check_only)
    dependencies.ensure_requirements(sys.executable, PROFILE, args.check_only)
    # Probe imports in this fresh selected interpreter, not just package metadata.
    import cv2
    import matplotlib
    import numpy
    import scipy
    import torch
    import yaml

    if args.with_tmux:
        ensure_tmux(args.check_only)
    report = dict(status="ready", scope="analysis" if args.analysis_only else "navigation",
                  python=sys.executable, python_version=platform.python_version(), os=platform.platform(),
                  torch=torch.__version__, torch_cuda=torch.version.cuda, device=args.device,
                  renderer=args.renderer, render_device=args.render_device, render_python=None,
                  matlab_required=False, requirements_sha256=sha256(PROFILE),
                  scientific_result=False)
    if args.analysis_only:
        return report
    if args.device == "cuda" and (torch.__version__.split("+")[0], torch.version.cuda) != ("2.8.0", "12.8"):
        raise RuntimeError("Production requires PyTorch 2.8.0 CUDA 12.8 after dependency repair")
    config_path = project_path(args.config)
    checkpoint_value = args.checkpoint or config.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("Navigation config must specify checkpoint (or pass --checkpoint to this checker)")
    checkpoint = project_path(checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}; restore the configured weights before running")
    with checkpoint.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Checkpoint is a Git LFS pointer, not weights: {checkpoint}")
    report.update(config=str(config_path), config_sha256=sha256(config_path), config_files_sha256=config_hashes,
                  checkpoint=check_inference(checkpoint, args.device))
    report["source"] = str(project_path(args.source)) if args.source else None
    report["trajectory"] = str(project_path(args.trajectory)) if args.trajectory else None
    report["trajectory_kind"] = "recorded_fixed_input" if args.source else ("recorded_trajectory" if args.trajectory else "analytical_kinematic")
    if args.renderer == "blender" and not args.source:
        report["render_python"], report["render_probe"] = setup_renderer(
            args.check_only, args.render_python or os.environ.get("RENDER_PYTHON") or config.get("render_python"),
            args.render_device, args.report_dir, int(config.get("render_threads", 8)))
    if args.renderer == "cpu" and not args.source:
        print("[navigation] Synthetic CPU rendering is a plumbing check, NOT scientific sensor evidence.", flush=True)
    if args.device == "cpu":
        print("[navigation] CPU FP32 inference can evaluate accuracy on Blender/recorded inputs; report CPU timing separately.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", "--check", action="store_true", help="never install packages or runtimes")
    parser.add_argument("--config", default="configs/navigation_tests.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="config default, normally cuda")
    parser.add_argument("--renderer", choices=("cpu", "blender"), help="config default, normally blender")
    parser.add_argument("--render-device", choices=("CPU", "CUDA", "OPTIX"), help="config default, normally OPTIX")
    parser.add_argument("--render-python")
    parser.add_argument("--checkpoint", help="checker-only override; benchmark weights are selected by --config")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", help="recorded run directory; fixed-input replay needs no renderer")
    source.add_argument("--trajectory", help="recorded body/deck trajectory to rerender")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--with-tmux", action="store_true")
    parser.add_argument("--analysis-only", action="store_true", help="skip checkpoint, GPU and renderer probes")
    parser.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if args.report_dir:
        args.report_dir = project_path(args.report_dir)
    try:
        if not args.analysis_only and not project_path(args.config).is_file():
            raise FileNotFoundError(f"Navigation benchmark config missing: {project_path(args.config)}")
        if args.source and not (project_path(args.source) / "run.json").is_file():
            raise FileNotFoundError(f"Recorded --source must contain run.json: {project_path(args.source)}")
        if args.trajectory and not project_path(args.trajectory).exists():
            raise FileNotFoundError(f"Recorded trajectory missing: {project_path(args.trajectory)}")
        if not args.runtime:
            python = select_python(args.check_only, "cpu" if args.analysis_only else (args.device or "cpu"))
            # Always start a fresh interpreter after selection; do not import torch
            # in the stdlib bootstrap or reuse native modules after pip repairs.
            result = subprocess.run([python, str(Path(__file__).absolute()), "--runtime", *arguments], cwd=ROOT)
            return result.returncode
        report = setup(args)
        if args.report_dir:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
            (args.report_dir / "python.txt").write_text(sys.executable + "\n")
            (args.report_dir / "render-python.txt").write_text((report["render_python"] or "") + "\n")
            for name in ("device", "renderer", "render_device"):
                (args.report_dir / (name.replace("_", "-") + ".txt")).write_text(report[name] + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    except Exception as exc:
        # CLI boundary: YAML/checkpoint deserialization failures also need a
        # failed report, not stale success metadata or an unclassified traceback.
        if args.report_dir:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "environment.json").write_text(json.dumps(dict(status="failed", error=str(exc)), indent=2) + "\n")
        print(f"Navigation dependency check failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
