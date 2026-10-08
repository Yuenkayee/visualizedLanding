"""Launch rendering with portable Blender, isolated from the training Python ABI."""

import argparse
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
VERSION = "4.5.3"
ARCHIVE = f"blender-{VERSION}-linux-x64.tar.xz"
URL = f"https://download.blender.org/release/Blender4.5/{ARCHIVE}"
SHA256 = "975c58fcb244273838534bba771e64ad87739216b0f9b39a888531a49a72d845"
LOCK = ROOT / "requirements-render.txt"
ENTRY = ROOT / "visual_training/data/blender_multi_camera_entry.py"
MODULE = "visual_training.data.generate_multi_camera_dataset"
REQUIRED = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "opencv-python-headless": "4.10.0.84",
    "PyYAML": "6.0.2",
}


def isolated_env():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LANDING_SITE_PACKAGES"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def download_archive(path):
    """Retry interrupted HTTPS transfers; publish only a checksum-verified file."""
    if path.exists():
        with path.open("rb") as cached:
            if hashlib.file_digest(cached, "sha256").hexdigest() == SHA256:
                return
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 6):
        try:
            print(f"Downloading {URL} (attempt {attempt}/5)", flush=True)
            digest = hashlib.sha256()
            with (
                urllib.request.urlopen(URL, timeout=90) as source,
                partial.open("wb") as dest,
            ):
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != SHA256:
                raise ValueError("Blender archive SHA256 mismatch; refusing to extract")
            partial.replace(path)
            return
        except (
            OSError,
            http.client.HTTPException,
            urllib.error.URLError,
            ValueError,
        ) as error:
            if attempt == 5:
                raise RuntimeError(
                    f"Blender download failed: {error}. Retry this command, or download "
                    f"{URL} separately and pass --blender /path/to/blender."
                ) from error
            print(f"Download interrupted: {error}; retrying", flush=True)
            time.sleep(min(2 ** (attempt - 1), 8))


def portable_blender(tools_dir):
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError(
            "Automatic Blender installation supports Linux x86_64. "
            "Pass --blender /path/to/Blender-4.5/blender on this platform."
        )
    folder = ARCHIVE.removesuffix(".tar.xz")
    executable = tools_dir / folder / "blender"
    if not executable.is_file():
        archive = tools_dir / ARCHIVE
        download_archive(archive)
        with tempfile.TemporaryDirectory(prefix="extract-", dir=tools_dir) as staging:
            with tarfile.open(archive) as bundle:
                bundle.extractall(staging, filter="data")
            extracted = Path(staging) / folder
            if not (extracted / "blender").is_file():
                raise RuntimeError("Blender archive is missing its executable")
            extracted.rename(executable.parent)
    return executable


def blender_info(executable):
    expression = (
        "import bpy, sys, json; print('LANDING_BLENDER_INFO=' + json.dumps("
        "dict(blender=list(bpy.app.version), python=list(sys.version_info[:3]), "
        "executable=sys.executable)))"
    )
    result = run(
        [
            executable,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
        ],
        env=isolated_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("LANDING_BLENDER_INFO="):
            info = json.loads(line.split("=", 1)[1])
            if info["blender"][:2] != [4, 5] or info["python"][:2] != [3, 11]:
                raise RuntimeError(
                    f"Need Blender 4.5 LTS with embedded Python 3.11; got {info}"
                )
            print(f"Blender runtime: {json.dumps(info)}", flush=True)
            return info
    raise RuntimeError(f"Cannot identify Blender's embedded Python:\n{result.stdout}")


def dependency_check(python, site=None):
    # -I ignores Conda/user site paths. The only added site is our cp311 target.
    source = (
        "import sys; "
        + (f"sys.path.insert(0, {str(site)!r}); " if site else "")
        + "import importlib.metadata as m; "
        + f"required={REQUIRED!r}; "
        + "assert all(m.version(k)==v for k,v in required.items()), 'render dependency versions differ'; "
        + "import numpy, scipy, scipy.spatial.transform, cv2, yaml; "
        + "print('Render dependencies OK')"
    )
    return subprocess.run(
        [str(python), "-I", "-c", source],
        env=isolated_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def ensure_dependencies(python, site=None):
    if dependency_check(python, site).returncode == 0:
        print(f"Render dependencies ready: {site or python}", flush=True)
        return
    # pip's --python installs wheels for Blender's interpreter, not the launcher ABI.
    if subprocess.run(
        [sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL
    ).returncode:
        run([sys.executable, "-m", "ensurepip", "--upgrade"])
    command = [
        sys.executable,
        "-m",
        "pip",
        "--python",
        python,
        "install",
        "--only-binary=:all:",
        "--retries",
        "5",
        "--timeout",
        "60",
        "-r",
        LOCK,
    ]
    if site:
        site.mkdir(parents=True, exist_ok=True)
        command.extend(["--target", site, "--upgrade"])
    run(command, env=isolated_env())
    result = dependency_check(python, site)
    if result.returncode:
        raise RuntimeError(
            f"Rendering dependency import check failed:\n{result.stdout}"
        )
    print(result.stdout.strip(), flush=True)


@contextlib.contextmanager
def setup_lock(tools_dir):
    import fcntl

    tools_dir.mkdir(parents=True, exist_ok=True)
    with (tools_dir / "render-setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Dataset options are forwarded unchanged: --config PATH, --output DIR, "
            "--preview, --sorties-per-weather N (>=3), --frames-per-stage N, "
            "--width N, --samples N. The shell also accepts --python PATH. "
            "CPU renderer is simplified test imagery; use Blender for training."
        ),
    )
    parser.add_argument(
        "--blender",
        default=os.environ.get("LANDING_BLENDER"),
        help="existing Blender 4.5 executable; default downloads pinned 4.5.3 on Linux x86_64",
    )
    parser.add_argument(
        "--cycles-device", choices=["OPTIX", "CUDA", "CPU", "METAL"], default="OPTIX"
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="index in Cycles devices for this backend",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="install rendering dependencies, render a probe, then exit",
    )
    parser.add_argument("--renderer", choices=["blender", "cpu"], default="blender")
    parser.add_argument("--plan-only", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    forwarded += ["--renderer", args.renderer]
    if args.plan_only:
        forwarded.append("--plan-only")
    if args.plan_only or args.renderer == "cpu":
        ensure_dependencies(sys.executable)
        if not args.setup_only:
            run([sys.executable, "-u", "-m", MODULE, *forwarded], cwd=ROOT)
        return
    tools_dir = ROOT / ".tools"
    with setup_lock(tools_dir):
        if args.blender:
            executable = shutil.which(args.blender)
            if not executable:
                raise RuntimeError(f"Cannot execute Blender: {args.blender}")
        else:
            executable = portable_blender(tools_dir)
        info = blender_info(executable)
        site = tools_dir / "blender-4.5-py311-site"
        ensure_dependencies(info["executable"], site)
    env = isolated_env()
    env["LANDING_RENDER_SITE"] = str(site)
    command = [
        executable,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        ENTRY,
        "--",
        "--cycles-device",
        args.cycles_device,
        "--gpu-index",
        str(args.gpu_index),
    ]
    if args.setup_only:
        command.append("--setup-only")
    run([*command, *forwarded], env=env, cwd=ROOT)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        print(
            f"Rendering command failed (exit {error.returncode}). See error above.",
            file=sys.stderr,
        )
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except (OSError, RuntimeError) as error:
        print(f"Rendering setup failed: {error}", file=sys.stderr)
        sys.exit(1)
