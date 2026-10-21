"""OS/Python/MATLAB bootstrap. Runs with Ubuntu's Python 3.10, stdlib only.

The launcher invokes this before main.check_environment checks Python packages,
licenses, CUDA and Cycles. Downloads and builds stay under ignored .tools/.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools/closed-loop"
PYTHON_RELEASES = {"3.12": "3.12.12", "3.11": "3.11.14"}
PRODUCTS = {
    "MATLAB": "bin/matlab",
    "Simulink": "toolbox/simulink/simulink/Contents.m",
    "Control_System_Toolbox": "toolbox/control/control/Contents.m",
    "Robust_Control_Toolbox": "toolbox/robust/robust/Contents.m",
}


def run(command, **kwargs):
    print("[deploy] " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def ubuntu():
    info = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value.strip('"')
    if info.get("ID") != "ubuntu" or info.get("VERSION_ID") != "22.04" or platform.machine() != "x86_64":
        raise RuntimeError("Automatic server installation requires Ubuntu 22.04 x86_64")


def privilege():
    if os.geteuid() == 0:
        return []
    if not shutil.which("sudo"):
        raise RuntimeError("Installing system packages requires root or sudo; ask the server administrator")
    # sudo reads its password from the controlling terminal, never from a log.
    run(["sudo", "-v"])
    return ["sudo", "-n"]


def ensure_apt(packages, check=False):
    missing = []
    for name in sorted(set(packages)):
        p = subprocess.run(["dpkg-query", "-W", "-f=${Status}", name], capture_output=True, text=True)
        if p.returncode or p.stdout.strip() != "install ok installed":
            missing.append(name)
    if not missing:
        return
    if check:
        raise RuntimeError("Missing Ubuntu packages: " + " ".join(missing) + "; rerun without --check")
    prefix = privilege()
    run([*prefix, "apt-get", "update"])
    run([*prefix, "env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y",
         "--no-install-recommends", *missing])


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    run(["curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
         "--connect-timeout", "20", "--max-time", "1800", "--output", partial, url])
    partial.replace(target)


def python_matches(binary, versions):
    try:
        p = subprocess.run([str(binary), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True)
        return p.returncode == 0 and p.stdout.strip() in versions
    except OSError:
        return False


def absolute_executable(value):
    # abspath deliberately keeps venv symlinks: resolve() loses the environment.
    result = shutil.which(str(value))
    return os.path.abspath(result or os.path.expanduser(str(value)))


def ensure_uv():
    python = TOOLS / "bootstrap/bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", python.parent.parent])
    p = subprocess.run([str(python), "-m", "uv", "--version"], capture_output=True, text=True)
    if p.returncode or p.stdout.strip() != "uv 0.9.5":
        run([python, "-m", "pip", "install", "uv==0.9.5"])
    return [str(python), "-m", "uv"]


def source_python(version):
    """Fallback to python.org when GitHub binary downloads fail (e.g. TLS EOF)."""
    release = PYTHON_RELEASES[version]
    prefix = TOOLS / "python" / release
    binary = prefix / "bin" / ("python" + version)
    if python_matches(binary, [version]):
        return binary
    ensure_apt(["build-essential", "pkg-config", "libssl-dev", "zlib1g-dev", "libbz2-dev",
                "libreadline-dev", "libsqlite3-dev", "libffi-dev", "liblzma-dev",
                "libncursesw5-dev", "libgdbm-dev", "uuid-dev"])
    archive = TOOLS / f"downloads/Python-{release}.tgz"
    download(f"https://www.python.org/ftp/python/{release}/Python-{release}.tgz", archive)
    print(f"[deploy] Building Python {release} from python.org; this may take several minutes", flush=True)
    # Build in a fresh folder so an interrupted build can be retried safely.
    with tempfile.TemporaryDirectory(prefix="python-build-", dir=TOOLS) as temp:
        run(["tar", "-xzf", archive, "-C", temp, "--strip-components=1", "--no-same-owner"])
        run(["./configure", f"--prefix={prefix}", "--enable-shared", "--with-ensurepip=install",
             f"LDFLAGS=-Wl,-rpath,{prefix}/lib"], cwd=temp)
        run(["make", "-j", str(min(os.cpu_count() or 2, 16))], cwd=temp)
        run(["make", "install"], cwd=temp)
    run([binary, "-c", "import ssl, sqlite3, bz2, lzma, ctypes, venv; print('Python source build: OK')"])
    return binary


def ensure_venv(folder, version, check=False):
    folder = Path(folder)
    binary = folder / "bin/python"
    if python_matches(binary, [version]):
        return str(binary)
    if check:
        raise RuntimeError(f"Missing usable Python {version}: {binary}; rerun without --check")
    if folder.exists() or folder.is_symlink():
        backup = folder.with_name(folder.name + f".backup.{datetime.now():%Y%m%d_%H%M%S}.{os.getpid()}")
        folder.rename(backup)
        print(f"[deploy] Preserved incomplete/incompatible environment: {backup}", flush=True)
    candidates = [os.environ.get("LANDING_PYTHON_" + version.replace(".", ""), ""),
                  f"python{version}", sys.executable, str(Path.home() / ".venv/bin/python"),
                  str(Path.home() / "miniconda3/bin/python"), "/opt/conda/bin/python"]
    existing = next((absolute_executable(p) for p in candidates if p and python_matches(p, [version])), None)
    uv = ensure_uv()
    try:
        run([*uv, "venv", "--python", existing or version, "--seed", folder],
            env={**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "120"),
                 "UV_HTTP_RETRIES": os.environ.get("UV_HTTP_RETRIES", "5")})
    except subprocess.CalledProcessError:
        if existing:
            # Seeding can fail independently of interpreter downloads.
            raise RuntimeError("Virtual environment creation failed; inspect pip/index connectivity above") from None
        print("[deploy] Managed Python download failed; trying python.org source build", flush=True)
        compiled = source_python(version)
        # venv's ensurepip uses bundled wheels, without a further network request.
        run([compiled, "-m", "venv", folder])
    if not python_matches(binary, [version]):
        raise RuntimeError(f"Python {version} environment creation failed: {binary}")
    return str(binary)


def host_python(check=False):
    requested = os.environ.get("PYTHON_BIN")
    versions = ["3.11", "3.12"]
    if requested:
        if not python_matches(requested, versions):
            raise RuntimeError(f"PYTHON_BIN={requested} is unavailable or not Python 3.11/3.12; unset PYTHON_BIN to auto-install")
        return absolute_executable(requested)
    local = ROOT / ".venv/bin/python"
    if python_matches(local, versions):
        return str(local)
    if platform.system() == "Linux":
        # Reuse a cloud PyTorch image's Python 3.12 and CUDA wheels if available.
        for candidate in (sys.executable, "python", "python3.12", "python3"):
            if python_matches(candidate, ["3.12"]):
                return absolute_executable(candidate)
        return ensure_venv(ROOT / ".venv", "3.12", check)
    for candidate in (sys.executable, "python3", "python3.11", "python3.12", "python"):
        if python_matches(candidate, versions):
            return absolute_executable(candidate)
    raise RuntimeError("Local macOS requires Python 3.11/3.12; set PYTHON_BIN")


def matlab_destination():
    if os.environ.get("MATLAB_ROOT"):
        return Path(os.environ["MATLAB_ROOT"]).expanduser().absolute()
    candidate = shutil.which("matlab")
    candidates = ([Path(candidate).resolve().parent.parent] if candidate else []) + [
        Path("/Applications/MATLAB_R2024b.app"), Path.home() / "MATLAB/R2024b",
        Path("/usr/local/MATLAB/R2024b"), Path("/opt/MATLAB/R2024b")]
    for root in candidates:
        if (root / "bin/matlab").is_file():
            return root
    return Path.home() / "MATLAB/R2024b"


def ensure_matlab(root, check=False):
    release = "R2024b"
    info = root / "VersionInfo.xml"
    if info.exists():
        tree = ET.parse(info)
        if tree.findtext("release") != release:
            raise RuntimeError(f"MATLAB_ROOT must point to R2024b: {root}")
        match = re.search(r"Update\s+(\d+)", tree.findtext("description") or "")
        # When adding products to an existing installation, preserve its update.
        release += "U" + (match.group(1) if match else "0")
    missing = [name for name, marker in PRODUCTS.items() if not (root / marker).is_file()]
    if not info.exists() and "MATLAB" not in missing:
        missing.append("MATLAB")
    if not missing:
        print(f"[deploy] MATLAB R2024b and required products installed: {root}", flush=True)
        return
    if check or platform.system() != "Linux":
        raise RuntimeError(f"Missing MATLAB products {missing} at {root}; Ubuntu: rerun without --check; macOS: use MathWorks installer")
    mpm = TOOLS / "mpm"
    download("https://www.mathworks.com/mpm/glnxa64/mpm", mpm)
    mpm.chmod(0o755)
    parent = root
    while not parent.exists():
        parent = parent.parent
    prefix = [] if os.access(parent, os.W_OK) else privilege()
    temp = TOOLS / "matlab-tmp"
    temp.mkdir(parents=True, exist_ok=True)
    run([*prefix, "env", f"TMPDIR={temp}", mpm, "install", f"--release={release}",
         f"--destination={root}", "--products=MATLAB", *list(PRODUCTS)[1:]])
    # Recheck file installation separately from account/license authorization.
    ensure_matlab(root, check=True)


def matlab_login(root):
    mode = os.environ.get("MATLAB_LICENSE_MODE", "online")
    if mode not in ("online", "onlinelicensing"):
        raise RuntimeError("MATLAB_LICENSE_MODE must be online or onlinelicensing")
    print("[deploy] MATLAB account login: follow the terminal prompts; enter exit at >> to continue deployment. "
          "Login input/output bypass console.log.", flush=True)
    try:
        tty_input = open("/dev/tty", "rb", buffering=0)
    except OSError as exc:
        raise RuntimeError("--login needs an interactive SSH/tmux terminal; run the check command with --login there") from exc
    with tty_input, open("/dev/tty", "wb", buffering=0) as tty_output:
        subprocess.run([str(root / "bin/matlab"), "-licmode", mode, "-nodisplay"],
                       stdin=tty_input, stdout=tty_output, stderr=tty_output, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--login", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    a = p.parse_args()
    if a.check and a.login:
        p.error("--check cannot be combined with --login")
    if platform.system() == "Linux":
        ubuntu()
        packages = ["ca-certificates", "curl", "python3-venv", "python3-pip"]
        if not a.analysis_only:
            packages = [line.split("#", 1)[0].strip() for line in
                        (ROOT / "shell/requirements-ubuntu2204-closed-loop.txt").read_text().splitlines()]
        ensure_apt([v for v in packages if v], a.check)
    elif platform.system() != "Darwin":
        raise RuntimeError("Supported hosts: Ubuntu 22.04 x86_64 and macOS")
    python = host_python(a.check)
    # Minimal images or explicitly supplied Python may not have pip.
    if subprocess.run([python, "-m", "pip", "--version"], capture_output=True).returncode:
        if a.check:
            raise RuntimeError(f"pip missing from {python}; rerun without --check")
        run([python, "-m", "ensurepip", "--upgrade"])
    root = matlab_destination()
    if not a.analysis_only:
        ensure_matlab(root, a.check)
        if a.login:
            matlab_login(root)
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / "python.txt").write_text(python + "\n")
    (a.report_dir / "matlab-root.txt").write_text(str(root) + "\n")
    (a.report_dir / "bootstrap.json").write_text(json.dumps(dict(
        status="installed", python=python, matlab_root=str(root), check_only=a.check,
        note="Licenses, Python packages, CUDA and renderer are verified by the next stage."
    ), indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Deployment failed: {exc}")
