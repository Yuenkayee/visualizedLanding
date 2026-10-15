"""Small non-rendering dependency profiles; preserve the server CUDA environment."""
import argparse
from pathlib import Path
import platform
import sys

# This file is launched directly, before any third-party import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shell.server_dependencies import ROOT, ensure_requirements, ensure_torch, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=("prepare", "replay"))
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    run([sys.executable, "-m", "pip", "--version"])
    if a.profile == "replay" and platform.system() == "Linux" and platform.machine() == "x86_64" and sys.version_info[:2] == (3, 12):
        ensure_torch(a.check)
    profile = "requirements-data.txt" if a.profile == "prepare" else "requirements-replay.txt"
    ensure_requirements(sys.executable, ROOT / profile, a.check)
    imports = "import numpy, scipy, cv2, yaml" + (", torch, matplotlib" if a.profile == "replay" else "")
    run([sys.executable, "-c", imports])


if __name__ == "__main__":
    main()
