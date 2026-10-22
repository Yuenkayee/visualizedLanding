"""Check Blender's ELF dependencies and install known Ubuntu/Debian runtime libs."""

import os
import platform
import re
import shlex
import shutil
import subprocess

# Runtime libraries only. NVIDIA drivers and CUDA are managed by the host.
PACKAGES = {
    "libSM.so.6": "libsm6",
    "libICE.so.6": "libice6",
    "libX11.so.6": "libx11-6",
    "libXext.so.6": "libxext6",
    "libXrender.so.1": "libxrender1",
    "libXi.so.6": "libxi6",
    "libXfixes.so.3": "libxfixes3",
    "libXrandr.so.2": "libxrandr2",
    "libXxf86vm.so.1": "libxxf86vm1",
    "libXcursor.so.1": "libxcursor1",
    "libXinerama.so.1": "libxinerama1",
    "libxkbcommon.so.0": "libxkbcommon0",
    "libxkbcommon-x11.so.0": "libxkbcommon-x11-0",
    "libxcb.so.1": "libxcb1",
    "libGL.so.1": "libgl1",
    "libGLX.so.0": "libglx0",
    "libOpenGL.so.0": "libopengl0",
    "libEGL.so.1": "libegl1",
    "libGLdispatch.so.0": "libglvnd0",
    "libgomp.so.1": "libgomp1",
    "libdbus-1.so.3": "libdbus-1-3",
    "libXau.so.6": "libxau6",
    "libXdmcp.so.6": "libxdmcp6",
}


def missing_libraries(executable, env):
    ldd = shutil.which("ldd")
    if not ldd:
        raise RuntimeError("ldd not found; install libc-bin on Ubuntu/Debian and retry")
    result = subprocess.run(
        [ldd, str(executable)],
        env=dict(env, LC_ALL="C"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    missing = sorted(
        set(re.findall(r"^\s*(\S+)\s+=>\s+not found\s*$", result.stdout, re.M))
    )
    if result.returncode and not missing:
        raise RuntimeError(f"Cannot check Blender shared libraries:\n{result.stdout}")
    return missing


def ensure_blender_libraries(executable, env):
    if platform.system() != "Linux":
        return
    missing = missing_libraries(executable, env)
    if not missing:
        print("Blender system libraries OK", flush=True)
        return
    print("Missing Blender system libraries: " + ", ".join(missing), flush=True)
    unknown = [name for name in missing if name not in PACKAGES]
    if unknown:
        raise RuntimeError(
            "Cannot automatically install these Blender libraries: "
            + ", ".join(unknown)
            + ". Check the Blender installation and host runtime/driver libraries. "
            "No driver or CUDA packages were installed."
        )
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    distro = {release.get("ID", ""), *release.get("ID_LIKE", "").split()}
    apt = shutil.which("apt-get")
    if not apt or not distro.intersection({"ubuntu", "debian"}):
        raise RuntimeError(
            "Automatic system-library installation requires Ubuntu/Debian with apt-get. "
            "Install these libraries using your system package manager: "
            + ", ".join(missing)
        )
    packages = sorted({PACKAGES[name] for name in missing})
    install = [apt, "install", "-y", "--no-install-recommends", *packages]
    prefix = []
    if os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if (
            not sudo
            or subprocess.run(
                [sudo, "-n", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
        ):
            raise RuntimeError(
                "Installing missing system libraries requires root or passwordless sudo. "
                "Ask an administrator to run:\nsudo apt-get update\n"
                + shlex.join(["sudo", *install])
                + "\nThen rerun the original generation command."
            )
        prefix = [sudo, "-n"]
    print("Installing Blender runtime packages: " + " ".join(packages), flush=True)
    # Refresh first: fresh server/container images may have no package lists.
    subprocess.run([*prefix, apt, "update"], check=True)
    subprocess.run([*prefix, *install], check=True)
    remaining = missing_libraries(executable, env)
    if remaining:
        raise RuntimeError(
            "Blender libraries still unavailable after installation: "
            + ", ".join(remaining)
            + ". Check the system loader paths, including LD_LIBRARY_PATH."
        )
    print("Blender system libraries OK after installation", flush=True)
