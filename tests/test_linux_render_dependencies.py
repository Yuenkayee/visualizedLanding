"""Exercise apt/ldd handling with stand-ins; never change host packages."""

import subprocess
from types import SimpleNamespace

import pytest

from visual_training.data import linux_render_dependencies as native
from visual_training.data import blender_runtime as runtime


@pytest.fixture
def ubuntu(monkeypatch):
    monkeypatch.setattr(native.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        native.platform,
        "freedesktop_os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": "22.04"},
    )
    monkeypatch.setattr(native.os, "geteuid", lambda: 0)
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(native.subprocess, "run", run)
    return calls


def test_parse_ldd_reports_all_missing_libraries(monkeypatch):
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/ldd")

    def run(command, **kwargs):
        assert command == ["/usr/bin/ldd", "/tools/blender"]
        assert kwargs["env"]["LC_ALL"] == "C"
        assert kwargs["env"]["LD_LIBRARY_PATH"] == "/custom/lib"
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "\tlibSM.so.6 => not found\n\tlibXrender.so.1 => not found\n"
                "\tlibSM.so.6 => not found\n\tlibm.so.6 => /lib/libm.so.6 (0x1)\n"
            ),
        )

    monkeypatch.setattr(native.subprocess, "run", run)
    assert native.missing_libraries(
        "/tools/blender", {"LD_LIBRARY_PATH": "/custom/lib"}
    ) == ["libSM.so.6", "libXrender.so.1"]


def test_non_linux_does_not_check_or_install(monkeypatch):
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native, "missing_libraries", lambda *a: pytest.fail("ldd"))
    native.ensure_blender_libraries("blender", {})


def test_complete_libraries_do_not_run_apt(ubuntu, monkeypatch):
    monkeypatch.setattr(native, "missing_libraries", lambda *a: [])
    native.ensure_blender_libraries("blender", {})
    assert ubuntu == []


@pytest.mark.parametrize("uid", [0, 1000])
def test_install_missing_packages_then_recheck(ubuntu, monkeypatch, uid):
    monkeypatch.setattr(native.os, "geteuid", lambda: uid)
    states = iter([["libSM.so.6", "libXrender.so.1"], []])
    monkeypatch.setattr(native, "missing_libraries", lambda *a: next(states))
    native.ensure_blender_libraries("blender", {})
    prefix = ["/usr/bin/sudo", "-n"] if uid else []
    expected = [
        [*prefix, "/usr/bin/apt-get", "update"],
        [
            *prefix,
            "/usr/bin/apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "libsm6",
            "libxrender1",
        ],
    ]
    if uid:
        expected.insert(0, ["/usr/bin/sudo", "-n", "true"])
    assert ubuntu == expected


def test_no_privilege_gives_command_without_attempting_install(ubuntu, monkeypatch):
    monkeypatch.setattr(native.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(native, "missing_libraries", lambda *a: ["libSM.so.6"])

    def run(command, **kwargs):
        ubuntu.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="sudo apt-get update") as error:
        native.ensure_blender_libraries("blender", {})
    assert "libsm6" in str(error.value)
    assert ubuntu == [["/usr/bin/sudo", "-n", "true"]]


def test_unknown_driver_library_does_not_install_any_packages(ubuntu, monkeypatch):
    monkeypatch.setattr(
        native, "missing_libraries", lambda *a: ["libSM.so.6", "libcuda.so.1"]
    )
    with pytest.raises(RuntimeError, match="libcuda.so.1"):
        native.ensure_blender_libraries("blender", {})
    assert ubuntu == []


@pytest.mark.parametrize("failed_step", ["update", "install"])
def test_apt_failure_stops_setup(ubuntu, monkeypatch, failed_step):
    monkeypatch.setattr(native, "missing_libraries", lambda *a: ["libSM.so.6"])

    def run(command, **kwargs):
        ubuntu.append(command)
        if failed_step in command:
            raise subprocess.CalledProcessError(100, command)

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        native.ensure_blender_libraries("blender", {})
    assert error.value.returncode == 100
    assert len(ubuntu) == (1 if failed_step == "update" else 2)


def test_failed_post_install_check_stops(ubuntu, monkeypatch):
    monkeypatch.setattr(native, "missing_libraries", lambda *a: ["libSM.so.6"])
    with pytest.raises(RuntimeError, match="still unavailable"):
        native.ensure_blender_libraries("blender", {})
    assert len(ubuntu) == 2


def test_unsupported_distribution_is_not_modified(ubuntu, monkeypatch):
    monkeypatch.setattr(
        native.platform, "freedesktop_os_release", lambda: {"ID": "fedora"}
    )
    monkeypatch.setattr(native, "missing_libraries", lambda *a: ["libSM.so.6"])
    with pytest.raises(RuntimeError, match="Ubuntu/Debian"):
        native.ensure_blender_libraries("blender", {})
    assert ubuntu == []


def test_native_dependency_failure_prevents_blender_and_pip(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.delenv("LANDING_BLENDER", raising=False)
    monkeypatch.setattr(runtime, "portable_blender", lambda *a, **kw: "/tools/blender")

    def fail(executable, env):
        assert executable == "/tools/blender"
        raise RuntimeError("native libraries unavailable")

    monkeypatch.setattr(runtime, "ensure_blender_libraries", fail)
    monkeypatch.setattr(
        runtime, "blender_info", lambda *a: pytest.fail("Blender launched")
    )
    monkeypatch.setattr(
        runtime, "ensure_dependencies", lambda *a: pytest.fail("pip launched")
    )
    with pytest.raises(RuntimeError, match="native libraries"):
        runtime.main(["--setup-only"])
