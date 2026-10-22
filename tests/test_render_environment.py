"""Render-process native libraries must be isolated from the MATLAB host."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shell import render_environment as render


@pytest.fixture
def matlab_env(tmp_path, monkeypatch):
    root = tmp_path / "MATLAB/R2024b"
    (root / "bin/glnxa64").mkdir(parents=True)
    monkeypatch.setattr(render.platform, "system", lambda: "Linux")
    monkeypatch.setenv("MATLAB_ROOT", str(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{root}/bin/glnxa64:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:{root}/sys/os/glnxa64")
    monkeypatch.setenv("LD_PRELOAD", f"{root}/bin/glnxa64/libvulkan.so.1:/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return root


def test_remove_matlab_libraries_and_keep_gpu_and_system_cpp(matlab_env):
    before = dict(os.environ)
    env = render.renderer_environment()
    assert env["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
    assert env["LD_PRELOAD"] == "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    assert env["MATLAB_ROOT"] == str(matlab_env)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert dict(os.environ) == before  # MATLAB still needs its own environment.


def test_matlab_only_paths_are_unset_in_child(matlab_env):
    env = render.renderer_environment({"MATLAB_ROOT": str(matlab_env),
                                       "LD_LIBRARY_PATH": f":{matlab_env}/bin/glnxa64:",
                                       "LD_PRELOAD": str(matlab_env / "lib.so")})
    assert "LD_LIBRARY_PATH" not in env and "LD_PRELOAD" not in env


def test_symlink_and_second_matlab_are_removed_but_sibling_directory_is_kept(matlab_env, tmp_path):
    alias = tmp_path / "matlab-libs"
    alias.symlink_to(matlab_env / "bin/glnxa64")
    other = tmp_path / "another-install"
    (other / "bin").mkdir(parents=True)
    (other / "bin/matlab").touch()
    (other / "VersionInfo.xml").touch()
    sibling = str(matlab_env) + "-user-libs"
    env = render.renderer_environment({"MATLAB_ROOT": str(matlab_env),
        "LD_LIBRARY_PATH": f"{alias}:{other}/bin/glnxa64:{sibling}"})
    assert env["LD_LIBRARY_PATH"] == sibling


def test_no_matlab_configuration_preserves_normal_linux_libraries(monkeypatch):
    monkeypatch.setattr(render.platform, "system", lambda: "Linux")
    env = {"LD_LIBRARY_PATH": "/cuda/lib64", "LD_PRELOAD": "libstdc++.so.6", "PATH": "/usr/bin"}
    assert render.renderer_environment(env) == env


def test_macos_environment_is_unchanged(matlab_env, monkeypatch):
    monkeypatch.setattr(render.platform, "system", lambda: "Darwin")
    assert render.renderer_environment() == dict(os.environ)


def test_render_child_receives_isolated_environment(matlab_env, monkeypatch):
    # Do not try to load Linux preloads on a macOS test host.
    monkeypatch.delenv("LD_PRELOAD")
    result = subprocess.run([sys.executable, "-c", "import os; print(os.environ['LD_LIBRARY_PATH'])"],
                            env=render.renderer_environment(), capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
    assert str(matlab_env) in os.environ["LD_LIBRARY_PATH"]


def test_probe_writes_report_for_real_child_import_failure(tmp_path, matlab_env, monkeypatch):
    monkeypatch.delenv("LD_PRELOAD")
    (tmp_path / "bpy.py").write_text("raise ImportError('undefined symbol: vkGetPhysicalDeviceXlibPresentationSupportKHR')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    report_path = tmp_path / "logs/renderer-environment.json"
    with pytest.raises(RuntimeError, match="vkGetPhysicalDeviceXlibPresentationSupportKHR"):
        render.probe_renderer(sys.executable, report_path)
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["returncode"] != 0
    assert str(matlab_env) not in report["LD_LIBRARY_PATH"]


def test_probe_writes_success_report_without_mutating_parent(tmp_path, matlab_env, monkeypatch):
    monkeypatch.delenv("LD_PRELOAD")
    (tmp_path / "bpy.py").write_text("from types import SimpleNamespace\napp=SimpleNamespace(version_string='test')\n")
    for name in ("cv2", "numpy", "scipy", "yaml"):
        (tmp_path / f"{name}.py").touch()
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    report_path = tmp_path / "renderer-environment.json"
    result = render.probe_renderer(sys.executable, report_path)
    assert result["status"] == "ready" and result["blender"] == "test"
    assert json.loads(report_path.read_text()) == result
    assert str(matlab_env) in os.environ["LD_LIBRARY_PATH"]


def test_environment_check_uses_same_probe(tmp_path, matlab_env, monkeypatch):
    from main import check_environment as check
    monkeypatch.setattr(check, "ensure_libraries", lambda *a: None)
    monkeypatch.setattr(check, "ensure_requirements", lambda *a: None)
    monkeypatch.setattr(check, "python_matches", lambda *a: True)
    seen = []
    monkeypatch.setattr(check, "probe_renderer", lambda python, report: seen.append((python, report)))
    report = tmp_path / "renderer-environment.json"
    assert check.renderer(True, sys.executable, report) == sys.executable
    assert seen == [(sys.executable, report)]


def test_persistent_sensor_worker_uses_isolated_environment(tmp_path, matlab_env, monkeypatch):
    from main import sensors
    seen = []

    def popen(command, **kwargs):
        seen.append(kwargs["env"])
        return SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(sensors.PREFIX + '{"renderer":"blender"}\n'),
                               poll=lambda: 0)

    monkeypatch.setattr(sensors.subprocess, "Popen", popen)
    with sensors.Sensors({"renderer": "blender", "render_python": sys.executable}, {}, {}, tmp_path) as sensor:
        assert sensor.metadata["renderer"] == "blender"
    assert seen[0]["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"


def test_dataset_workers_use_same_isolation(tmp_path, matlab_env, monkeypatch):
    from visual_training.data import generate_server_dataset as dataset
    seen = []

    def popen(command, **kwargs):
        seen.append(kwargs["env"])
        return SimpleNamespace(poll=lambda: 0, wait=lambda **kw: 0)

    monkeypatch.setattr(dataset.subprocess, "Popen", popen)
    dataset.run_workers(sys.executable, tmp_path, 2, 1)
    assert len(seen) == 2
    assert all(env["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64:/usr/local/nvidia/lib64" for env in seen)
    assert seen[0]["OMP_NUM_THREADS"] == "1"


def test_vulkan_loader_is_checked_as_system_dependency(monkeypatch):
    from shell import server_dependencies as server
    monkeypatch.setattr(server.ctypes.util, "find_library", lambda name: None if name == "vulkan" else "present")
    with pytest.raises(RuntimeError, match="libvulkan1"):
        server.ensure_libraries(check=True)
    assert "libvulkan1" in (server.ROOT / "shell/requirements-ubuntu2204-closed-loop.txt").read_text().splitlines()
