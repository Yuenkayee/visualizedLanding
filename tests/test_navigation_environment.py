"""Catch lazy SciPy failures before the environment is declared ready."""

import json
import os
import sys

import pytest

from shell import navigation_environment as navigation


def test_healthy_runtime_checks_real_operations_without_starting_matlab(tmp_path, monkeypatch):
    (tmp_path / "sitecustomize.py").write_text('''
import importlib.abc, sys
class BlockHeavyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("matlab.engine", "bpy"):
            raise RuntimeError("Runtime probe must not start MATLAB or Blender")
sys.meta_path.insert(0, BlockHeavyRuntime())
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/navigation-test/matlab:/navigation-test/cuda")
    report_path = tmp_path / "logs/navigation-environment.json"
    report = navigation.probe_navigation(sys.executable, report_path)
    assert report["status"] == "ready" and report["returncode"] == 0
    assert report["checks"] == ["main.run import", "NumPy/SciPy operations",
                                "OpenCV/NumPy operations", "Torch/NumPy operations",
                                "matplotlib import"]
    assert report["LD_LIBRARY_PATH"] == os.environ["LD_LIBRARY_PATH"]
    assert report["packages"]["numpy"]["path"]
    assert report["packages"]["scipy"]["version"]
    assert json.loads(report_path.read_text()) == report


def test_scipy_stats_recursion_fails_in_actual_entrypoint_import_order(tmp_path, monkeypatch):
    # Top-level numpy/scipy imports and their metadata still work, as in the
    # original false-ready report. Only the lazy scipy.stats import is broken.
    (tmp_path / "sitecustomize.py").write_text('''
import importlib.abc, sys
class BrokenStats(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy.stats":
            assert "main.config" in sys.modules and "main.sensors" in sys.modules
            raise RecursionError("simulated NumPy dtype recursion")
sys.meta_path.insert(0, BrokenStats())
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    report_path = tmp_path / "navigation-environment.json"
    with pytest.raises(RuntimeError, match="simulated NumPy dtype recursion") as error:
        navigation.probe_navigation(sys.executable, report_path)
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["returncode"] != 0
    assert report["stage"] == "main.run import"
    assert report["error_type"] == "RecursionError"
    assert report["packages"]["scipy"]["installed_version"]
    assert report["packages"]["numpy"]["path"]
    assert "measurement_gating.py" in report["traceback"]
    assert sys.executable in str(error.value)
    assert "--no-deps" in str(error.value)


def test_numerical_failure_is_detected_even_when_imports_succeed(tmp_path, monkeypatch):
    (tmp_path / "sitecustomize.py").write_text('''
import builtins
original_import = builtins.__import__
def altered_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    if name == "scipy.stats" and hasattr(module, "chi2"):
        module.chi2.ppf = lambda *args, **kwargs: 0.
    return module
builtins.__import__ = altered_import
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    report_path = tmp_path / "navigation-environment.json"
    with pytest.raises(RuntimeError, match="NumPy/SciPy operations"):
        navigation.probe_navigation(sys.executable, report_path)
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["checks"] == ["main.run import"]


@pytest.mark.parametrize("read_only", [False, True])
def test_environment_check_stops_before_matlab_on_bad_navigation(tmp_path, monkeypatch, read_only):
    from main import check_environment as check

    root = tmp_path / "MATLAB"
    root.mkdir()
    (root / "VersionInfo.xml").write_text("<root><release>R2024b</release></root>")
    monkeypatch.setattr(check, "matlab_root", lambda: root)
    monkeypatch.setattr(check, "matlab_engine_environment", lambda root: {})
    monkeypatch.setattr(check, "ensure_torch", lambda *args: None)
    profiles = []
    monkeypatch.setattr(check, "ensure_requirements",
                        lambda python, profile, flag: profiles.append((profile.name, flag)))
    monkeypatch.setattr(check, "start_matlab", lambda *args: pytest.fail("MATLAB started"))
    report_path = tmp_path / "environment.json"
    monkeypatch.setattr(sys, "argv", ["check_environment", "--report", str(report_path)]
                        + (["--check"] if read_only else []))

    def fail_probe(python, report):
        assert python == sys.executable
        assert report == tmp_path / "navigation-environment.json"
        assert profiles == [("requirements-closed-loop.txt", read_only),
                            ("requirements-matlab.txt", read_only)]
        raise RuntimeError("broken numerical runtime")

    monkeypatch.setattr(check, "probe_navigation", fail_probe)
    with pytest.raises(RuntimeError, match="broken numerical runtime"):
        check.main()
    assert not report_path.exists()
