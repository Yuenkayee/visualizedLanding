"""C++ runtime selection without touching the host's libraries or package manager."""

import json
import os
from types import SimpleNamespace

import pytest

from shell import closed_loop_dependencies as deploy


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    conda = tmp_path / "conda"
    library = conda / "lib/libstdc++.so.6"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"GLIBCXX_3.4.29\x00")
    system = tmp_path / "system/libstdc++.so.6"
    system.parent.mkdir()
    system.write_bytes(deploy.CPP_SYMBOL)
    matlab = tmp_path / "MATLAB"
    (matlab / "bin/glnxa64").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(deploy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(deploy.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(deploy, "SYSTEM_CPP_LIBRARY", system)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        # A venv can share its base Conda runtime; inspect both prefixes.
        return SimpleNamespace(returncode=0, stdout=json.dumps([str(tmp_path / "venv"), str(conda)]), stderr="")

    monkeypatch.setattr(deploy, "run", run)
    monkeypatch.setattr(deploy.subprocess, "run", run)
    return SimpleNamespace(conda=conda, library=library, system=system, matlab=matlab, calls=calls)


def test_old_conda_runtime_is_validated_in_fresh_child_and_preserves_preloads(runtime, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/existing/one.so:/existing/two.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cuda/lib64")
    result = deploy.matlab_cpp_runtime("/selected/python", runtime.matlab, check=True)
    assert result["status"] == "preload"
    assert result["outdated_libraries"] == [str(runtime.library)]
    assert result["ld_preload"] == f"{runtime.system}:/existing/one.so:/existing/two.so"
    command, options = runtime.calls[-1]
    assert command[0] == "/selected/python"
    assert options["env"]["LD_PRELOAD"] == result["ld_preload"]
    assert options["env"]["LD_LIBRARY_PATH"] == f"{runtime.matlab}/bin/glnxa64:/cuda/lib64"
    assert os.environ["LD_PRELOAD"] == "/existing/one.so:/existing/two.so"
    assert runtime.library.read_bytes() == b"GLIBCXX_3.4.29\x00"


def test_compatible_conda_needs_no_preload_or_probe(runtime):
    runtime.library.write_bytes(deploy.CPP_SYMBOL)
    assert deploy.matlab_cpp_runtime("python", runtime.matlab) == {"status": "default"}
    assert len(runtime.calls) == 1


def test_no_private_runtime_needs_no_override(runtime):
    runtime.library.unlink()
    assert deploy.matlab_cpp_runtime("python", runtime.matlab) == {"status": "default"}


def test_symbol_version_match_has_boundary(runtime):
    runtime.system.write_bytes(b"GLIBCXX_3.4.300\x00")
    assert not deploy.supports_matlab_cpp(runtime.system)


def test_check_only_never_updates_old_system_library(runtime, monkeypatch):
    runtime.system.write_bytes(b"GLIBCXX_3.4.29\x00")
    monkeypatch.setattr(deploy, "privilege", lambda: pytest.fail("read-only attempted apt"))
    with pytest.raises(RuntimeError, match="rerun without --check"):
        deploy.matlab_cpp_runtime("python", runtime.matlab, check=True)


def test_installer_updates_system_runtime_and_rechecks_symbol(runtime, monkeypatch):
    runtime.system.write_bytes(b"old")
    original = deploy.run
    monkeypatch.setattr(deploy, "privilege", lambda: [])

    def install(command, **kwargs):
        result = original(command, **kwargs)
        if "install" in command:
            assert "libstdc++6" in command and "libgcc-s1" in command
            runtime.system.write_bytes(deploy.CPP_SYMBOL)
        return result

    monkeypatch.setattr(deploy, "run", install)
    assert deploy.matlab_cpp_runtime("python", runtime.matlab)["status"] == "preload"
    assert any(command == ["apt-get", "update"] for command, _ in runtime.calls)


def test_failed_upgrade_cannot_claim_compatibility(runtime, monkeypatch):
    runtime.system.write_bytes(b"old")
    monkeypatch.setattr(deploy, "privilege", lambda: [])
    with pytest.raises(RuntimeError, match="still lacks GLIBCXX_3.4.30"):
        deploy.matlab_cpp_runtime("python", runtime.matlab)


def test_failed_native_import_blocks_preload_selection(runtime, monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout="", stderr="CXXABI mismatch"))
    with pytest.raises(RuntimeError, match="compatibility probe failed.*",):
        deploy.matlab_cpp_runtime("python", runtime.matlab)


def test_selected_preload_is_not_duplicated(runtime, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", f"{runtime.system} /existing.so")
    result = deploy.matlab_cpp_runtime("python", runtime.matlab)
    assert result["ld_preload"] == f"{runtime.system}:/existing.so"


def test_macos_does_not_probe_or_modify_runtime(monkeypatch):
    monkeypatch.setattr(deploy.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(deploy, "run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    assert deploy.matlab_cpp_runtime("python", "/Applications/MATLAB_R2024b.app") == {"status": "not_applicable"}
