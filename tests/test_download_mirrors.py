"""Download routing contracts; tests do not download or change system sources."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shell import closed_loop_dependencies as deploy
from shell import server_dependencies as server


def mirror_environment(profile, **variables):
    keys = ("LANDING_DOWNLOAD_PROFILE", "LANDING_PYPI_INDEX", "LANDING_PYTHON_BINARY_MIRROR",
            "LANDING_PYTHON_SOURCE_MIRROR", "LANDING_APT_MIRROR", "LANDING_TORCH_INDEX")
    env = {k: v for k, v in os.environ.items() if k not in keys}
    env.update(LANDING_DOWNLOAD_PROFILE=profile, **variables)
    result = subprocess.run(
        ["bash", "-c", 'source "$1" && "$2" -c "import json,os; print(json.dumps(dict(os.environ)))"',
         "mirror-test", str(deploy.ROOT / "shell/download_mirrors.sh"), sys.executable],
        env=env, capture_output=True, text=True,
    )
    return result


def test_china_profile_routes_packages_and_interpreters_separately():
    result = mirror_environment("china", PIP_INDEX_URL="https://pypi.org/simple",
                                UV_DEFAULT_INDEX="https://pypi.org/simple",
                                UV_PYTHON_INSTALL_MIRROR="https://old.example/binaries")
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    assert env["PIP_INDEX_URL"] == env["UV_DEFAULT_INDEX"] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    assert env["UV_PYTHON_INSTALL_MIRROR"] == "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"
    assert env["LANDING_PYTHON_SOURCE_MIRROR"] == "https://mirrors.tuna.tsinghua.edu.cn/python"
    assert env["LANDING_APT_MIRROR"] == "https://mirrors.tuna.tsinghua.edu.cn/ubuntu"


def test_default_profile_keeps_existing_configuration():
    result = mirror_environment("default", PIP_INDEX_URL="https://custom.example/simple",
                                UV_PYTHON_INSTALL_MIRROR="https://custom.example/python")
    env = json.loads(result.stdout)
    assert env["PIP_INDEX_URL"] == "https://custom.example/simple"
    assert env["UV_PYTHON_INSTALL_MIRROR"] == "https://custom.example/python"
    assert "LANDING_APT_MIRROR" not in env


def test_explicit_overrides_are_forwarded():
    result = mirror_environment("china", LANDING_PYPI_INDEX="https://custom.example/simple",
                                LANDING_PYTHON_BINARY_MIRROR="https://custom.example/binaries")
    env = json.loads(result.stdout)
    assert env["PIP_INDEX_URL"] == env["UV_DEFAULT_INDEX"] == "https://custom.example/simple"
    assert env["UV_PYTHON_INSTALL_MIRROR"] == "https://custom.example/binaries"


def test_invalid_profile_stops_before_launch():
    result = mirror_environment("typo")
    assert result.returncode == 2
    assert "choose default or china" in result.stderr
    assert not result.stdout


def test_mirrored_apt_update_and_install_use_same_signed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "TOOLS", tmp_path)
    monkeypatch.setattr(deploy, "ubuntu", lambda: None)
    monkeypatch.setenv("LANDING_APT_MIRROR", "https://mirror.example/ubuntu/")
    monkeypatch.setattr(deploy, "privilege", lambda: ["sudo", "-n"])
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    calls = []
    monkeypatch.setattr(deploy, "run", lambda command: calls.append(command))
    deploy.ensure_apt(["libssl-dev"])
    source = tmp_path / "apt/sources.list"
    lines = source.read_text().splitlines()
    assert len(lines) == 4
    for suite in ("jammy", "jammy-updates", "jammy-backports", "jammy-security"):
        assert any(f"https://mirror.example/ubuntu {suite} " in line for line in lines)
    assert all("signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg" in line for line in lines)
    for command in calls:
        assert command[:2] == ["sudo", "-n"]
        assert f"Dir::Etc::sourcelist={source}" in command
        assert "Dir::Etc::sourceparts=-" in command
    assert calls[0][-1] == "update" and calls[1][-1] == "libssl-dev"


def test_default_apt_does_not_write_sources(tmp_path, monkeypatch):
    monkeypatch.delenv("LANDING_APT_MIRROR", raising=False)
    monkeypatch.setattr(deploy, "TOOLS", tmp_path)
    assert deploy.apt_command("update") == ["apt-get", "update"]
    assert not list(tmp_path.iterdir())


def test_read_only_mirror_check_does_not_create_apt_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "TOOLS", tmp_path)
    monkeypatch.setenv("LANDING_APT_MIRROR", "https://mirror.example/ubuntu")
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(RuntimeError, match="Missing Ubuntu packages"):
        deploy.ensure_apt(["libssl-dev"], check=True)
    assert not list(tmp_path.iterdir())


def test_source_build_download_uses_selected_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "TOOLS", tmp_path)
    monkeypatch.setenv("LANDING_PYTHON_SOURCE_MIRROR", "https://mirror.example/python/")
    monkeypatch.setattr(deploy, "python_matches", lambda *a: False)
    monkeypatch.setattr(deploy, "ensure_apt", lambda *a: None)
    monkeypatch.setattr(deploy, "run", lambda *a, **k: None)
    downloads = []
    monkeypatch.setattr(deploy, "download", lambda url, target: downloads.append((url, target)))
    binary = deploy.source_python("3.11")
    assert downloads[0][0] == "https://mirror.example/python/3.11.14/Python-3.11.14.tgz"
    assert binary == tmp_path / "python/3.11.14/bin/python3.11"


def test_uv_receives_binary_mirror_and_seed_index(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_PYTHON_INSTALL_MIRROR", "https://binary.example/releases")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://packages.example/simple")
    monkeypatch.setattr(deploy, "ensure_uv", lambda: ["uv-test"])
    ready = False
    monkeypatch.setattr(deploy, "python_matches", lambda *a: ready)

    def run(command, **kwargs):
        nonlocal ready
        assert command[:2] == ["uv-test", "venv"]
        assert kwargs["env"]["UV_PYTHON_INSTALL_MIRROR"] == "https://binary.example/releases"
        assert kwargs["env"]["UV_DEFAULT_INDEX"] == "https://packages.example/simple"
        ready = True

    monkeypatch.setattr(deploy, "run", run)
    deploy.ensure_venv(tmp_path / "render", "3.11")


@pytest.mark.parametrize("mirror", [None, "https://torch.example/cu128"])
def test_torch_keeps_cuda_build_and_uses_only_its_dedicated_index(monkeypatch, mirror):
    monkeypatch.delenv("LANDING_TORCH_INDEX", raising=False)
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")
    if mirror:
        monkeypatch.setenv("LANDING_TORCH_INDEX", mirror)
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    calls = []
    monkeypatch.setattr(server, "run", lambda command: calls.append(command))
    server.ensure_torch(False)
    assert "torch==2.8.0+cu128" in calls[0]
    assert calls[0][-2:] == ["--index-url", mirror or server.TORCH_INDEX]


def test_correct_torch_is_reused_after_mirror_switch(monkeypatch):
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout="2.8.0 12.8\n"))
    monkeypatch.setattr(server, "run", lambda *a, **k: pytest.fail("unnecessary reinstall"))
    server.ensure_torch(False)
