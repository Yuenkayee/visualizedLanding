"""Render setup, ABI isolation, device failure handling and dataset publication."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from visual_modeling.blender.cycles_device import configure_cycles
from visual_training.data import blender_runtime as runtime
from visual_training.data import generate_multi_camera_dataset as dataset


@pytest.mark.parametrize("options", [["--plan-only"], ["--renderer", "cpu"]])
def test_plan_and_cpu_do_not_require_blender(monkeypatch, options):
    calls = []
    monkeypatch.setattr(runtime, "ensure_dependencies", lambda *a: calls.append(a))
    monkeypatch.setattr(runtime, "run", lambda c, **kw: calls.append(c))
    monkeypatch.setattr(runtime, "portable_blender", lambda *a: pytest.fail("download"))
    runtime.main([*options, "--output", "/tmp/data with spaces"])
    assert calls[0] == (sys.executable,)
    assert calls[1][:4] == [sys.executable, "-u", "-m", runtime.MODULE]
    assert "/tmp/data with spaces" in calls[1]


def test_blender_launch_isolates_python_and_propagates_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "portable_blender", lambda p: "/portable/blender")
    monkeypatch.setattr(
        runtime, "blender_info", lambda p: {"executable": "/embedded/python3.11"}
    )
    monkeypatch.setenv("PYTHONPATH", "/conda/lib/python3.12/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/conda")
    monkeypatch.setenv("LANDING_SITE_PACKAGES", "/old/venv")
    installed = []
    monkeypatch.setattr(runtime, "ensure_dependencies", lambda *a: installed.append(a))

    def fail(command, **kwargs):
        assert command[command.index("--python-exit-code") + 1] == "1"
        assert command[command.index("--cycles-device") + 1] == "CUDA"
        assert command[command.index("--gpu-index") + 1] == "1"
        assert "--setup-only" in command
        env = kwargs["env"]
        assert not {"PYTHONHOME", "PYTHONPATH", "LANDING_SITE_PACKAGES"} & env.keys()
        assert env["LANDING_RENDER_SITE"] == str(installed[0][1])
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(runtime, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        runtime.main(["--setup-only", "--cycles-device", "CUDA", "--gpu-index", "1"])
    assert error.value.returncode == 17
    assert installed[0][0] == "/embedded/python3.11"


@pytest.mark.parametrize(
    "blender,python,valid",
    [
        ([4, 5, 3], [3, 11, 13], True),
        ([4, 2, 0], [3, 11, 9], False),
        ([4, 5, 3], [3, 12, 3], False),
    ],
)
def test_embedded_version_guard(monkeypatch, blender, python, valid):
    info = dict(blender=blender, python=python, executable="/embedded/python")
    monkeypatch.setattr(
        runtime,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="Blender log\nLANDING_BLENDER_INFO="
            + json.dumps(info)
            + "\nBlender quit\n"
        ),
    )
    if valid:
        assert runtime.blender_info("blender") == info
    else:
        with pytest.raises(RuntimeError, match="Python 3.11"):
            runtime.blender_info("blender")


def test_install_targets_embedded_python_and_rechecks_imports(monkeypatch, tmp_path):
    checks = iter([1, 0])
    monkeypatch.setattr(
        runtime,
        "dependency_check",
        lambda *a: SimpleNamespace(returncode=next(checks), stdout="OK"),
    )
    calls = []
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda c, **kw: (
            calls.append([str(x) for x in c]) or SimpleNamespace(returncode=0)
        ),
    )
    site = tmp_path / "render site"
    runtime.ensure_dependencies("/blender/python3.11", site)
    install = calls[1]
    assert install[:6] == [
        sys.executable,
        "-m",
        "pip",
        "--python",
        "/blender/python3.11",
        "install",
    ]
    assert install[install.index("--target") + 1] == str(site)
    assert str(runtime.LOCK) in install
    assert "--only-binary=:all:" in install


def test_download_retries_truncated_archive_and_checks_cache(monkeypatch, tmp_path):
    payload = b"verified archive"
    monkeypatch.setattr(runtime, "SHA256", hashlib.sha256(payload).hexdigest())
    responses = iter(
        [runtime.http.client.IncompleteRead(b"partial"), b"truncated", payload]
    )

    def response(*args, **kwargs):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return io.BytesIO(item)

    monkeypatch.setattr(runtime.urllib.request, "urlopen", response)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    path = tmp_path / "blender.tar.xz"
    runtime.download_archive(path)
    assert path.read_bytes() == payload
    runtime.download_archive(path)  # Uses checksum-verified cache; no further request.
    assert not path.with_suffix(".xz.part").exists()


def test_bad_archive_never_published(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad")
    )
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    path = tmp_path / "blender.tar.xz"
    with pytest.raises(RuntimeError, match="SHA256"):
        runtime.download_archive(path)
    assert not path.exists()


def test_archive_extraction_rejects_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime, "download_archive", lambda p: None)
    with tarfile.open(tmp_path / runtime.ARCHIVE, "w:xz") as archive:
        member = tarfile.TarInfo("../../escaped")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        runtime.portable_blender(tmp_path)
    assert not (tmp_path.parent / "escaped").exists()


def fake_bpy(monkeypatch):
    devices = [
        SimpleNamespace(type=t, id=str(i), name=f"device-{i}", use=True)
        for i, t in enumerate(["CPU", "CUDA", "OPTIX", "OPTIX"])
    ]
    prefs = SimpleNamespace(devices=devices, refresh_devices=lambda: None)
    scene = SimpleNamespace(render=SimpleNamespace(), cycles=SimpleNamespace())
    bpy = SimpleNamespace(
        context=SimpleNamespace(
            scene=scene,
            preferences=SimpleNamespace(
                addons={"cycles": SimpleNamespace(preferences=prefs)}
            ),
        ),
        app=SimpleNamespace(version_string="4.5.3"),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    return prefs, scene


def test_cycles_uses_only_selected_gpu(monkeypatch):
    prefs, scene = fake_bpy(monkeypatch)
    report = configure_cycles("OPTIX", 1)
    assert scene.cycles.device == "GPU"
    assert [d.use for d in prefs.devices] == [False, False, False, True]
    assert report["devices"] == ["device-3"]
    configure_cycles("CPU")
    assert scene.cycles.device == "CPU"


def test_cycles_does_not_silently_fall_back(monkeypatch):
    fake_bpy(monkeypatch)
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        configure_cycles("CUDA", 1)


def small_config():
    config = dataset.load_config("configs/multi_camera_dataset.yaml")
    config.update(sorties_per_weather=3, frames_per_stage=2, trajectory_hz=4)
    config["camera"].update(width=80, height=60)
    config["weather"] = {"clear": config["weather"]["clear"]}
    for key in ("approach_duration_s", "hover_duration_s", "descent_duration_s"):
        config["mission"][key] = [1, 1]
    return config


def test_partial_camera_failure_does_not_publish_splits(monkeypatch, tmp_path):
    render = dataset.render_dataset
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            assert not (tmp_path / "splits").exists()
            raise RuntimeError("second camera failed")
        return render(*args, **kwargs)

    monkeypatch.setattr(dataset, "render_dataset", fail_second)
    with pytest.raises(RuntimeError, match="second camera"):
        dataset.generate(small_config(), tmp_path, "cpu")
    assert list((tmp_path / "rendered").rglob("*.png"))
    assert not (tmp_path / "splits").exists()


def test_complete_dataset_is_readable_and_protected(tmp_path):
    from visual_training.data.h_dataset import HDataset

    report = dataset.generate(small_config(), tmp_path, "cpu")
    assert report["split_image_counts"] == dict(train=24, val=24, test=24)
    for split in ("train", "val", "test"):
        loaded = HDataset(
            tmp_path / "splits" / f"{split}.json", tmp_path, image_size=32
        )
        assert len(loaded) == 24
        assert loaded[0]["image"].shape == (3, 32, 32)
    assert not (tmp_path / ".splits_pending").exists()
    with pytest.raises(FileExistsError, match="new --output"):
        dataset.generate(small_config(), tmp_path, "cpu", preview=True)


def test_failed_gpu_probe_never_starts_generator(monkeypatch, tmp_path):
    from visual_modeling.blender import cycles_device
    from visual_training.data import blender_multi_camera_entry as entry

    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--cycles-device", "OPTIX"])
    monkeypatch.setenv("LANDING_RENDER_SITE", str(tmp_path))

    def failed_probe(*args):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(cycles_device, "smoke_render", failed_probe)
    monkeypatch.setattr(
        dataset, "main", lambda *a, **kw: pytest.fail("generated after failed probe")
    )
    with pytest.raises(RuntimeError, match="kernel failure"):
        entry.main()
