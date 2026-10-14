import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from shell.server_dependencies import audit, requirements, ensure_torch
from visual_modeling.blender.cycles_device import configure_cycles
from visual_training.data.generate_landing_dataset import plan_dataset, render_dataset
from visual_training.data.generate_server_dataset import run_workers, merge_shards
from visual_training.data.validate_server_dataset import validate_dataset


def small_config():
    config = yaml.safe_load(Path("configs/landing_dataset.yaml").read_text())
    config.update(sorties_per_weather=3, frames_per_stage=2, trajectory_hz=4)
    config["camera"].update(width=80, height=60)
    for name in ("approach_duration_s", "hover_duration_s", "descent_duration_s"):
        config["mission"][name] = [1, 2]
    config["weather"] = {
        name: config["weather"][name] for name in ("clear", "night_fog")
    }
    return config


def test_parallel_dataset_matches_serial_and_training_contract(tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    config = small_config()
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    serial_plan = plan_dataset(config, serial)
    parallel_plan = plan_dataset(config, parallel, workers=2)
    for a, b in zip(serial_plan, parallel_plan):
        assert {k: v for k, v in a.items() if k != "trajectory"} == {
            k: v for k, v in b.items() if k != "trajectory"
        }
        assert json.loads(Path(a["trajectory"]).read_text()) == json.loads(
            Path(b["trajectory"]).read_text()
        )
    serial_report = render_dataset(config, serial_plan, serial, "cpu")
    run_workers(sys.executable, parallel, 2, 1, "CPU", renderer="cpu")
    assert not (parallel / "dataset_report.json").exists()
    assert not (parallel / "splits").exists()
    report, runtimes = merge_shards(config, parallel_plan, parallel, 2, renderer="cpu")
    assert len(runtimes) == 2
    assert report["frames"] == serial_report["frames"] == 36
    assert report["by_stage_weather"] == serial_report["by_stage_weather"]
    for source in (serial / "annotations").rglob("*.json"):
        relative = source.relative_to(serial)
        row = json.loads(source.read_text())
        assert row == json.loads((parallel / relative).read_text())
        assert np.array_equal(
            cv2.imread(str(serial / row["image"])),
            cv2.imread(str(parallel / row["image"])),
        )
    validation = validate_dataset(parallel, all_images=True)
    assert validation["splits"] == dict(train=12, val=12, test=12)
    assert validation["image_lidar_samples_decoded"] == 36
    assert (parallel / "preview.png").is_file()
    # Coordinator rejects a successful process with incomplete/corrupt output.
    fragment = parallel / "workers/00.json"
    shard = json.loads(fragment.read_text())
    shard["summary"].pop()
    fragment.write_text(json.dumps(shard))
    with pytest.raises(ValueError, match="frame summaries"):
        merge_shards(config, parallel_plan, parallel, 2, renderer="cpu")


def test_worker_failure_does_not_publish_dataset(tmp_path):
    # A malformed plan makes the child fail; the parent must propagate failure.
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Render worker"):
        run_workers(sys.executable, tmp_path, 2, 1, "CPU", renderer="cpu")
    assert not (tmp_path / "dataset_report.json").exists()


def test_cycles_selects_only_requested_gpu_and_never_silently_falls_back(monkeypatch):
    cpu = SimpleNamespace(name="Xeon", type="CPU", use=True)
    gpu = SimpleNamespace(name="RTX 4090", type="OPTIX", use=False)
    other = SimpleNamespace(name="Other GPU", type="OPTIX", use=True)
    prefs = SimpleNamespace(devices=[cpu, gpu, other], refresh_devices=lambda: None)
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
    runtime = configure_cycles("OPTIX", threads=8)
    assert runtime["devices"] == ["RTX 4090"]
    assert gpu.use and not cpu.use and not other.use
    assert scene.cycles.device == "GPU" and scene.render.threads == 8
    assert scene.cycles.use_denoising and scene.cycles.denoiser == "OPTIX"
    prefs.devices = [cpu]
    with pytest.raises(RuntimeError, match="unavailable"):
        configure_cycles("OPTIX")


def test_server_profiles_and_dependency_version_checks(tmp_path, monkeypatch):
    from importlib import metadata
    import shell.server_dependencies as dependencies

    profile = {
        r.name: str(r.specifier) for r in requirements("requirements-server.txt")
    }
    assert profile["torch"] == "==2.8.0"
    assert "bpy" not in profile
    render = {
        r.name: str(r.specifier) for r in requirements("requirements-render-server.txt")
    }
    assert render["bpy"] == "==4.5.3" and "torch" not in render
    required = tmp_path / "requirements.txt"
    required.write_text("torch==2.8.0\nmissing==1.0\n")

    def distribution(name):
        if name == "torch":
            return SimpleNamespace(version="2.8.0+cu128", requires=["numpy>=1.26"])
        if name == "numpy":
            return SimpleNamespace(version="1.25.0", requires=[])
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(dependencies.metadata, "distribution", distribution)
    problems = audit(required)
    assert len(problems) == 2
    assert any("missing==1.0" in problem for problem in problems)
    assert any("numpy>=1.26" in problem for problem in problems)


def test_torch_correct_build_is_preserved_and_wrong_cuda_is_rejected(monkeypatch):
    import shell.server_dependencies as dependencies

    result = SimpleNamespace(returncode=0, stdout="2.8.0 12.8\n")
    monkeypatch.setattr(dependencies.subprocess, "run", lambda *a, **k: result)
    installs = []
    monkeypatch.setattr(
        dependencies, "run", lambda command, **kwargs: installs.append(command)
    )
    ensure_torch(check=False)
    assert not installs
    result.stdout = "2.8.0 12.6\n"
    with pytest.raises(RuntimeError, match="CUDA 12.8"):
        ensure_torch(check=True)
    ensure_torch(check=False)
    assert "torch==2.8.0+cu128" in installs[0]
    assert "https://download.pytorch.org/whl/cu128" in installs[0]
