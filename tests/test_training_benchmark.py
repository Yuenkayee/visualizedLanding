import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from visual_training import benchmark
from visual_training.benchmark_resources import summarize
from visual_training.data.build_dataset import build_dataset
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints


def test_candidates_include_baseline_and_deduplicate_ineffective_prefetch():
    config = {"batch_size": 16, "num_workers": 8, "threads": 8, "prefetch_factor": 2}
    rows = benchmark.candidates(config, [16, 32], [0, 8], [8], [2, 4])
    assert rows[0] == config
    assert len(rows) == 6
    assert all(row["prefetch_factor"] == 2 for row in rows if row["num_workers"] == 0)
    assert rows == benchmark.candidates(config, [16, 32], [0, 8], [8], [2, 4])


def test_sampler_covers_split_and_stops_at_prefetch_boundary():
    sampler = benchmark.BenchmarkSampler(5, 9, 42)
    samples = list(sampler)
    assert len(sampler) == len(samples) == 9
    assert set(samples[:5]) == set(range(5))
    assert len(set(samples[5:])) == 4
    assert samples == list(sampler)


def trial(candidate, seconds, status="ok", stage="confirm", batch=16):
    return {
        "candidate_id": candidate,
        "stage": stage,
        "status": status,
        "estimated_epoch_s": seconds,
        "settings": {
            "batch_size": batch,
            "num_workers": 8,
            "threads": 8,
            "prefetch_factor": 2,
        },
    }


def test_selection_requires_stable_confirmations_and_memory_margin():
    rows = [
        trial(0, 10),
        trial(0, 10.1),
        trial(1, 5),
        trial(1, 9),
        trial(2, 1, "oom"),
        trial(2, 1),
        trial(3, 2, "insufficient_headroom"),
        trial(3, 2),
        trial(4, 0.5, stage="screen"),
        trial(5, 0.5),
        trial(6, 9.9, batch=32),
        trial(6, 10, batch=32),
        trial(7, float("nan")),
        trial(7, 1),
    ]
    selected = benchmark.choose_recommendation(rows, 2, 0.15)
    # Similar throughput: keep the smaller batch rather than use extra memory.
    assert selected["candidate_id"] == 0
    assert benchmark.choose_recommendation([trial(0, 10)], 2, 0.15) is None


def test_telemetry_handles_missing_nvidia_values():
    assert summarize(
        [
            {"time_s": 0, "gpu_util_percent": 0},
            {"time_s": 1, "gpu_util_percent": 80, "gpu_power_w": None},
        ]
    ) == {"gpu_util_percent": {"mean": 40, "min": 0, "max": 80}}


def test_worker_reports_oom_without_touching_checkpoints(tmp_path, monkeypatch):
    def oom(_):
        raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")

    monkeypatch.setattr(benchmark, "run_trial", oom)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"result_path": str(tmp_path / "result.json")}))
    benchmark.worker(spec)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "oom"
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("workers", [0, 2])
def test_real_cpu_subprocess_uses_dataset_and_shared_training_step(tmp_path, workers):
    root = tmp_path / "data"
    build_dataset(root, sequences=3, frames=2)
    model = HSegmentationKeypoints(channels=4, predict_visibility=True)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": model.model_config}, checkpoint
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = {
        "data_root": str(root),
        "device": "cpu",
        "precision": "bf16",
        "threads": 2,
        "num_workers": workers,
        "multiprocessing_context": "spawn",
        "persistent_workers": True,
        "batch_size": 2,
        "image_size": 32,
        "model": model.model_config,
    }
    result = benchmark.launch_trial(
        {
            "config": config,
            "checkpoint": str(checkpoint),
            "warmup": 1,
            "steps": 2,
            "val_steps": 1,
            "memory_margin": 0.15,
        },
        tmp_path / "trial",
        timeout=90,
    )
    assert result["status"] == "ok", result
    assert result["train"]["images"] == 4
    assert result["val"]["images"] == 2
    assert result["train"]["images_per_second"] > 0
    assert result["train"]["cuda_step_mean_ms"] is None
    assert result["train"]["dataset_restarts"] > 0
    assert result["estimated_epoch_s"] > 0
    assert result["train"]["metrics"]["loss"] > 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    assert not list((tmp_path / "trial").glob("*.pt"))


@pytest.mark.parametrize("unstable_baseline", [False, True])
def test_search_confirms_baseline_and_exports_both_configs(
    tmp_path, monkeypatch, unstable_baseline
):
    root = tmp_path / "data"
    (root / "splits").mkdir(parents=True)
    for name in ("train", "val"):
        (root / f"splits/{name}.json").write_text("[]")
    checkpoint = tmp_path / "old_best.pt"
    checkpoint.write_bytes(b"untouched")
    original = {
        "batch_size": 16,
        "num_workers": 8,
        "threads": 8,
        "prefetch_factor": 2,
        "image_size": 512,
        "learning_rate": 0.0001,
        "epochs": 20,
        "require_init_checkpoint": True,
        "data_root": str(root),
        "device": "cuda:0",
        "output": "original_checkpoints",
        "lr_scheduler": {"name": "plateau"},
        "model": {"channels": 16},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(original))
    calls = []

    def measured(spec, directory, timeout):
        settings = spec["config"]
        calls.append((spec, directory))
        seconds = {(16, 8): 100, (16, 12): 70, (32, 8): 50, (32, 12): 45}[
            (settings["batch_size"], settings["num_workers"])
        ]
        if unstable_baseline and directory.name == "confirm_000_1":
            seconds = 200
        return {
            "status": "ok",
            "estimated_epoch_s": seconds,
            "train": {"images_per_second": 1000 / seconds},
        }

    monkeypatch.setattr(benchmark, "launch_trial", measured)
    args = argparse.Namespace(
        config=str(config_path),
        data_root=None,
        device=None,
        checkpoint=str(checkpoint),
        output=str(tmp_path / "results"),
        batch_sizes=[16, 32],
        workers=[8, 12],
        threads=None,
        prefetch_factors=[2],
        warmup=2,
        steps=3,
        val_steps=1,
        confirm_steps=12,
        memory_margin=0.15,
        trial_timeout=30,
        top_k=1,
        repeats=2,
        max_variation=0.15,
    )
    report = benchmark.run_benchmark(args)
    assert report["recommendation"]["settings"]["batch_size"] == 32
    assert report["recommendation_same_batch"]["settings"]["num_workers"] == 12
    if unstable_baseline:
        assert "speedup_over_baseline" not in report["recommendation"]
    else:
        assert report["recommendation"]["speedup_over_baseline"] == pytest.approx(
            100 / 45
        )
    assert (
        len(calls) == 4 + 3 * 2
    )  # screen + fastest / baseline / best at original batch
    for name in ("recommended.yaml", "recommended_same_batch.yaml"):
        exported = yaml.safe_load((Path(args.output) / name).read_text())
        assert exported["learning_rate"] == original["learning_rate"]
        assert exported["image_size"] == 512 and exported["epochs"] == 20
        assert exported["model"] == original["model"]
        assert exported["lr_scheduler"] == original["lr_scheduler"]
        assert exported["init_checkpoint"] == str(checkpoint)
        assert exported["output"] != original["output"]
    assert checkpoint.read_bytes() == b"untouched"
    assert yaml.safe_load(config_path.read_text()) == original
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(args)


def test_timeout_terminates_isolated_trial(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    processes = []

    def stalled(_command, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(benchmark.subprocess, "Popen", stalled)
    result = benchmark.launch_trial({}, tmp_path / "timeout", timeout=0.1)
    assert result["status"] == "timeout"
    assert processes[0].poll() is not None
