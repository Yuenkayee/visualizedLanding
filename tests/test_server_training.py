import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shell.server_dependencies import requirements
from visual_training.data.build_dataset import build_dataset
from visual_training.evaluate import evaluate
from visual_training.models.losses import segmentation_keypoint_loss
from visual_training.server_train import (
    execute_job,
    tmux_command,
    validate_data,
    parser,
    build_config,
)
from visual_training.train import train, choose_batch, fit_shared_memory, seed_worker
from visual_training.data.training_loader import TrainingDataLoader


class OnlyFirstBatchDataset(torch.utils.data.Dataset):
    """Fail if prefetch reads beyond the requested short-run batch."""

    augmentation = None

    def __len__(self):
        return 20

    def __getitem__(self, index):
        if index >= 2:
            raise RuntimeError("prefetched beyond the requested batch limit")
        return index


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    build_dataset(root, sequences=3, frames=4)
    return root


def tiny_config(dataset, tmp_path):
    return dict(
        seed=42,
        data_root=str(dataset),
        image_size=32,
        batch_size=2,
        model=dict(channels=4),
        epochs=2,
        device="cpu",
        threads=1,
        output=str(tmp_path / "weights"),
        log_dir=str(tmp_path / "logs"),
        max_train_batches=1,
        max_val_batches=1,
        num_workers=2,
        persistent_workers=True,
        channels_last=True,
        log_every=1,
    )


def test_pretrain_updates_weights_but_saves_logs_only(dataset, tmp_path, monkeypatch):
    config = tiny_config(dataset, tmp_path)
    config["save_checkpoints"] = False
    steps = []
    original = torch.optim.AdamW.step

    def step(self, *args, **kwargs):
        before = self.param_groups[0]["params"][0].detach().clone()
        result = original(self, *args, **kwargs)
        steps.append(not torch.equal(before, self.param_groups[0]["params"][0]))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    assert train(config) is None
    assert steps == [True, True]
    assert not Path(config["output"]).exists()
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("history.json"))
    metrics = [
        json.loads(line)
        for line in (tmp_path / "logs/metrics.jsonl").read_text().splitlines()
    ]
    epochs = [row for row in metrics if row["event"] == "epoch"]
    assert len(epochs) == 2 and all(
        row["phases"]["train"]["samples"] == 2 for row in epochs
    )
    assert all(0 <= row["val_mask_iou"] <= 1 for row in epochs)
    assert metrics[-1]["event"] == "training_complete"


def test_full_checkpoint_can_be_loaded_by_existing_evaluator(dataset, tmp_path):
    config = tiny_config(dataset, tmp_path)
    config.update(epochs=1, num_workers=0)
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(dict(mode="full", config=config)))
    execute_job(job_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "complete" and status["exit_code"] == 0
    payload = torch.load(status["checkpoint"], weights_only=True)
    assert payload["epoch"] == 1 and payload["optimizer"]["state"]
    result = evaluate(status["checkpoint"], dataset / "splits/test.json", dataset, 32)
    assert np.isfinite(result["mask_iou"])
    assert (tmp_path / "weights/last.pt").is_file()
    assert (tmp_path / "weights/history.json").is_file()


def test_failed_training_keeps_failure_status(tmp_path):
    config = tiny_config(tmp_path / "missing", tmp_path)
    config.update(save_checkpoints=False, num_workers=0)
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(dict(mode="pretrain", config=config)))
    with pytest.raises(FileNotFoundError):
        execute_job(job_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "weights").exists()


def test_dataset_gate_rejects_leakage_and_generation_test(dataset):
    assert validate_data(dataset) == dict(train=4, val=4, test=4)
    (dataset / "server_run.json").write_text(
        json.dumps(dict(status="complete", mode="test"))
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_data(dataset)
    (dataset / "SUCCESS.json").write_text("{}")
    with pytest.raises(ValueError, match="90-frame"):
        validate_data(dataset, formal=True)
    paths = json.loads((dataset / "splits/train.json").read_text())
    (dataset / "splits/val.json").write_text(json.dumps(paths))
    with pytest.raises(ValueError, match="Repeated"):
        validate_data(dataset)


def test_batch_probe_selects_throughput_with_memory_headroom(monkeypatch):
    import visual_training.train as module

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1000, 1000))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def probe(config, dataset, device, batch):
        if batch == 16:
            raise torch.cuda.OutOfMemoryError("test")
        return dict(
            batch_size=batch,
            images_per_s={8: 150, 4: 100}[batch],
            peak_reserved_bytes=batch * 100,
        )

    monkeypatch.setattr(module, "probe_batch", probe)
    events = []
    assert (
        choose_batch(
            dict(batch_size="auto", max_batch_size=16),
            None,
            torch.device("cuda"),
            events.append,
        )
        == 8
    )
    assert events[0]["status"] == "out_of_memory"
    assert events[-1]["event"] == "batch_selected"


def test_shared_memory_budget_and_independent_augmentation(monkeypatch):
    import visual_training.train as module

    monkeypatch.setattr(module.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda path: SimpleNamespace(free=64 * 1024**2)
    )
    config = dict(num_workers=8, image_size=512, prefetch_factor=2)
    events = []
    fit_shared_memory(config, 16, events.append)
    assert config["num_workers"] == 0 and events
    augmentation = SimpleNamespace(rng=None)
    monkeypatch.setattr(
        module,
        "get_worker_info",
        lambda: SimpleNamespace(dataset=SimpleNamespace(augmentation=augmentation)),
    )
    monkeypatch.setattr(torch, "initial_seed", lambda: 123)
    seed_worker(0)
    first = augmentation.rng.random()
    monkeypatch.setattr(torch, "initial_seed", lambda: 124)
    seed_worker(1)
    assert first != augmentation.rng.random()


def test_bf16_heatmap_loss_remains_finite_in_fp32():
    heatmaps = torch.randn(2, 4, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    prediction = dict(
        heatmap_logits=heatmaps,
        mask_logits=heatmaps[:, :1],
        keypoints=torch.rand(2, 4, 2, requires_grad=True),
    )
    target = dict(
        mask=torch.zeros(2, 1, 64, 64),
        keypoints=torch.rand(2, 4, 2),
        visibility=torch.ones(2, 4, dtype=torch.bool),
    )
    loss, metrics = segmentation_keypoint_loss(prediction, target, return_metrics=False)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.isfinite(heatmaps.grad).all() and metrics == {}


def test_tmux_command_preserves_paths_and_runtime_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    job = tmp_path / "job with spaces.json"
    log = tmp_path / "train with spaces.log"
    command = tmux_command("landing_train", job, log)
    assert command[:5] == ["tmux", "new-session", "-d", "-s", "landing_train"]
    argv = shlex.split(command[-1])
    assert "CUDA_VISIBLE_DEVICES=0" in argv
    assert argv[-3:] == [sys.executable, str(job), str(log)]


def test_shell_worker_preserves_failure_exit_and_log(tmp_path):
    config = tiny_config(tmp_path / "missing", tmp_path)
    config.update(save_checkpoints=False, num_workers=0)
    job = tmp_path / "job with spaces.json"
    job.write_text(json.dumps(dict(mode="pretrain", config=config)))
    log = tmp_path / "error.log"
    process = subprocess.run(
        ["bash", "shell/run_training_job.sh", sys.executable, str(job), str(log)],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "FileNotFoundError" in log.read_text()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"


def test_modes_and_training_dependency_profile(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "data_root: data/landing\nbatch_size: auto\nepochs: 40\nmax_train_batches: 1\n"
    )
    args = parser().parse_args(["pretrain", "--config", str(config)])
    c = build_config(args, tmp_path)
    assert c["epochs"] == 2 and c["max_train_batches"] == 50
    assert not c["save_checkpoints"] and "output" not in c
    args = parser().parse_args(
        ["full", "--config", str(config), "--output", str(tmp_path / "out")]
    )
    c = build_config(args, tmp_path)
    assert c["epochs"] == 40 and "max_train_batches" not in c and c["save_checkpoints"]
    with pytest.raises(ValueError, match="not empty"):
        build_config(args, tmp_path)
    profile = {
        r.name: str(r.specifier)
        for r in requirements("requirements-training-server.txt")
    }
    assert profile["torch"] == "==2.8.0" and "bpy" not in profile


@pytest.mark.parametrize("persistent", [True, False])
def test_short_loader_limits_prefetch_and_closes_workers(persistent):
    from visual_training.train import make_loader

    config = dict(
        num_workers=2,
        persistent_workers=persistent,
        prefetch_factor=4,
        max_val_batches=1,
    )
    loader = make_loader(OnlyFirstBatchDataset(), config, 2, False, torch.device("cpu"))
    workers = []
    try:
        assert len(loader) == 1
        for _ in range(2):
            iterator = iter(loader)
            workers.extend(iterator._workers)
            assert [item.tolist() for item in iterator] == [[0, 1]]
    finally:
        loader.close()
    assert all(not w.is_alive() and w.exitcode == 0 for w in workers)
    loader.close()  # Idempotent; no fallback to destructor cleanup.


def test_limited_sampler_keeps_shuffle_and_final_partial_batch():
    from visual_training.train import make_loader

    dataset = torch.utils.data.TensorDataset(torch.arange(11))
    config = dict(max_train_batches=2, max_val_batches=10, seed=42)
    loader = make_loader(dataset, config, 4, True, torch.device("cpu"))
    first = torch.cat([sample[0] for sample in loader]).tolist()
    second = torch.cat([sample[0] for sample in loader]).tolist()
    assert len(first) == len(second) == 8 and first != second
    assert len(set(first)) == 8
    validation = make_loader(dataset, config, 4, False, torch.device("cpu"))
    assert len(validation) == 3
    assert [sample[0].tolist() for sample in validation] == [
        list(range(4)),
        list(range(4, 8)),
        [8, 9, 10],
    ]


def test_cleanup_error_fails_job_without_completion_event(
    dataset, tmp_path, monkeypatch
):
    config = tiny_config(dataset, tmp_path)
    config.update(save_checkpoints=False, num_workers=0)
    job = tmp_path / "job.json"
    job.write_text(json.dumps(dict(mode="pretrain", config=config)))
    closed = []
    original = TrainingDataLoader.close

    def fail_first_close(self):
        original(self)
        closed.append(self)
        if len(closed) == 1:
            raise RuntimeError("injected worker shutdown failure")

    monkeypatch.setattr(TrainingDataLoader, "close", fail_first_close)
    with pytest.raises(RuntimeError, match="shutdown failure"):
        execute_job(job)
    assert len(closed) == 2  # Both pools must be cleaned even when the first fails.
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    metrics = (tmp_path / "logs/metrics.jsonl").read_text()
    assert (
        '"event": "epoch"' in metrics and '"event": "training_complete"' not in metrics
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA required to exercise pinned-memory thread shutdown",
            ),
        ),
    ],
)
def test_pretrain_process_exits_cleanly_after_completion(dataset, tmp_path, device):
    config = tiny_config(dataset, tmp_path)
    config.update(
        save_checkpoints=False,
        device=device,
        pin_memory=device == "cuda",
        prefetch_factor=4,
    )
    job = tmp_path / "job.json"
    job.write_text(json.dumps(dict(mode="pretrain", config=config)))
    process = subprocess.run(
        [sys.executable, "-m", "visual_training.server_train", "--run-job", str(job)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Exception ignored" not in process.stderr
    assert "Traceback" not in process.stderr and "Aborted" not in process.stderr
    assert (
        json.loads(process.stdout.strip().splitlines()[-1])["event"]
        == "training_complete"
    )
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "complete"
    assert not (tmp_path / "weights").exists()


def test_training_failure_reaps_all_started_workers(dataset, tmp_path, monkeypatch):
    import visual_training.train as module

    config = tiny_config(dataset, tmp_path)
    config["save_checkpoints"] = False
    closed_workers = []
    original = TrainingDataLoader.close

    def record_close(self):
        iterator = getattr(self, "_training_iterator", None)
        if iterator is not None:
            closed_workers.extend(iterator._workers)
        original(self)

    def fail_loss(*args, **kwargs):
        raise ValueError("injected training failure")

    monkeypatch.setattr(module, "segmentation_keypoint_loss", fail_loss)
    monkeypatch.setattr(TrainingDataLoader, "close", record_close)
    with pytest.raises(ValueError, match="training failure"):
        train(config)
    assert len(closed_workers) == 2
    assert all(not worker.is_alive() for worker in closed_workers)


def test_close_rejects_worker_abort_even_if_torch_shutdown_does_not_raise():
    worker = SimpleNamespace(
        pid=123, exitcode=-6, join=lambda timeout: None, is_alive=lambda: False
    )
    iterator = SimpleNamespace(_workers=[worker], _shutdown_workers=lambda: None)
    loader = TrainingDataLoader(OnlyFirstBatchDataset(), num_workers=1)
    loader._training_iterator = iterator
    with pytest.raises(RuntimeError, match=r"123, -6"):
        loader.close()
    assert loader._training_iterator is None and loader._iterator is None
