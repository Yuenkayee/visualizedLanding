import argparse
import json
import math
import random
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from visual_training.data.augmentations import PhotometricAugmentation
from visual_training.data.h_dataset import HDataset
from visual_training.metrics import EpochMetrics
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss


def initialize_worker(_):
    """Independent photometric noise per worker, including persistent workers."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)
    info = torch.utils.data.get_worker_info()
    if info.dataset.augmentation is not None:
        info.dataset.augmentation.rng = np.random.default_rng(seed)


def make_loader(dataset, config, training, seed):
    workers = int(config.get("num_workers", 0))
    if workers < 0 or config.get("batch_size", 8) <= 0:
        raise ValueError("invalid batch size or worker count")
    extra = {}
    if workers:
        prefetch = int(config.get("prefetch_factor", 2))
        if prefetch <= 0:
            raise ValueError("prefetch_factor must be positive")
        extra.update(
            persistent_workers=bool(config.get("persistent_workers", True)),
            prefetch_factor=prefetch,
        )
        if config.get("multiprocessing_context"):
            extra["multiprocessing_context"] = config["multiprocessing_context"]
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=training,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", False)),
        worker_init_fn=initialize_worker,
        generator=torch.Generator().manual_seed(seed),
        **extra,
    )


def training_device(config):
    device = torch.device(config.get("device", "cpu"))
    precision = config.get("precision", "fp32")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. RTX 5090 requires a compatible NVIDIA driver and requirements-rtx5090.txt (PyTorch cu128)."
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist")
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected CUDA GPU does not support BF16")
        torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", False))
        torch.backends.cudnn.allow_tf32 = bool(config.get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", False))
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable on this machine")
    elif device.type != "cpu":
        raise ValueError("supported devices: cpu, mps, cuda[:index]")
    if precision != "fp32" and (
        device.type not in ("cpu", "cuda")
        or (precision == "fp16" and device.type != "cuda")
    ):
        raise ValueError(
            "FP16 requires CUDA; BF16 requires CPU or CUDA; use FP32 for MPS"
        )
    return device, precision


def process_batch(
    model, sample, optimizer, scaler, device, precision, config, metrics, training
):
    """Shared by training and benchmarking; caller sets model/gradient mode."""
    sample = {
        k: v.to(
            device,
            non_blocking=device.type == "cuda" and config.get("pin_memory", False),
        )
        for k, v in sample.items()
    }
    if config.get("channels_last", False):
        sample["image"] = sample["image"].contiguous(memory_format=torch.channels_last)
    if training:
        optimizer.zero_grad(set_to_none=True)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with (
        torch.autocast(device_type=device.type, dtype=dtype)
        if precision != "fp32"
        else nullcontext()
    ):
        prediction = model(sample["image"])
    loss, components = segmentation_keypoint_loss(
        prediction, sample, metrics_as_tensors=True
    )
    if training:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
    metrics.update(prediction, sample, loss, components)


def run_epoch(
    model, loader, optimizer, scaler, device, precision, config, epoch, phase
):
    training = phase == "train"
    model.train(training)
    metrics = EpochMetrics(device, config.get("image_size", 256))
    log_every = int(config.get("log_every_batches", 100))
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for step, sample in enumerate(loader, 1):
            process_batch(
                model,
                sample,
                optimizer,
                scaler,
                device,
                precision,
                config,
                metrics,
                training,
            )
            if log_every > 0 and step % log_every == 0:
                print(
                    json.dumps(
                        dict(
                            event="batch_progress",
                            epoch=epoch,
                            phase=phase,
                            batch=step,
                            batches=len(loader),
                            **metrics.result(),
                            elapsed_s=time.perf_counter() - started,
                        )
                    ),
                    flush=True,
                )
    result = metrics.result()
    if not all(math.isfinite(result[key]) for key in metrics.losses):
        raise FloatingPointError(
            f"non-finite {phase} loss at epoch {epoch}; checkpoint not saved"
        )
    result["elapsed_s"] = time.perf_counter() - started
    return result


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_history(history, path):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2, allow_nan=False))
    temporary.replace(path)


def train(config):
    c = dict(config)
    if c.get("resume") and c.get("init_checkpoint"):
        raise ValueError("use only one of resume and init_checkpoint")
    checkpoint_path = c.get("resume") or c.get("init_checkpoint")
    if c.get("require_init_checkpoint") and not checkpoint_path:
        raise ValueError("this fine-tuning config requires --init-checkpoint")
    source = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint_path
        else None
    )
    resuming = bool(c.get("resume"))
    if resuming:
        if source.get("training_state_version") != 2:
            raise ValueError(
                "legacy checkpoint: use --init-checkpoint and a new output directory"
            )
        # Resume the saved experiment; only runtime settings and total epochs
        # may change. Learning rate/optimizer/scheduler are restored together.
        allowed = {
            "resume",
            "epochs",
            "device",
            "num_workers",
            "threads",
            "data_root",
            "log_every_batches",
            "multiprocessing_context",
        }
        if set(c) - allowed:
            raise ValueError(
                "resume restores saved config; only epochs/device/workers/threads/data_root/logging may override it"
            )
        checkpoint_path = Path(checkpoint_path).resolve()
        if checkpoint_path.name != "last.pt":
            raise ValueError(
                "resume requires last.pt; use --init-checkpoint for best.pt"
            )
        best_path = checkpoint_path.with_name("best.pt")
        if not best_path.is_file():
            raise FileNotFoundError("resume requires best.pt beside last.pt")
        saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
        if saved_best.get("run_id") != source.get("run_id"):
            raise ValueError("best.pt and last.pt must belong to the same training run")
        # If interrupted between the atomic best.pt and last.pt writes, the
        # former contains the newest completed epoch, including optimizer state.
        if saved_best["epoch"] > source["epoch"]:
            source = saved_best
        c = {**source["config"], **c}
        c.pop("init_checkpoint", None)
        c["output"] = str(checkpoint_path.parent)
    run_id = source["run_id"] if resuming else uuid.uuid4().hex
    seed = c.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(c.get("threads", 4))
    device, precision = training_device(c)
    root = Path(c.get("data_root", "data"))
    out = Path(c.get("output", "visual_training/checkpoints"))
    if not resuming and any(
        (out / name).exists() for name in ("best.pt", "last.pt", "history.json")
    ):
        raise FileExistsError(
            f"training output already exists: {out}; use a new --output or --resume last.pt"
        )
    epochs = int(c.get("epochs", 20))
    start_epoch = int(source["epoch"]) if resuming else 0
    if epochs <= start_epoch:
        raise ValueError(f"epochs must exceed completed epochs ({start_epoch})")
    early = c.get("early_stopping", {})
    patience = int(early.get("patience", 0))
    min_delta = float(early.get("min_delta", 0.0))
    if patience < 0 or min_delta < 0:
        raise ValueError("early-stopping patience and min_delta must be nonnegative")
    size = c.get("image_size", 256)
    train_set = HDataset(
        root / "splits/train.json", root, size, PhotometricAugmentation(seed)
    )
    val_set = HDataset(root / "splits/val.json", root, size)
    loaders = [
        make_loader(d, c, i == 0, seed + i) for i, d in enumerate((train_set, val_set))
    ]
    model = HSegmentationKeypoints(
        **c.get("model", source["model_config"] if source else {})
    ).to(device)
    if source:
        if model.model_config != source["model_config"]:
            raise ValueError("checkpoint model_config must match the training model")
        model.load_state_dict(source["model"])
    if c.get("channels_last", False):
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=c.get("learning_rate", 0.001),
        weight_decay=c.get("weight_decay", 1e-4),
        fused=bool(c.get("fused_optimizer", False)) if device.type == "cuda" else None,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    schedule = c.get("lr_scheduler", {})
    if schedule.get("name", "none") not in ("none", "plateau"):
        raise ValueError("lr_scheduler.name must be none or plateau")
    scheduler = None
    if schedule.get("name") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=schedule.get("factor", 0.5),
            patience=schedule.get("patience", 2),
            threshold=schedule.get("min_delta", 0.001),
            threshold_mode="abs",
            min_lr=schedule.get("min_lr", 1e-6),
        )
    best = float("inf")
    significant_best = float("inf")
    bad_epochs = 0
    history = []
    if resuming:
        optimizer.load_state_dict(source["optimizer"])
        scaler.load_state_dict(source["scaler"])
        if scheduler:
            scheduler.load_state_dict(source["scheduler"])
        best = source["best_val_loss"]
        significant_best = source["significant_best_val_loss"]
        bad_epochs = source["bad_epochs"]
        history = source["history"]
        if patience and bad_epochs >= patience:
            raise ValueError(
                "run already early-stopped; start a new --init-checkpoint experiment"
            )
        torch.set_rng_state(source["torch_rng_state"])
        for loader, state in zip(loaders, source["loader_rng_states"]):
            loader.generator.set_state(state)
        train_set.augmentation.rng.bit_generator.state = source[
            "augmentation_rng_state"
        ]
        if device.type == "cuda" and source.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(source["cuda_rng_states"])
    out.mkdir(parents=True, exist_ok=True)

    def payload(epoch):
        return {
            "training_state_version": 2,
            "run_id": run_id,
            "model": model.state_dict(),
            "model_config": model.model_config,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "config": c,
            "best_val_loss": best,
            "significant_best_val_loss": significant_best,
            "bad_epochs": bad_epochs,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None,
            "loader_rng_states": [loader.generator.get_state() for loader in loaders],
            "augmentation_rng_state": train_set.augmentation.rng.bit_generator.state,
        }

    print(
        json.dumps(
            {
                "event": "training_start",
                "device": str(device),
                "precision": precision,
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "image_size": size,
                "batch_size": c.get("batch_size", 8),
                "num_workers": c.get("num_workers", 0),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "init_checkpoint": c.get("init_checkpoint"),
                "resume": c.get("resume"),
                "output": str(out),
                "completed_epochs": start_epoch,
            }
        ),
        flush=True,
    )
    if source and not resuming:
        # Evaluate under the current validation settings before any update. A
        # failed fine-tune must still return the original best weights.
        baseline = run_epoch(
            model, loaders[1], optimizer, scaler, device, precision, c, 0, "val"
        )
        best = significant_best = baseline["loss"]
        if scheduler:
            scheduler.step(best)
        history.append(
            {"epoch": 0, "event": "baseline", "val_loss": best, "val_metrics": baseline}
        )
        atomic_save(payload(0), out / "best.pt")
        atomic_save(payload(0), out / "last.pt")
        save_history(history, out / "history.json")
        print(json.dumps(history[-1]), flush=True)
    for epoch in range(start_epoch + 1, epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model, loaders[0], optimizer, scaler, device, precision, c, epoch, "train"
        )
        val_metrics = run_epoch(
            model, loaders[1], optimizer, scaler, device, precision, c, epoch, "val"
        )
        value = val_metrics["loss"]
        improved = value < best
        best = min(best, value)
        if value < significant_best - min_delta:
            significant_best = value
            bad_epochs = 0
        else:
            bad_epochs += 1
        if scheduler:
            scheduler.step(value)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": value,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "learning_rate": learning_rate,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "best_val_loss": best,
            "bad_epochs": bad_epochs,
            "elapsed_s": time.perf_counter() - started,
        }
        if device.type == "cuda":
            record["cuda_peak_memory_gib"] = (
                torch.cuda.max_memory_allocated(device) / 2**30
            )
        history.append(record)
        print(json.dumps(record), flush=True)
        state = payload(epoch)
        if improved:
            atomic_save(state, out / "best.pt")
        atomic_save(state, out / "last.pt")
        save_history(history, out / "history.json")
        if patience and bad_epochs >= patience:
            print(
                json.dumps(
                    {"event": "early_stopping", "epoch": epoch, "best_val_loss": best}
                ),
                flush=True,
            )
            break
    print(
        json.dumps(
            {
                "event": "training_complete",
                "best_val_loss": best,
                "checkpoint": str(out / "best.pt"),
            }
        ),
        flush=True,
    )
    return out / "best.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    start = p.add_mutually_exclusive_group()
    start.add_argument(
        "--init-checkpoint", help="load weights and fine-tune in a new output directory"
    )
    start.add_argument(
        "--resume", help="restore a new-format last.pt in its original output directory"
    )
    p.add_argument("--epochs", type=int)
    p.add_argument("--data-root")
    p.add_argument("--output")
    p.add_argument("--image-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--log-every-batches", type=int)
    p.add_argument("--device")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    a = p.parse_args()
    # The shell launcher supplies a default --config, which is intentionally
    # ignored on resume: optimizer and schedule belong to the saved run.
    c = (
        {}
        if a.resume
        else yaml.safe_load(Path(a.config or "configs/training.yaml").read_text())
    )
    for name in (
        "epochs",
        "data_root",
        "output",
        "image_size",
        "batch_size",
        "num_workers",
        "device",
        "precision",
        "learning_rate",
        "log_every_batches",
        "init_checkpoint",
        "resume",
    ):
        if getattr(a, name) is not None:
            c[name] = getattr(a, name)
    train(c)


if __name__ == "__main__":
    main()
