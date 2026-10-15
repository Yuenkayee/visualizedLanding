"""Train segmentation/keypoints on CPU or CUDA, with optional short-run checks."""

import argparse
from contextlib import ExitStack, nullcontext
import gc
import json
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
    get_worker_info,
)

from visual_training.data.h_dataset import HDataset
from visual_training.data.augmentations import PhotometricAugmentation
from visual_training.data.training_loader import LimitedBatchSampler, TrainingDataLoader
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss
from visual_training.metrics import grouped_metrics, training_report


def seed_worker(worker_id):
    """Spawn-safe workers, independent augmentation RNGs, no OpenCV oversubscription."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)
    worker = get_worker_info()
    if worker.dataset.augmentation is not None:
        worker.dataset.augmentation.rng = np.random.default_rng(seed)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(config, device):
    if config.get("precision", "fp32") == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def move_batch(sample, device, channels_last=False):
    sample = {
        k: v.to(device, non_blocking=device.type == "cuda") for k, v in sample.items()
    }
    if channels_last:
        sample["image"] = sample["image"].contiguous(memory_format=torch.channels_last)
    return sample


def make_model_optimizer(config, device):
    model = HSegmentationKeypoints(**config.get("model", {})).to(device)
    if config.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("learning_rate", 0.001), weight_decay=1e-4
    )
    return model, optimizer


def probe_batch(config, dataset, device, batch_size):
    """Disposable real optimizer steps; state is discarded before actual training."""
    model, optimizer = make_model_optimizer(config, device)
    samples = [dataset[i % len(dataset)] for i in range(batch_size)]
    sample = move_batch(
        {k: torch.stack([s[k] for s in samples]) for k in samples[0]},
        device,
        config.get("channels_last", False),
    )
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = 0.0
    for step in range(3):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast(config, device):
            prediction = model(sample["image"])
            loss, _ = segmentation_keypoint_loss(
                prediction, sample, return_metrics=False, **config.get("loss", {})
            )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite batch probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        if step:  # First step includes cuDNN tuning and optimizer allocation.
            elapsed += time.perf_counter() - start
    return dict(
        batch_size=batch_size,
        images_per_s=2 * batch_size / elapsed,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def choose_batch(config, dataset, device, emit):
    if config.get("batch_size", 8) != "auto":
        batch = int(config.get("batch_size", 8))
        if batch < 1:
            raise ValueError("batch_size must be positive")
        return batch
    if device.type != "cuda":
        raise ValueError("automatic batch selection requires CUDA")
    maximum = int(config.get("max_batch_size", 16))
    fraction = float(config.get("memory_fraction", 0.85))
    if maximum < 1 or not 0 < fraction < 1:
        raise ValueError("invalid max_batch_size/memory_fraction")
    free, _ = torch.cuda.mem_get_info(device)
    budget = free * fraction
    candidates = sorted({min(maximum, n) for n in (1, 2, 4, 8, 16, 32, 64)} | {maximum})
    trials = []
    for batch in reversed(candidates):
        try:
            result = probe_batch(config, dataset, device, batch)
            result["within_memory_budget"] = result["peak_reserved_bytes"] <= budget
            emit(dict(event="batch_probe", **result, memory_budget_bytes=int(budget)))
            if result["within_memory_budget"]:
                trials.append(result)
        except torch.cuda.OutOfMemoryError:
            emit(dict(event="batch_probe", batch_size=batch, status="out_of_memory"))
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        if trials and batch <= 4:
            break
    if not trials:
        raise RuntimeError(
            "No batch fits the memory budget; free GPU memory before retrying"
        )
    best = max(trials, key=lambda trial: trial["images_per_s"])
    emit(dict(event="batch_selected", **best))
    return best["batch_size"]


def make_loader(dataset, config, batch, training, device):
    workers = int(config.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    options = {}
    if workers:
        options.update(
            multiprocessing_context="spawn",
            worker_init_fn=seed_worker,
            persistent_workers=config.get("persistent_workers", True),
            prefetch_factor=config.get("prefetch_factor", 2),
        )
    generator = torch.Generator().manual_seed(
        config.get("seed", 42) + int(not training)
    )
    sampler = (
        RandomSampler(dataset, generator=generator)
        if training
        else SequentialSampler(dataset)
    )
    if training and config.get("touchdown_sampling_weight", 1) != 1:
        weights = [config["touchdown_sampling_weight"] if row.get("stage") == "touchdown" else 1.
                   for row in dataset.rows]
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    batch_sampler = BatchSampler(sampler, batch_size=batch, drop_last=False)
    limit = config.get("max_train_batches" if training else "max_val_batches")
    if limit is not None:
        batch_sampler = LimitedBatchSampler(batch_sampler, limit)
    return TrainingDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=config.get("pin_memory", device.type == "cuda"),
        generator=generator,
        **options,
    )


def fit_shared_memory(config, batch, emit):
    """Docker often has only 64 MB /dev/shm despite abundant host RAM."""
    workers = int(config.get("num_workers", 0))
    if not workers or not Path("/dev/shm").is_dir():
        return
    free = shutil.disk_usage("/dev/shm").free
    # Both persistent train and validation pools may retain prefetched batches.
    batch_bytes = batch * config.get("image_size", 256) ** 2 * 16
    per_worker = 2 * (config.get("prefetch_factor", 2) + 1) * batch_bytes
    allowed = max(0, int(0.7 * free // per_worker))
    if workers > allowed:
        config["num_workers"] = allowed
        emit(
            dict(
                event="loader_shared_memory_limit",
                requested_workers=workers,
                selected_workers=allowed,
                shared_memory_free_bytes=free,
                hint="Increase container /dev/shm (e.g. --shm-size=4g) for more loader workers",
            )
        )


def save_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config):
    c = dict(config)
    seed = c.get("seed", 42)
    seed_all(seed)
    torch.set_num_threads(c.get("threads", 4))
    cv2.setNumThreads(1)
    device = torch.device(c.get("device", "cpu"))
    if c.get("precision", "fp32") not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if c.get("precision") == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but unsupported")
        torch.backends.cudnn.benchmark = c.get("cudnn_benchmark", False)
        torch.backends.cuda.matmul.allow_tf32 = c.get("tf32", False)
        torch.backends.cudnn.allow_tf32 = c.get("tf32", False)
    root = Path(c.get("data_root", "data"))
    out = Path(c.get("output", "visual_training/checkpoints"))
    save_results = c.get("save_checkpoints", True)
    if save_results:
        out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(c["log_dir"]) if c.get("log_dir") else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    def emit(record):
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        if log_dir:
            with (log_dir / "metrics.jsonl").open("a") as stream:
                stream.write(line + "\n")

    size = int(c.get("image_size", 256))
    if size < 8 or int(c.get("epochs", 20)) < 1:
        raise ValueError("image_size >=8 and epochs >=1 required")
    for name in ("max_train_batches", "max_val_batches"):
        if c.get(name) is not None and c[name] < 1:
            raise ValueError(f"{name} must be positive")
    train_set = HDataset(
        root / "splits/train.json", root, size,
        PhotometricAugmentation(seed, c.get("difficult_augmentation", False)), c.get("weathers")
    )
    val_set = HDataset(root / "splits/val.json", root, size, weathers=c.get("weathers"))
    batch = choose_batch(c, train_set, device, emit)
    c["batch_size"] = batch
    fit_shared_memory(c, batch, emit)
    # Probes do not contribute trained weights or consume actual training RNGs.
    seed_all(seed)
    train_set.augmentation = PhotometricAugmentation(seed, c.get("difficult_augmentation", False))
    loaders = [
        make_loader(d, c, batch, i == 0, device)
        for i, d in enumerate((train_set, val_set))
    ]
    model, optimizer = make_model_optimizer(c, device)
    if log_dir:
        (log_dir / "resolved_config.json").write_text(json.dumps(c, indent=2) + "\n")
    emit(
        dict(
            event="training_start",
            config=c,
            train_samples=len(train_set),
            val_samples=len(val_set),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            torch=str(torch.__version__),
        )
    )
    best, history = float("inf"), []
    best_keypoint = (float("inf"),) * 3
    stale = 0
    schedule = c.get("scheduler")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **schedule
    ) if schedule else None
    start = time.monotonic()
    # Close both worker pools before reporting success, also on training errors.
    with ExitStack() as cleanup:
        for loader in loaders:
            cleanup.callback(loader.close)
        for epoch in range(c.get("epochs", 20)):
            phases = []
            for phase, loader in enumerate(loaders):
                training = phase == 0
                name = "train" if training else "val"
                model.train(training)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                phase_start = time.monotonic()
                total = torch.zeros((), device=device)
                intersection, union, error, visible = [
                    torch.zeros((), device=device) for _ in range(4)
                ]
                count = 0
                component_sums = {}
                gradient_norm_sum = torch.zeros((), device=device)
                tails, visibility_rows, sample_indices = [], [], []
                with torch.set_grad_enabled(training):
                    for index, sample in enumerate(loader):
                        sample = move_batch(
                            sample, device, c.get("channels_last", False)
                        )
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                        with autocast(c, device):
                            prediction = model(sample["image"])
                            loss, components = segmentation_keypoint_loss(
                                prediction, sample, return_metrics="tensor", **c.get("loss", {})
                            )
                        if not torch.isfinite(loss):
                            raise ValueError(
                                f"Nonfinite {name} loss at epoch {epoch + 1}, batch {index + 1}"
                            )
                        if training:
                            loss.backward()
                            gradient_norm_sum += torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 5.0, error_if_nonfinite=True
                            ).detach()
                            optimizer.step()
                        else:
                            pred = prediction["mask_logits"] > 0
                            truth = sample["mask"] > 0.5
                            intersection += (pred & truth).sum()
                            union += (pred | truth).sum()
                            vis = sample["visibility"]
                            error += torch.linalg.vector_norm(
                                prediction["keypoints"] - sample["keypoints"], dim=-1
                            )[vis].sum()
                            visible += vis.sum()
                            tails.append((torch.linalg.vector_norm(
                                prediction["keypoints"].float() - sample["keypoints"], dim=-1) * size).detach())
                            visibility_rows.append(vis)
                            sample_indices.append(sample["index"])
                        n = sample["image"].shape[0]
                        for key, value in components.items():
                            component_sums[key] = component_sums.get(key, 0) + value * n
                        total += loss.detach() * n
                        count += n
                        if (index + 1) % c.get("log_every", 20) == 0:
                            emit(
                                dict(
                                    event="progress",
                                    epoch=epoch + 1,
                                    phase=name,
                                    batch=index + 1,
                                    samples=count,
                                    loss=float(total / count),
                                )
                            )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.monotonic() - phase_start
                phases.append(
                    dict(
                        loss=float(total / count),
                        loss_components={key: float(value / count) for key, value in component_sums.items()},
                        gradient_norm_before_clip=float(gradient_norm_sum / (index + 1)) if training else None,
                        samples=count,
                        seconds=seconds,
                        images_per_s=count / seconds,
                        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else 0,
                    )
                )
            tail_metrics = grouped_metrics(
                torch.cat(tails).cpu().numpy(), torch.cat(visibility_rows).cpu().numpy(),
                [val_set.rows[i] for i in torch.cat(sample_indices).cpu().tolist()])
            record = dict(
                event="epoch",
                epoch=epoch + 1,
                train_loss=phases[0]["loss"],
                val_loss=phases[1]["loss"],
                phases=dict(train=phases[0], val=phases[1]),
                val_mask_iou=float(intersection / union.clamp_min(1)),
                val_normalized_keypoint_error=float(error / visible)
                if visible > 0
                else None,
                visible_keypoints=int(visible),
                keypoint_metrics=tail_metrics,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
            history.append(record)
            emit(record)
            improved = record["val_loss"] < best
            stale = 0 if improved else stale + 1
            if improved:
                best = record["val_loss"]
            if scheduler:
                scheduler.step(record["val_loss"])
            tail = tail_metrics["all"]
            score = (1 - (tail["four_point_success_5px"] or 0),
                     tail["p99_px"] if tail["p99_px"] is not None else float("inf"),
                     tail["mean_px"] if tail["mean_px"] is not None else float("inf"))
            if save_results:
                payload = dict(
                    model=model.state_dict(),
                    model_config=model.model_config,
                    epoch=epoch + 1,
                    optimizer=optimizer.state_dict(),
                    config=c,
                )
                save_checkpoint(payload, out / "last.pt")
                if improved:
                    save_checkpoint(payload, out / "best.pt")
                    save_checkpoint(payload, out / "best_loss.pt")
                if score < best_keypoint:
                    best_keypoint = score
                    save_checkpoint(payload, out / "best_keypoint.pt")
                (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if c.get("early_stopping_patience", 0) and stale >= c["early_stopping_patience"]:
                emit(dict(event="early_stopping", epoch=epoch + 1, stale_epochs=stale))
                break
    # Pretrain produces logs only, including diagnostics, never training results.
    if save_results:
        training_report(history, out / "metrics")
    emit(
        dict(
            event="training_complete",
            epochs=len(history),
            elapsed_s=time.monotonic() - start,
            checkpoints_saved=save_results,
            best_checkpoint=str(out / "best.pt") if save_results else None,
        )
    )
    return out / "best.pt" if save_results else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--data-root")
    p.add_argument("--output")
    p.add_argument("--image-size", type=int)
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    for name in ("epochs", "data_root", "output", "image_size"):
        if getattr(a, name) is not None:
            c[name] = getattr(a, name)
    train(c)


if __name__ == "__main__":
    main()
