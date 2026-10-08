import argparse
import json
from pathlib import Path
import random
import time
from contextlib import nullcontext
import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from visual_training.data.h_dataset import HDataset
from visual_training.data.augmentations import PhotometricAugmentation
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


def train(config):
    c = config
    seed = c.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(c.get("threads", 4))
    device, precision = training_device(c)
    root = Path(c.get("data_root", "data"))
    out = Path(c.get("output", "visual_training/checkpoints"))
    out.mkdir(parents=True, exist_ok=True)
    size = c.get("image_size", 256)
    train_set = HDataset(
        root / "splits/train.json", root, size, PhotometricAugmentation(seed)
    )
    val_set = HDataset(root / "splits/val.json", root, size)
    loaders = [
        make_loader(d, c, i == 0, seed + i) for i, d in enumerate((train_set, val_set))
    ]
    model = HSegmentationKeypoints(**c.get("model", {})).to(device)
    if c.get("channels_last", False):
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=c.get("learning_rate", 0.001),
        weight_decay=1e-4,
        fused=bool(c.get("fused_optimizer", False)) if device.type == "cuda" else None,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    non_blocking = device.type == "cuda" and c.get("pin_memory", False)
    print(
        json.dumps(
            dict(
                event="training_start",
                device=str(device),
                precision=precision,
                torch_version=torch.__version__,
                cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                image_size=size,
                batch_size=c.get("batch_size", 8),
                num_workers=c.get("num_workers", 0),
            )
        ),
        flush=True,
    )
    best = float("inf")
    history = []
    for epoch in range(c.get("epochs", 20)):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses = []
        for phase, loader in enumerate(loaders):
            model.train(phase == 0)
            total = torch.zeros((), device=device)
            count = 0
            with torch.set_grad_enabled(phase == 0):
                for sample in loader:
                    sample = {
                        k: v.to(device, non_blocking=non_blocking)
                        for k, v in sample.items()
                    }
                    if c.get("channels_last", False):
                        sample["image"] = sample["image"].contiguous(
                            memory_format=torch.channels_last
                        )
                    if phase == 0:
                        optimizer.zero_grad(set_to_none=True)
                    with (
                        torch.autocast(device_type=device.type, dtype=dtype)
                        if precision != "fp32"
                        else nullcontext()
                    ):
                        prediction = model(sample["image"])
                    # Loss stays in FP32; large masks must not overflow FP16 sums.
                    loss, _ = segmentation_keypoint_loss(
                        prediction, sample, return_metrics=False
                    )
                    if phase == 0:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        scaler.step(optimizer)
                        scaler.update()
                    n = sample["image"].shape[0]
                    total += loss.detach() * n
                    count += n
            losses.append(float(total / count))
        record = dict(
            epoch=epoch + 1,
            train_loss=losses[0],
            val_loss=losses[1],
            elapsed_s=time.perf_counter() - started,
        )
        if device.type == "cuda":
            record["cuda_peak_memory_gib"] = (
                torch.cuda.max_memory_allocated(device) / 2**30
            )
        history.append(record)
        print(json.dumps(record), flush=True)
        payload = dict(
            model=model.state_dict(),
            model_config=model.model_config,
            epoch=epoch + 1,
            optimizer=optimizer.state_dict(),
            config=c,
        )
        torch.save(payload, out / "last.pt")
        if losses[1] < best:
            best = losses[1]
            torch.save(payload, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return out / "best.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--data-root")
    p.add_argument("--output")
    p.add_argument("--image-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    for name in (
        "epochs",
        "data_root",
        "output",
        "image_size",
        "batch_size",
        "num_workers",
        "device",
        "precision",
    ):
        if getattr(a, name) is not None:
            c[name] = getattr(a, name)
    train(c)


if __name__ == "__main__":
    main()
