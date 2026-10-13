"""Benchmark real training in isolated processes and export measured settings."""

import argparse
import itertools
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from visual_training.benchmark_resources import ResourceMonitor, inventory
from visual_training.data.augmentations import PhotometricAugmentation
from visual_training.data.h_dataset import HDataset
from visual_training.metrics import EpochMetrics
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.train import make_loader, process_batch, training_device


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class BenchmarkSampler:
    """Uniformly cover the split; stop prefetch exactly at the measured window."""

    def __init__(self, dataset_size, samples, seed):
        self.dataset_size = dataset_size
        self.samples = samples
        self.seed = seed

    def __len__(self):
        return self.samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        remaining = self.samples
        while remaining:
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
            count = min(remaining, self.dataset_size)
            yield from indices[:count]
            remaining -= count


def measure_phase(
    model, loader, optimizer, scaler, device, precision, config, warmup, steps, training
):
    """CUDA events diagnose device time; synchronize only at window boundaries."""
    model.train(training)
    samples = (warmup + steps) * loader.batch_size
    loader.batch_sampler.sampler = BenchmarkSampler(
        len(loader.dataset), samples, config.get("seed", 42) + (0 if training else 1)
    )
    iterator = iter(loader)

    warm_metrics = EpochMetrics(device, config["image_size"])
    warm_started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for _ in range(warmup):
            process_batch(
                model,
                next(iterator),
                optimizer,
                scaler,
                device,
                precision,
                config,
                warm_metrics,
                training,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warm_seconds = time.perf_counter() - warm_started
        metrics = EpochMetrics(device, config["image_size"])
        waits, events = [], []
        started = time.perf_counter()
        for step in range(steps):
            before = time.perf_counter()
            sample = next(iterator)
            waits.append(time.perf_counter() - before)
            if device.type == "cuda":
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
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
            if device.type == "cuda":
                end.record()
                events.append((begin, end))
            if (step + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "event": "trial_progress",
                            "phase": "train" if training else "val",
                            "steps": step + 1,
                            "target_steps": steps,
                        }
                    ),
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    # Finish the iterator so training workers have no outstanding augmentation
    # or prefetched batches competing with the subsequent validation phase.
    if next(iterator, None) is not None:
        raise RuntimeError("benchmark sampler exceeded its window")
    result = metrics.result()
    if not all(math.isfinite(result[k]) for k in metrics.losses):
        raise FloatingPointError("non-finite measured loss")
    return {
        "steps": steps,
        "images": metrics.samples,
        "elapsed_s": elapsed,
        "images_per_second": metrics.samples / elapsed,
        "batch_wall_ms": elapsed * 1000 / steps,
        "loader_wait_mean_ms": statistics.mean(waits) * 1000,
        "loader_wait_p95_ms": float(np.percentile(waits, 95)) * 1000,
        "loader_wait_wall_fraction": sum(waits) / elapsed,
        "cuda_step_mean_ms": statistics.mean(a.elapsed_time(b) for a, b in events)
        if events
        else None,
        "warmup_s": warm_seconds,
        "dataset_restarts": (samples - 1) // len(loader.dataset),
        "metrics": result,
    }


def run_trial(spec):
    config = spec["config"]
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("threads", 8)))
    device, precision = training_device(config)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("benchmark supports CUDA servers or CPU smoke tests")
    hardware = inventory()
    gpu_uuid = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_uuid = str(properties.uuid) if getattr(properties, "uuid", None) else None
        free, total = torch.cuda.mem_get_info(device)
        hardware.update(
            gpu=properties.name,
            gpu_uuid=gpu_uuid,
            gpu_total_gib=total / 2**30,
            gpu_free_before_gib=free / 2**30,
        )
        torch.cuda.reset_peak_memory_stats(device)
    root = Path(config["data_root"])
    datasets = [
        HDataset(
            root / "splits/train.json",
            root,
            config["image_size"],
            PhotometricAugmentation(seed),
        ),
        HDataset(root / "splits/val.json", root, config["image_size"]),
    ]
    loaders = [
        make_loader(ds, config, i == 0, seed + i) for i, ds in enumerate(datasets)
    ]
    model = HSegmentationKeypoints(**config.get("model", {})).to(device)
    if spec.get("checkpoint"):
        payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
        if payload["model_config"] != model.model_config:
            raise ValueError("benchmark model_config differs from checkpoint")
        model.load_state_dict(payload["model"])
        del payload
    if config.get("channels_last", False):
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-4),
        fused=bool(config.get("fused_optimizer", False))
        if device.type == "cuda"
        else None,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    with ResourceMonitor(gpu_uuid) as monitor:
        train = measure_phase(
            model,
            loaders[0],
            optimizer,
            scaler,
            device,
            precision,
            config,
            spec["warmup"],
            spec["steps"],
            True,
        )
        if not all(
            bool(torch.isfinite(p).all())
            and (p.grad is None or bool(torch.isfinite(p.grad).all()))
            for p in model.parameters()
        ):
            raise FloatingPointError("non-finite model parameters or gradients")
        validation = measure_phase(
            model,
            loaders[1],
            optimizer,
            scaler,
            device,
            precision,
            config,
            spec["warmup"],
            spec["val_steps"],
            False,
        )
    result = {
        "status": "ok",
        "hardware": hardware,
        "torch_version": str(torch.__version__),
        "python_version": sys.version.split()[0],
        "cuda_runtime": torch.version.cuda,
        "train": train,
        "val": validation,
        "train_images": len(datasets[0]),
        "val_images": len(datasets[1]),
        "estimated_epoch_s": len(datasets[0]) / train["images_per_second"]
        + len(datasets[1]) / validation["images_per_second"],
        "telemetry": monitor.report(),
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 2**30
        )
        result["cuda_peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved(device) / 2**30
        )
        if result["cuda_peak_reserved_gib"] > hardware["gpu_free_before_gib"] * (
            1 - spec["memory_margin"]
        ):
            result["status"] = "insufficient_headroom"
    return result


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    try:
        result = run_trial(spec)
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "error": str(error)}
    except Exception as error:  # noqa: BLE001 -- report isolated worker failures; do not recommend them
        import traceback

        traceback.print_exc()
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    write_json(spec["result_path"], result)


def stop_process(process):
    # Every trial owns its process group, including spawned DataLoader workers.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        if process.poll() is not None:
            return
        process.terminate()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader; descendants may still need cleanup.
        if os.name == "posix":
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
        elif process.returncode is not None:
            return
        time.sleep(0.05)
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def launch_trial(spec, directory, timeout):
    directory.mkdir()
    spec = {**spec, "result_path": str((directory / "result.json").resolve())}
    write_json(directory / "spec.json", spec)
    started = time.perf_counter()
    with (directory / "trial.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "visual_training.benchmark",
                "--worker-spec",
                str(directory / "spec.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            heartbeat = started
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed > timeout:
                    stop_process(process)
                    return {
                        "status": "timeout",
                        "error": f"exceeded {timeout}s; see {directory / 'trial.log'}",
                    }
                if time.perf_counter() - heartbeat >= 30:
                    print(
                        json.dumps(
                            {
                                "event": "trial_running",
                                "trial": directory.name,
                                "elapsed_s": round(elapsed),
                                "log": str(directory / "trial.log"),
                            }
                        ),
                        flush=True,
                    )
                    heartbeat = time.perf_counter()
                time.sleep(0.2)
        finally:
            stop_process(process)
    result_path = Path(spec["result_path"])
    if process.returncode != 0 or not result_path.exists():
        return {
            "status": "error",
            "error": f"worker exit {process.returncode}; see {directory / 'trial.log'}",
        }
    return json.loads(result_path.read_text())


def candidates(config, batches, workers, threads, prefetch):
    baseline = {
        key: config.get(key, default)
        for key, default in (
            ("batch_size", 16),
            ("num_workers", 8),
            ("threads", 8),
            ("prefetch_factor", 2),
        )
    }
    rows = [baseline]
    for batch, worker_count, thread_count, factor in itertools.product(
        batches, workers, threads, prefetch
    ):
        row = {
            "batch_size": batch,
            "num_workers": worker_count,
            "threads": thread_count,
            "prefetch_factor": factor if worker_count else baseline["prefetch_factor"],
        }
        if row not in rows:
            rows.append(row)
    # Baseline first; randomize the rest to reduce ordering/cache/thermal bias.
    tail = rows[1:]
    random.Random(config.get("seed", 42)).shuffle(tail)
    return [baseline, *tail]


def choose_recommendation(rows, repeats, max_variation):
    """Only confirmed, finite, memory-safe candidates can be recommended."""
    grouped = {}
    for row in rows:
        if row["stage"] == "confirm":
            grouped.setdefault(row["candidate_id"], []).append(row)
    eligible = []
    for key, trials in grouped.items():
        if len(trials) != repeats or any(row["status"] != "ok" for row in trials):
            continue
        times = [row["estimated_epoch_s"] for row in trials]
        if any(not math.isfinite(t) or t <= 0 for t in times):
            continue
        median = statistics.median(times)
        variation = (max(times) - min(times)) / median
        if variation <= max_variation:
            eligible.append(
                {
                    "candidate_id": key,
                    "settings": trials[0]["settings"],
                    "estimated_epoch_s": median,
                    "relative_range": variation,
                }
            )
    if not eligible:
        return None
    # Within 3% of the fastest result prefer less memory/CPU pressure.
    fastest = min(row["estimated_epoch_s"] for row in eligible)
    near_best = [row for row in eligible if row["estimated_epoch_s"] <= fastest * 1.03]
    return min(
        near_best,
        key=lambda r: (
            r["settings"]["batch_size"],
            r["settings"]["num_workers"],
            r["settings"]["prefetch_factor"],
            r["settings"]["threads"],
        ),
    )


def run_benchmark(args):
    config = yaml.safe_load(Path(args.config).read_text())
    for key in ("data_root", "device"):
        if getattr(args, key):
            config[key] = getattr(args, key)
    config.setdefault("image_size", 512)
    config["data_root"] = str(
        Path(config.get("data_root", "data/multi_camera")).resolve()
    )
    config["multiprocessing_context"] = "spawn"
    config["log_every_batches"] = 0
    checkpoint = args.checkpoint or config.get("init_checkpoint")
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    for name in ("train", "val"):
        if not (Path(config["data_root"]) / f"splits/{name}.json").is_file():
            raise FileNotFoundError(f"missing {name} split under {config['data_root']}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    space = candidates(
        config,
        args.batch_sizes,
        args.workers,
        args.threads or [config.get("threads", 8)],
        args.prefetch_factors,
    )
    report = {
        "config": config,
        "checkpoint": checkpoint,
        "hardware": inventory(),
        "search_space": space,
        "arguments": vars(args),
        "trials": [],
        "recommendation": None,
    }
    write_json(output / "report.json", report)

    def test(index, settings, stage, repeat=0):
        name = f"{stage}_{index:03d}_{repeat}"
        print(
            json.dumps({"event": "trial_start", "trial": name, **settings}), flush=True
        )
        spec = {
            "config": {**config, **settings},
            "checkpoint": checkpoint,
            "warmup": args.warmup,
            "steps": args.steps if stage == "screen" else args.confirm_steps,
            "val_steps": args.val_steps
            if stage == "screen"
            else max(args.val_steps, args.confirm_steps // 4),
            "memory_margin": args.memory_margin,
        }
        result = launch_trial(spec, output / name, args.trial_timeout)
        row = {
            "candidate_id": index,
            "stage": stage,
            "repeat": repeat,
            "settings": settings,
            "directory": str(output / name),
            **result,
        }
        report["trials"].append(row)
        write_json(output / "report.json", report)
        print(
            json.dumps(
                {
                    "event": "trial_complete",
                    "trial": name,
                    "status": row["status"],
                    "train_images_per_second": row.get("train", {}).get(
                        "images_per_second"
                    ),
                    "estimated_epoch_s": row.get("estimated_epoch_s"),
                    "error": row.get("error"),
                }
            ),
            flush=True,
        )
        return row

    screened = [test(i, settings, "screen") for i, settings in enumerate(space)]
    successful = sorted(
        (row for row in screened if row["status"] == "ok"),
        key=lambda r: r["estimated_epoch_s"],
    )
    finalists = successful[: args.top_k]
    baseline = next((row for row in successful if row["candidate_id"] == 0), None)
    if baseline and baseline not in finalists:
        finalists.append(baseline)
    same_batch = next(
        (
            row
            for row in successful
            if row["settings"]["batch_size"] == space[0]["batch_size"]
        ),
        None,
    )
    if same_batch and same_batch not in finalists:
        finalists.append(same_batch)
    for repeat in range(args.repeats):
        order = list(finalists)
        random.Random(config.get("seed", 42) + repeat).shuffle(order)
        for row in order:
            test(row["candidate_id"], row["settings"], "confirm", repeat)
    best = choose_recommendation(report["trials"], args.repeats, args.max_variation)
    report["recommendation"] = best
    fixed_batch = choose_recommendation(
        [
            row
            for row in report["trials"]
            if row["settings"]["batch_size"] == space[0]["batch_size"]
        ],
        args.repeats,
        args.max_variation,
    )
    report["recommendation_same_batch"] = fixed_batch

    def export_recommendation(choice, filename, training_directory):
        recommended = yaml.safe_load(Path(args.config).read_text())
        recommended.update(
            choice["settings"],
            data_root=config["data_root"],
            device=config.get("device", "cpu"),
            multiprocessing_context="spawn",
            output=str(output / training_directory),
        )
        recommended.pop("resume", None)
        if checkpoint:
            recommended["init_checkpoint"] = checkpoint
        recommended_path = output / filename
        recommended_path.write_text(
            "# Measured performance settings; learning rate and training objective preserved.\n"
            + yaml.safe_dump(recommended, sort_keys=False)
        )
        choice["config_path"] = str(recommended_path)
        choice["batch_size_changed"] = (
            choice["settings"]["batch_size"] != space[0]["batch_size"]
        )
        confirmed_baseline = choose_recommendation(
            [row for row in report["trials"] if row["candidate_id"] == 0],
            args.repeats,
            args.max_variation,
        )
        if confirmed_baseline:
            choice["speedup_over_baseline"] = (
                confirmed_baseline["estimated_epoch_s"] / choice["estimated_epoch_s"]
            )

    if best:
        export_recommendation(best, "recommended.yaml", "training_checkpoints")
    if fixed_batch:
        export_recommendation(
            fixed_batch,
            "recommended_same_batch.yaml",
            "training_checkpoints_same_batch",
        )
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "event": "benchmark_complete",
                "report": str(output / "report.json"),
                "recommendation": best,
                "recommendation_same_batch": fixed_batch,
            }
        ),
        flush=True,
    )
    if not best:
        raise RuntimeError(
            "no stable confirmed candidate; inspect report.json and trial.log files"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument(
        "--config", default="configs/multi_camera_finetune_rtx5090.yaml"
    )
    parser.add_argument("--data-root")
    parser.add_argument(
        "--checkpoint", help="optional weights to copy into disposable trial models"
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--output", default=f"outputs/benchmarks/{time.strftime('%Y%m%d_%H%M%S')}"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64]
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--threads", type=int, nargs="+")
    parser.add_argument("--prefetch-factors", type=int, nargs="+", default=[2])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--confirm-steps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--trial-timeout", type=float, default=900)
    parser.add_argument("--memory-margin", type=float, default=0.15)
    parser.add_argument("--max-variation", type=float, default=0.15)
    args = parser.parse_args()
    if args.worker_spec:
        worker(args.worker_spec)
        return
    positive = [
        *args.batch_sizes,
        *(args.threads or []),
        *args.prefetch_factors,
        args.warmup,
        args.steps,
        args.val_steps,
        args.confirm_steps,
        args.top_k,
        args.trial_timeout,
    ]
    if min(positive) <= 0 or min(args.workers) < 0 or args.repeats < 2:
        parser.error("sizes/steps/timeout must be positive; workers >= 0; repeats >= 2")
    if not 0 <= args.memory_margin < 1 or not 0 < args.max_variation < 1:
        parser.error("invalid memory margin or variability threshold")
    run_benchmark(args)


if __name__ == "__main__":
    main()
