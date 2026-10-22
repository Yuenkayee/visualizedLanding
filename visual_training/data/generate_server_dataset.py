"""Ubuntu 22.04 / RTX 4090 dataset coordinator (imports dependencies after setup)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from shell.render_environment import renderer_environment

ROOT = Path(__file__).resolve().parents[2]


def available_cpus():
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    # Some containers expose all host cores despite a smaller cgroup CPU quota.
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                count = min(count, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return min(16, count)


def run_workers(
    python, root, workers, threads, device="OPTIX", gpu_index=0, renderer="blender"
):
    """Long-lived independent interpreters, round-robin sorties, bounded memory."""
    root = Path(root).resolve()
    processes, logs = [], []
    (root / "logs").mkdir(exist_ok=True)
    try:
        for i in range(workers):
            log = (root / "logs" / f"worker_{i:02d}.log").open("w")
            logs.append(log)
            command = [
                str(python),
                "-u",
                "-m",
                "visual_training.data.render_landing_shard",
                "--plan",
                str(root / "plan.json"),
                "--worker-id",
                str(i),
                "--workers",
                str(workers),
                "--threads",
                str(threads),
                "--device",
                device,
                "--gpu-index",
                str(gpu_index),
                "--renderer",
                renderer,
            ]
            env = {
                **renderer_environment(),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            processes.append(
                subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            )
        last_update = 0
        while True:
            statuses = [process.poll() for process in processes]
            for i, status in enumerate(statuses):
                if status not in (None, 0):
                    raise RuntimeError(
                        f"Render worker {i} exited {status}; see {logs[i].name}"
                    )
            if time.monotonic() - last_update >= 30 or all(s == 0 for s in statuses):
                frames = 0
                for i in range(workers):
                    path = root / "workers" / f"{i:02d}_progress.json"
                    if path.exists():
                        frames += json.loads(path.read_text())["frames"]
                print(
                    f"[render] {frames} frames; {statuses.count(0)}/{workers} workers finished; "
                    f"logs: {root / 'logs'}",
                    flush=True,
                )
                last_update = time.monotonic()
            if all(s == 0 for s in statuses):
                break
            time.sleep(1)
    finally:
        # Also covers Ctrl-C, startup failures and one worker exhausting VRAM.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def merge_shards(config, plan, root, workers, renderer="blender"):
    from visual_training.data.generate_landing_dataset import finalize_dataset
    from visual_modeling.trajectories.landing import selected_frames

    root = Path(root).resolve()
    paths = {name: [] for name in ("train", "val", "test")}
    summary, sequences, runtimes = [], [], []
    for i in range(workers):
        shard = json.loads((root / "workers" / f"{i:02d}.json").read_text())
        if shard["worker_id"] != i:
            raise ValueError("incorrect shard id")
        sequences.extend(shard["sequences"])
        runtimes.append(
            dict(worker_id=i, **shard["runtime"], elapsed_s=shard["elapsed_s"])
        )
        for split in paths:
            paths[split].extend(shard["paths"][split])
        summary.extend(shard["summary"])
    if sorted(sequences) != sorted(entry["sequence_id"] for entry in plan):
        raise ValueError("missing or duplicated sorties in render shards")
    expected = {name: [] for name in paths}
    expected_images = []
    for entry in plan:
        sortie = json.loads(Path(entry["trajectory"]).read_text())
        for row in selected_frames(sortie, config["frames_per_stage"]):
            stem = f"{entry['sequence_id']}/{row['frame_id']:06d}"
            expected[entry["split"]].append(str(root / "annotations" / f"{stem}.json"))
            expected_images.append(str(root / "rendered" / f"{stem}.png"))
    if any(sorted(paths[s]) != sorted(expected[s]) for s in paths):
        raise ValueError("shard frames/splits differ from plan")
    if sorted(item["image"] for item in summary) != sorted(expected_images):
        raise ValueError("missing or duplicated frame summaries")
    # Match serial generation order, independent of process completion order.
    order = {image: i for i, image in enumerate(expected_images)}
    summary.sort(key=lambda row: order[row["image"]])
    report = finalize_dataset(config, expected, summary, root, renderer)
    return report, runtimes


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        choices=["test", "full"],
        help="test: 72 frames, 640x480, 4 samples; full: config defaults (8640 frames)",
    )
    p.add_argument("--config", default="configs/landing_dataset.yaml")
    p.add_argument(
        "--output",
        help="new/empty directory; default data/landing_server_test or data/landing",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(2, available_cpus()),
        help="GPU render processes (default 2)",
    )
    p.add_argument(
        "--plan-workers",
        type=int,
        default=available_cpus(),
        help="CPU planning processes (default up to 16)",
    )
    p.add_argument(
        "--threads",
        type=int,
        help="Cycles CPU threads per process (default CPUs/workers)",
    )
    p.add_argument("--device", choices=["OPTIX", "CUDA"], default="OPTIX")
    p.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Cycles GPU index within the chosen backend",
    )
    p.add_argument("--sorties-per-weather", type=int)
    p.add_argument("--frames-per-stage", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--samples", type=int)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    os.chdir(ROOT)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if min(args.workers, args.plan_workers) < 1 or args.gpu_index < 0:
        p.error("worker counts must be positive and GPU index nonnegative")
    args.threads = (
        args.threads
        if args.threads is not None
        else max(1, available_cpus() // args.workers)
    )
    if not 1 <= args.threads <= 64:
        p.error("--threads must be in [1, 64]")
    for name in ("frames_per_stage", "width", "samples"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.sorties_per_weather is not None and args.sorties_per_weather < 3:
        p.error("--sorties-per-weather must be at least 3")
    root = Path(
        args.output
        or ("data/landing_server_test" if args.mode == "test" else "data/landing")
    ).resolve()
    if root.exists() and any(root.iterdir()):
        p.error(f"Output is not empty: {root}; select a new --output directory")
    if not Path(args.config).is_file():
        p.error(f"Missing config: {args.config}")
    from shell.server_dependencies import setup

    # Setup must precede imports of numpy/yaml/torch, even on a fresh image.
    render_python = setup(
        device=args.device, gpu_index=args.gpu_index, threads=args.threads
    )
    import yaml
    from visual_training.data.generate_landing_dataset import plan_dataset, write_json
    from visual_training.data.validate_server_dataset import validate_dataset

    config = yaml.safe_load(Path(args.config).read_text())
    if args.mode == "test":
        config.update(sorties_per_weather=3, frames_per_stage=2)
        config["render"]["samples"] = 4
    for name in ("sorties_per_weather", "frames_per_stage"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    width = args.width or (640 if args.mode == "test" else config["camera"]["width"])
    config["camera"]["height"] = round(
        width * config["camera"]["height"] / config["camera"]["width"]
    )
    config["camera"]["width"] = width
    if args.samples is not None:
        config["render"]["samples"] = args.samples
    config["output"] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    # Never publish SUCCESS until worker merge AND training/data checks pass.
    start = time.monotonic()
    # Exclusive creation also rejects two commands that both checked an empty
    # output directory before their environment setup completed.
    with (root / "server_run.json").open("x") as stream:
        json.dump(
            dict(
                status="planning",
                arguments=vars(args),
                python=sys.version,
                render_python=render_python,
            ),
            stream,
            indent=2,
        )
    print(
        f"[plan] {args.plan_workers} CPU workers; "
        f"trajectory {config['trajectory_hz']} Hz with configured gimbal dynamics",
        flush=True,
    )
    plan = plan_dataset(config, root, workers=args.plan_workers)
    planning_s = time.monotonic() - start
    workers = min(args.workers, len(plan))
    write_json(
        root / "server_run.json",
        dict(
            status="rendering",
            arguments=vars(args),
            planning_s=planning_s,
            render_python=render_python,
        ),
    )
    print(
        f"[render] {len(plan)} sorties, {workers} {args.device} workers, "
        f"{args.threads} CPU threads each",
        flush=True,
    )
    run_workers(render_python, root, workers, args.threads, args.device, args.gpu_index)
    report, runtimes = merge_shards(config, plan, root, workers)
    validation = validate_dataset(
        root, all_images=args.mode == "test", training_device="cuda"
    )
    result = dict(
        status="complete",
        mode=args.mode,
        arguments=vars(args),
        frames=report["frames"],
        output=str(root),
        planning_s=planning_s,
        total_s=time.monotonic() - start,
        python=sys.version,
        render_python=render_python,
        workers=runtimes,
        validation=validation,
    )
    write_json(root / "server_run.json", result)
    write_json(
        root / "SUCCESS.json", dict(frames=report["frames"], total_s=result["total_s"])
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Dataset generation failed: {exc}")
