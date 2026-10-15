"""Server training launcher; bootstrap without importing third-party packages."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_logged(command, log):
    """Include stdout/stderr from dependency installation and training in logs."""
    with Path(log).open("a") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            result = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if result:
            raise RuntimeError(f"Command exited {result}; see {log}")


def validate_data(root, formal=False, allow_incomplete_preparation=False, weathers=None):
    """Validate split membership before launching a detached training process."""
    root = Path(root)
    if (root / "preparation.json").exists() and not allow_incomplete_preparation:
        prepared = json.loads((root / "preparation.json").read_text())
        if prepared.get("status") != "complete" or not (root / "SUCCESS.json").is_file():
            raise ValueError("Prepared dataset is incomplete; require SUCCESS.json")
    if (root / "server_run.json").exists():
        run = json.loads((root / "server_run.json").read_text())
        if run.get("status") != "complete" or not (root / "SUCCESS.json").is_file():
            raise ValueError(
                "Server dataset is incomplete; wait for generation SUCCESS.json"
            )
        if formal and run.get("mode") == "test":
            raise ValueError(
                "Full training requires the full dataset, not the 90-frame generation test"
            )
    if (root / "dataset_report.json").exists():
        if json.loads((root / "dataset_report.json").read_text()).get("preview_only"):
            raise ValueError("Preview images are not a training dataset")
    seen, sequences, counts = set(), {}, {}
    for split in ("train", "val", "test"):
        paths = json.loads((root / "splits" / f"{split}.json").read_text())
        if not paths:
            raise ValueError(f"Empty {split} split")
        counts[split] = 0
        for path in paths:
            from visual_training.data.manifest import resolve_annotation
            label = resolve_annotation(path, root).resolve()
            if label in seen:
                raise ValueError(f"Repeated annotation across splits: {label}")
            seen.add(label)
            row = json.loads(label.read_text())
            sequence = row["sequence_id"]
            if (
                sequences.setdefault(sequence, split) != split
                or row.get("split", split) != split
            ):
                raise ValueError(f"Sortie leaks across splits: {sequence}")
            if weathers is not None and row.get("weather") not in weathers:
                continue
            counts[split] += 1
            for key in ("image", "mask"):
                artifact = root / row[key]
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError(f"Missing/empty {key}: {artifact}")
        if counts[split] == 0:
            raise ValueError(f"Empty {split} split after weather selection")
    return counts


def execute_job(job_path):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    status = job_path.parent / "status.json"
    write_json(status, dict(status="running", mode=job["mode"], pid=os.getpid()))
    try:
        from visual_training.train import train

        checkpoint = train(job["config"])
        write_json(
            status,
            dict(
                status="complete",
                mode=job["mode"],
                exit_code=0,
                checkpoint=str(checkpoint) if checkpoint else None,
            ),
        )
    except BaseException as exc:
        write_json(
            status,
            dict(
                status="failed",
                mode=job["mode"],
                exit_code=1,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def tmux_command(session, job_path, log_path):
    # tmux servers retain their original environment. Explicitly pass the
    # selected interpreter and relevant runtime variables into the new session.
    environment = {
        name: os.environ[name]
        for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CONDA_PREFIX")
        if name in os.environ
    }
    command = [
        "env",
        *(f"{k}={v}" for k, v in environment.items()),
        "bash",
        str(ROOT / "shell/run_training_job.sh"),
        sys.executable,
        str(job_path),
        str(log_path),
    ]
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-c",
        str(ROOT),
        shlex.join(command),
    ]


def build_config(args, run_dir):
    import yaml
    from visual_training.data.generate_server_dataset import available_cpus

    config = yaml.safe_load(Path(args.config).read_text())
    config["num_workers"] = min(
        config.get("num_workers", 8), max(0, available_cpus() - 2)
    )
    config["threads"] = min(config.get("threads", 4), available_cpus())
    for key in (
        "data_root",
        "image_size",
        "epochs",
        "num_workers",
        "threads",
        "max_batch_size",
        "precision",
    ):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.batch_size is not None:
        config["batch_size"] = (
            args.batch_size if args.batch_size == "auto" else int(args.batch_size)
        )
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config.update(
        device="cuda", log_dir=str(run_dir), save_checkpoints=args.mode == "full"
    )
    if args.mode == "pretrain":
        config["epochs"] = args.epochs or 2
        config["max_train_batches"] = args.train_batches
        config["max_val_batches"] = args.val_batches
        config.pop("output", None)
    else:
        config.pop("max_train_batches", None)
        config.pop("max_val_batches", None)
        output = Path(
            args.output or f"visual_training/checkpoints/landing/{run_dir.name}"
        ).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                f"Checkpoint output is not empty: {output}; use a new --output"
            )
        # Exclusive ownership prevents two formal jobs overwriting each other.
        output.mkdir(parents=True, exist_ok=True)
        with (output / "training_job.json").open("x") as stream:
            json.dump(dict(log_dir=str(run_dir)), stream)
        config["output"] = str(output)
    return config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        nargs="?",
        choices=["pretrain", "full"],
        help="pretrain: foreground check, no weights; full: tmux training",
    )
    p.add_argument("--config", default="configs/landing_training_server.yaml")
    p.add_argument(
        "--data-root",
        help="default data/landing_four_weather; pretrain may use data/landing_server_test",
    )
    p.add_argument("--output", help="full mode checkpoint directory; must be new/empty")
    p.add_argument("--log-root", default="outputs/logs/training")
    p.add_argument(
        "--session", default="landing_train", help="tmux session name for full mode"
    )
    p.add_argument("--epochs", type=int, help="default pretrain=2, full=40")
    p.add_argument(
        "--train-batches",
        type=int,
        default=50,
        help="pretrain maximum batches per epoch",
    )
    p.add_argument(
        "--val-batches",
        type=int,
        default=10,
        help="pretrain maximum validation batches per epoch",
    )
    p.add_argument("--batch-size", help="auto (default) or a positive integer")
    p.add_argument(
        "--max-batch-size", type=int, help="auto probe upper bound, default 16"
    )
    p.add_argument("--image-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument("--precision", choices=["bf16", "fp32"])
    p.add_argument("--run-job", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    os.chdir(ROOT)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    if args.run_job:
        execute_job(args.run_job)
        return
    if not args.mode:
        p.error("choose pretrain or full")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.session):
        p.error("session name must contain only letters, digits, '_' or '-'")
    for key in ("epochs", "train_batches", "val_batches", "max_batch_size", "threads"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        p.error("num-workers must be nonnegative")
    if args.image_size is not None and args.image_size < 8:
        p.error("image-size must be >=8")
    if args.batch_size is not None and args.batch_size != "auto":
        if not args.batch_size.isdigit() or int(args.batch_size) < 1:
            p.error("batch-size must be auto or a positive integer")
    if args.mode == "pretrain" and args.output:
        p.error("pretrain does not save weights; --output applies only to full mode")
    if not Path(args.config).is_file():
        p.error(f"Config not found: {args.config}")
    run_id = (
        f"{args.mode}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    run_dir = Path(args.log_root).resolve() / run_id
    run_dir.mkdir(parents=True)
    print(f"Logs: {run_dir}", flush=True)
    try:
        command = [
            sys.executable,
            "-u",
            str(ROOT / "shell/server_dependencies.py"),
            "--training",
        ]
        if args.mode == "full":
            command.append("--with-tmux")
        run_logged(command, run_dir / "setup.log")
        if args.mode == "full":
            result = subprocess.run(
                ["tmux", "has-session", "-t", f"={args.session}"], capture_output=True
            )
            if result.returncode == 0:
                raise ValueError(
                    f"tmux session {args.session} already exists; attach or choose --session"
                )
        root = args.data_root
        if not root:
            import yaml

            root = yaml.safe_load(Path(args.config).read_text())["data_root"]
        import yaml
        selected_weather = yaml.safe_load(Path(args.config).read_text()).get("weathers")
        counts = validate_data(Path(root).resolve(), formal=args.mode == "full", weathers=selected_weather)
        config = build_config(args, run_dir)
        write_json(
            run_dir / "job.json",
            dict(mode=args.mode, config=config, split_counts=counts),
        )
        write_json(
            run_dir / "status.json",
            dict(
                status="starting",
                mode=args.mode,
                session=args.session if args.mode == "full" else None,
            ),
        )
        job_path, log_path = run_dir / "job.json", run_dir / "train.log"
        if args.mode == "pretrain":
            run_logged(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "visual_training.server_train",
                    "--run-job",
                    str(job_path),
                ],
                log_path,
            )
            print(f"Pretrain completed; logs only: {run_dir}", flush=True)
        else:
            subprocess.run(tmux_command(args.session, job_path, log_path), check=True)
            print(
                f"Training launched in tmux: {args.session}\n"
                f"Attach: tmux attach -t {args.session}\n"
                f"Log: tail -f {shlex.quote(str(log_path))}\n"
                f"Status: {run_dir / 'status.json'}\n"
                f"Checkpoints: {config['output']}",
                flush=True,
            )
    except BaseException as exc:
        status = run_dir / "status.json"
        if (
            not status.exists()
            or json.loads(status.read_text()).get("status") != "complete"
        ):
            write_json(
                status,
                dict(
                    status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}"
                ),
            )
        with (run_dir / "launcher.log").open("a") as stream:
            traceback.print_exc(file=stream)
        raise


if __name__ == "__main__":
    main()
