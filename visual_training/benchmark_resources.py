"""Optional Linux/NVIDIA telemetry without adding Python dependencies."""

import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path


def read_number(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def available_memory():
    """Bytes available to this host/container (cgroup v2 where exposed)."""
    try:
        values = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
        }
        available = values["MemAvailable"]
    except (OSError, ValueError, KeyError):
        return None
    limit = read_number("/sys/fs/cgroup/memory.max")
    used = read_number("/sys/fs/cgroup/memory.current")
    if limit is not None and used is not None:
        available = min(available, max(0, limit - used))
    return available


def cpu_capacity():
    count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return min(count, int(quota) / int(period))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return float(count)


def inventory():
    memory = available_memory()
    return {
        "cpu_logical_count": os.cpu_count(),
        "cpu_capacity": cpu_capacity(),
        "available_memory_gib": memory / 2**30 if memory is not None else None,
        "shm_available_gib": shutil.disk_usage("/dev/shm").free / 2**30
        if Path("/dev/shm").exists()
        else None,
    }


def process_group_usage():
    """RSS is summed (shared pages can be counted more than once)."""
    if not Path("/proc/self/stat").exists():
        return None
    group = os.getpgrp()
    ticks = rss_pages = 0
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                ticks += int(fields[11]) + int(fields[12])
                rss_pages += int(fields[21])
        except (OSError, ValueError, IndexError):
            continue
    return ticks / os.sysconf("SC_CLK_TCK"), rss_pages * os.sysconf("SC_PAGE_SIZE")


def summarize(samples):
    return {
        key: {"mean": statistics.mean(values), "max": max(values), "min": min(values)}
        for key in sorted({key for row in samples for key in row if key != "time_s"})
        if (values := [row[key] for row in samples if row.get(key) is not None])
    }


class ResourceMonitor:
    def __init__(self, gpu_uuid=None):
        self.gpu_uuid = gpu_uuid
        self.samples = []
        self.errors = set()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        started = time.perf_counter()
        previous = None
        while not self.stop_event.is_set():
            now = time.perf_counter()
            row = {"time_s": now - started}
            usage = process_group_usage()
            if usage:
                cpu_seconds, rss = usage
                row["process_group_rss_gib"] = rss / 2**30
                if previous:
                    cores = max(0, cpu_seconds - previous[1]) / (now - previous[0])
                    row["process_group_cpu_cores"] = cores
                    row["allocated_cpu_percent"] = 100 * cores / cpu_capacity()
                previous = (now, cpu_seconds)
            memory = available_memory()
            if memory is not None:
                row["available_memory_gib"] = memory / 2**30
            if self.gpu_uuid:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--id={self.gpu_uuid}",
                            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=True,
                    )
                    for key, value in zip(
                        (
                            "gpu_util_percent",
                            "gpu_memory_util_percent",
                            "gpu_memory_used_mib",
                            "gpu_power_w",
                            "gpu_temperature_c",
                        ),
                        result.stdout.strip().split(","),
                    ):
                        try:
                            row[key] = float(value.strip())
                        except ValueError:
                            pass
                except (OSError, subprocess.SubprocessError) as error:
                    self.errors.add(str(error))
            self.samples.append(row)
            self.stop_event.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def report(self):
        return {
            "summary": summarize(self.samples),
            "samples": self.samples,
            "errors": sorted(self.errors),
            "gpu_uuid": self.gpu_uuid,
        }
