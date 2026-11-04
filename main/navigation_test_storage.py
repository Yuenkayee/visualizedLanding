"""Auditable sensor-cache retirement; numerical evidence is never pruned.

The receipt is committed BEFORE unlinking files. Resume can finish an interrupted
retirement only when every dependent result and the retained observations still
match their hashes. A damaged result requires regeneration of the entire group.
"""
import json
import math
from pathlib import Path

from main.navigation_test_io import digest, read_jsonl, write_json


def metadata_signature(record):
    record = Path(record)
    hashes = json.loads((record / "record_hashes.json").read_text())
    required = {"run.json", "steps.jsonl", "sensor_hashes.json"}
    if set(hashes) != required:
        raise ValueError("Invalid record metadata manifest")
    for name, expected in hashes.items():
        if digest(record / name) != expected:
            raise ValueError(f"Recorded metadata checksum mismatch: {record / name}")
    meta = json.loads((record / "run.json").read_text())
    if meta.get("status") != "complete":
        raise ValueError("Record is not complete")
    return digest(record / "record_hashes.json")


def verify_result(directory, input_record_sha256=None):
    directory = Path(directory)
    try:
        meta = json.loads((directory / "run.json").read_text())
        hashes = json.loads((directory / "result_hashes.json").read_text())
        if input_record_sha256 is not None and meta.get("input_record_sha256") != input_record_sha256:
            return False
        if meta.get("status") != "complete" or meta.get("completed_duration_s", 0)+1e-8 < meta["planned_duration_s"]:
            return False
        if set(hashes) != {"run.json", "steps.jsonl"}:
            return False
        return all(digest(directory / name) == expected for name, expected in hashes.items())
    except (OSError, KeyError, ValueError, TypeError):
        return False


def result_paths(output, job):
    return {f"{c['id']}/{m}": Path(output) / "results" / c["id"] / m
            for c in job["cases"] for m in c["methods"]}


def _local_sensors(record):
    """Only files owned by this generated archive may ever be removed."""
    record = Path(record)
    directory = record / "sensors"
    if record.is_symlink() or directory.is_symlink():
        raise ValueError("Refusing sensor cleanup through a symlink")
    hashes = json.loads((record / "sensor_hashes.json").read_text())
    for name in hashes:
        if Path(name).name != name or name in (".", "..") or not name.endswith(".npz"):
            raise ValueError(f"Not a locally owned sensor filename: {name}")
        if (directory / name).is_symlink():
            raise ValueError(f"Refusing sensor symlink: {name}")
    return directory, hashes


def compact_receipt(record):
    filename = Path(record) / "storage.json"
    if not filename.exists():
        return None
    receipt = json.loads(filename.read_text())
    if not isinstance(receipt, dict) or receipt.get("state") not in ("pruning", "pruned", "retained", "external"):
        raise ValueError("Invalid sensor storage receipt")
    if receipt.get("state") in ("pruning", "pruned"):
        return receipt
    return None


def _check_receipt(record, receipt):
    if receipt.get("schema_version") != 1 or receipt.get("mode") != "compact":
        raise ValueError("Unsupported sensor retirement receipt")
    if metadata_signature(record) != receipt.get("input_record_sha256"):
        raise ValueError("Sensor retirement receipt does not match input record")
    directory, hashes = _local_sensors(record)
    retained = receipt.get("retained_sensors")
    if not isinstance(retained, dict) or any(hashes.get(n) != h for n, h in retained.items()):
        raise ValueError("Invalid retained-sensor manifest")
    return directory, hashes, retained


def retired_results_valid(record, output, job, receipt):
    """False requests a complete group rebuild, never a partial replay."""
    directory, _, retained = _check_receipt(record, receipt)
    paths = result_paths(output, job)
    if set(paths) != set(receipt.get("result_manifests", {})):
        raise ValueError("Retired record's dependent methods changed")
    for key, folder in paths.items():
        if not verify_result(folder, receipt["input_record_sha256"]):
            return False
        if digest(folder / "result_hashes.json") != receipt["result_manifests"][key]:
            return False
    return all((directory / n).is_file() and digest(directory / n) == h for n, h in retained.items())


def retirement_result_errors(record, paths):
    """Analysis verifies numeric evidence; it need not read any retained images."""
    receipt = compact_receipt(record)
    if receipt is None:
        return {}
    _check_receipt(record, receipt)
    if set(paths) != set(receipt.get("result_manifests", {})):
        raise ValueError("Retired record's dependent results do not match the manifest")
    errors = {}
    for key, folder in paths.items():
        if (not verify_result(folder, receipt["input_record_sha256"]) or
                digest(folder / "result_hashes.json") != receipt["result_manifests"][key]):
            errors[key] = "Retired record result checksum/status/binding mismatch"
    return errors


def finish_retirement(record, receipt):
    """Idempotent after interruption. Caller must first verify every result."""
    directory, hashes, retained = _check_receipt(record, receipt)
    for name in hashes:
        if name not in retained:
            (directory / name).unlink(missing_ok=True)
    # These two Blender transport images are overwritten each frame, not evidence.
    for name in ("current_rgb.png", "current_mask.png"):
        (directory / name).unlink(missing_ok=True)
    receipt = dict(receipt, state="pruned")
    write_json(Path(record) / "storage.json", receipt)
    return receipt


def anomaly_windows(paths, config):
    """Select bounded diagnostic clips, NOT a subset for scientific statistics.

Keep the largest false-valid error and both ends of the longest outage, across
all methods/references sharing this record. N1 has no continuous-time validity.
All original numerical rows remain available, including every other failure.
"""
    error_limit = config["evaluation"].get("false_valid_error_m")
    outage_limit = config["evaluation"].get("max_outage_s")
    peak = longest = None
    references = (("gear", "gear_estimate_ned", "gear_truth_ned", "navigation_valid"),
                  ("cg", "cg_estimate_ned", "cg_truth_ned", "navigation_valid"),
                  ("cg_shaped", "cg_shaped_ned", "cg_truth_ned", "cg_signal_valid"))
    for key, folder in sorted(paths.items()):
        method = key.rsplit("/", 1)[1]
        outages = {}

        def end_outage(ref):
            nonlocal longest
            start, stop = outages.pop(ref)
            event = dict(result=key, reference=ref, start_s=start, end_s=stop, duration_s=stop-start)
            if outage_limit is not None and event["duration_s"] > outage_limit and (
                    longest is None or event["duration_s"] > longest["duration_s"]):
                longest = event

        for row in read_jsonl(folder / "steps.jsonl"):
            t, dt = float(row["timestamp"]), float(row["dt"])
            for ref, estimate, truth, valid in references:
                if method == "N1" and (ref == "cg_shaped" or not row.get("camera_sample")):
                    continue
                a, b = row.get(estimate), row.get(truth)
                finite = (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 3
                          and all(isinstance(v, (int, float)) and math.isfinite(v) for v in a+b))
                healthy = row.get(valid) is True and finite
                if healthy and error_limit is not None:
                    error = math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
                    if error > error_limit and (peak is None or error > peak["error_m"]):
                        peak = dict(result=key, reference=ref, timestamp=t, error_m=error)
                if method != "N1":
                    if ref in outages and (healthy or t > outages[ref][1]+1e-8):
                        end_outage(ref)
                    if not healthy:
                        start = outages.get(ref, (t, t))[0]
                        outages[ref] = (start, t+dt)
        for ref in list(outages):
            end_outage(ref)
    events = []
    if peak:
        events.append(dict(peak, kind="largest_false_valid_error"))
    if longest:
        events.extend(dict(longest, kind=kind, timestamp=longest[key]) for kind, key in
                      (("longest_outage_start", "start_s"), ("longest_outage_end", "end_s")))
    radius = config.get("anomaly_context_s", 1.)
    return [dict(e, window_start_s=e["timestamp"]-radius, window_end_s=e["timestamp"]+radius)
            for e in events[:config.get("anomaly_windows_per_record", 3)]]


def retire_record(record, output, job, config, external=False):
    """Verify a whole successful group, then retain all or bounded sensor data."""
    mode = config.get("sensor_retention", "all")
    if mode == "all":
        return None
    if mode != "compact":
        raise ValueError("Unknown sensor retention mode")
    signature = metadata_signature(record)
    paths = result_paths(output, job)
    if not all(verify_result(p, signature) for p in paths.values()):
        raise ValueError("Result verification failed; all sensors preserved")
    result_hashes = {key: digest(p / "result_hashes.json") for key, p in paths.items()}
    receipt = dict(schema_version=1, mode=mode, input_record_sha256=signature,
                   result_manifests=result_hashes, record_id=job["record_id"])
    if external:
        receipt.update(state="external", reason="Imported source observations are never removed")
    else:
        directory, hashes = _local_sensors(record)
        sizes = {}
        for name, expected in hashes.items():
            if digest(directory / name) != expected:
                raise ValueError(f"Sensor changed before retirement: {name}")
            sizes[name] = (directory / name).stat().st_size
        representative = job["cases"][0].get("replicate", 0) < config.get("retain_replicates", 1)
        windows = [] if representative else anomaly_windows(paths, config)
        keep = set(hashes) if representative else set()
        if windows:
            for row in read_jsonl(Path(record) / "steps.jsonl"):
                if (row["sensors"]["camera"] or row["sensors"]["lidar"]) and any(
                        w["window_start_s"]-1e-9 <= row["timestamp"] <= w["window_end_s"]+1e-9 for w in windows):
                    name = Path(row["sensor_file"]).name
                    if name not in hashes:
                        raise ValueError("Observation missing from sensor manifest")
                    keep.add(name)
        retained_bytes = sum(sizes[n] for n in keep)
        receipt.update(state="retained" if representative else "pruning",
                       reason="representative_replicate" if representative else "completed_group",
                       retained_sensors={n: hashes[n] for n in sorted(keep)}, anomaly_windows=windows,
                       original_sensor_files=len(hashes), retained_sensor_files=len(keep),
                       original_sensor_bytes=sum(sizes.values()), retained_sensor_bytes=retained_bytes,
                       released_sensor_bytes=sum(sizes.values())-retained_bytes)
    write_json(Path(record) / "storage.json", receipt)
    if receipt["state"] == "pruning":
        receipt = finish_retirement(record, receipt)
    return receipt


def storage_summary(root):
    """Account receipts without loading images; bytes are logical, not disk blocks."""
    records = []
    for folder in sorted((Path(root) / "records").glob("*")):
        if not folder.is_dir():
            continue
        filename = folder / "storage.json"
        if filename.exists():
            try:
                r = json.loads(filename.read_text())
                if not isinstance(r, dict):
                    raise ValueError("Storage receipt is not an object")
            except (OSError, ValueError) as exc:
                records.append(dict(record_id=folder.name, state="invalid", error=str(exc)))
                continue
            records.append({k: r.get(k) for k in ("record_id", "state", "reason", "original_sensor_files",
                "retained_sensor_files", "original_sensor_bytes", "retained_sensor_bytes", "released_sensor_bytes")})
        else:
            records.append(dict(record_id=folder.name, state="unpruned", reason="all policy, failed or unfinished group"))
    return dict(records=records, pruned_records=sum(r["state"] == "pruned" for r in records),
                released_sensor_bytes=sum((r.get("released_sensor_bytes") or 0) for r in records if r["state"] == "pruned"),
                retained_sensor_bytes_accounted=sum((r.get("retained_sensor_bytes") or 0) for r in records),
                note="Logical sensor bytes only; excludes numerical results/logs, unpruned/external records and filesystem overhead")
