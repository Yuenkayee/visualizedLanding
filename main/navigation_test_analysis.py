"""Navigation-only analysis for thesis 5.2.3 (tables 5-13/14, figures 5-8/9/10).

``analyze(root, bootstrap_samples=2000)`` writes artifacts directly under root
and returns a JSON-serializable, flight-level summary (never all the frames).
The nondefault argument controls bootstrap replication; otherwise the manifest
count (or 2000) is used. manifest.evaluation supplies the
seed and OPTIONAL requirement thresholds. No algorithm gate is an evaluation
requirement. Missing numbers are null/blank, never zero.

Contract clarifications for producers:
* timestamp is elapsed seconds, or run.start_timestamp specifies its origin.
  dt describes [timestamp, timestamp + dt); gaps and the uncompleted planned
  tail are uncovered. Overlapping cells are clipped at the next timestamp.
* N1 is scored ONLY on camera_sample rows, including its explicitly logged dt
  support. No positions, covariance, validity or diagnostics are interpolated
  or carried to another row. N1 shaped output is never scored.
* diagnostics/quality may contain held values. Vision events require a camera
  row. OPTIONAL lidar_sample is needed for LiDAR event rates; without it these
  rates are unidentifiable, not inferred from a held lidar_accepted flag.
* cg_shaped uses cg_signal_valid; raw gear/cg use navigation_valid. Covariance
  is matched by reference and NED coordinates. Shaped covariance is unknown.
* pair_id identifies a weather/motion/seed pair; condition defaults to baseline.
  source_record (or source_id) identifies reused input recordings and
  prevents treating copies of an imported recording as independent flights.
* optional manifest.methods / case.methods declare expected methods. Otherwise
  main expects N0..N3,A1..A3; gimbal/stress expect N3, plus discovered methods.
  A generation failure with no results is still a case-level failure.

Errors use original sample times and sample means, availability uses elapsed
time, and inference resamples independent flights within weather/motion/gimbal
strata. Complete, incomplete and excluded runs are always separate. Confidence
intervals are descriptive percentile intervals, without significance claims.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate
from scipy.stats import chi2
from main.navigation_test_storage import retirement_result_errors, storage_summary


STAGES = ("all", "approach", "hover", "descent")
REFERENCES = ("gear", "cg", "cg_shaped")
METHODS = ("N0", "N1", "N2", "N3", "A1", "A2", "A3")
SETS = ("finite", "declared_valid")
QUALITY = (
    "mask_iou", "keypoint_error_px", "keypoint_error_normalized",
    "target_scale_px", "edge_margin_px", "truncation_fraction",
    "gimbal_saturated", "gimbal_reference_error_rad", "lidar_points",
    "plane_inlier_ratio", "plane_rms",
)
DIAGNOSTICS = (
    "vision_accepted", "lidar_accepted", "reprojection_error",
    "vision_mahalanobis", "lidar_mahalanobis", "vision_rejection",
    "lidar_rejection", "vision_source", "vision_path", "vision_roi_attempted",
    "vision_reacquired", "vision_recovery_pending", "vision_position_correction_m",
    "cg_tracking_error_m", "cg_correction_limited",
    "vision_mahalanobis_threshold", "lidar_mahalanobis_threshold",
    "pnp_solved", "lidar_position_correction_m", "lidar_associated_points",
    "lidar_association_rejection_fraction",
)
TIMINGS = ("render_s", "matlab_s", "inference_s", "geometry_s", "filter_s",
           "camera_s", "lidar_s", "total_s")
META = ("case_id", "pair_id", "weather", "motion", "seed", "suite", "condition",
        "gimbal_mode", "method", "cohort", "independent_unit")
DIMENSIONS = META + ("reference", "stage", "sample_set", "common_methods")
CHI2_95 = float(chi2.ppf(.95, 3))
REPORT_METRICS = {
    "rmse_3d_m", "rmse_n_m", "rmse_e_m", "rmse_d_m", "bias_n_m", "bias_e_m", "bias_d_m",
    "p95_3d_m", "median_3d_m", "max_3d_m", "attitude_rmse_rad", "nees_mean", "chi2_95_coverage",
    "J_m_mean", "J_m_max", "availability", "camera_coverage", "first_valid_s", "longest_outage_s",
    "outage_count", "false_valid_events", "false_valid_duration_s", "outage_limit_exceedances",
    "recovery_time_s_mean", "vision_acceptance_fraction", "lidar_acceptance_fraction", "pnp_solve_fraction",
    "vision_innovation_rejection_fraction", "lidar_innovation_rejection_fraction",
    "lidar_association_rejection_fraction_pooled", "shaping_residual_m_mean", "lag_s",
    "seconds_mean", "seconds_p95", "seconds_max",
}


def _number(value):
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return float(value) if np.isfinite(value) else None
    return None


def _vector(value):
    try:
        a = np.asarray(value, dtype=float)
        return a if a.shape == (3,) and np.isfinite(a).all() else None
    except (TypeError, ValueError):
        return None


def _covariance(value):
    try:
        a = np.asarray(value, dtype=float)
        if a.shape != (3, 3) or not np.isfinite(a).all() or not np.allclose(a, a.T):
            return None
        np.linalg.cholesky(a)
        return a
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _clean(value):
    """Make even non-standard input JSON NaN safe in every output artifact."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _json(path, value):
    path.write_text(json.dumps(_clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _cell(value):
    return json.dumps(_clean(value), ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else _clean(value)


def _csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row)) or ["case_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)


def _stats(values, prefix):
    a = np.array([x for x in values if _number(x) is not None], dtype=float)
    return {f"{prefix}_n": len(a), f"{prefix}_mean": float(a.mean()) if len(a) else None,
            f"{prefix}_p95": float(np.percentile(a, 95)) if len(a) else None,
            f"{prefix}_max": float(a.max()) if len(a) else None}


@dataclass
class Run:
    case: dict
    method: str
    path: Path
    info: dict
    cohort: str
    issues: list

    @property
    def meta(self):
        c = self.case
        source = next((self.info.get(k) or c.get(k) for k in
                       ("source_record", "source_id")
                       if self.info.get(k) or c.get(k)), None)
        return dict(case_id=c["id"], pair_id=c.get("pair_id", c["id"]),
                    weather=c.get("weather", "unknown"), motion=c.get("motion", "unknown"),
                    seed=c.get("seed"), suite=c.get("suite", "main"),
                    condition=c.get("condition", "baseline"),
                    gimbal_mode=c.get("gimbal_mode", "unknown"), method=self.method,
                    cohort=self.cohort,
                    independent_unit="source:" + str(source) if source else "pair:" + str(c.get("pair_id", c["id"])))

    @property
    def planned(self):
        return _number(self.info.get("planned_duration_s"))

    @property
    def completed(self):
        return _number(self.info.get("completed_duration_s"))


def _rows(run, audit=False):
    """Read one run at a time; preserve salvageable rows of interrupted JSONL."""
    path = run.path / "steps.jsonl"
    if not path.exists():
        if audit:
            run.issues.append("missing steps.jsonl")
        return
    previous = -np.inf
    origin = _number(run.info.get("start_timestamp", run.info.get("start_time_s", 0)))
    if origin is None:
        origin = 0.
        if audit:
            run.issues.append("invalid start_timestamp; elapsed origin 0 used")
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                t, dt = _number(row.get("timestamp")), _number(row.get("dt"))
                if t is None or dt is None or dt <= 0 or t - origin < 0 or t <= previous:
                    raise ValueError("timestamp/dt invalid or nonmonotonic")
                previous = t
                row = dict(row, timestamp=t-origin, dt=dt)
                for key in ("diagnostics", "quality", "timing"):
                    if not isinstance(row.get(key), dict):
                        row[key] = {}
                yield row
            except (ValueError, TypeError) as exc:
                if audit:
                    run.issues.append(f"steps.jsonl:{line_no}: {exc}")


def _record_check_errors(root, runs):
    """An interrupted shared-record check cannot rehabilitate old result files."""
    grouped, errors, result_errors = defaultdict(list), defaultdict(list), defaultdict(list)
    for run in runs:
        grouped[run.case.get("record_id", run.case["id"])].append(run)
    for rid, dependent in grouped.items():
        if not isinstance(rid, str) or Path(rid).name != rid or rid in (".", ".."):
            raise ValueError(f"invalid record id: {rid!r}")
        marker = root / "record_checks" / (rid + ".json")
        if not marker.exists():
            continue  # Legacy outputs did not persist record-check state.
        try:
            check = json.loads(marker.read_text())
            if not isinstance(check, dict):
                raise ValueError("record check is not an object")
        except (OSError, ValueError) as exc:
            errors[rid].append(f"record evidence invalid: unreadable record check: {exc}")
            continue
        if check.get("status") != "passed":
            errors[rid].append(f"record evidence invalid: record check status {check.get('status')!r}; "
                               f"{check.get('error', 'validation did not finish')}")
            continue
        signature = check.get("input_record_sha256")
        if not isinstance(signature, str) or not signature:
            errors[rid].append("record evidence invalid: passed record check lacks input_record_sha256")
            continue
        for run in dependent:
            # A missing result already has no usable evidence. It must not
            # invalidate a different method that replayed this valid record.
            if run.info.get("status") == "missing":
                continue
            if run.info.get("input_record_sha256") != signature:
                result_errors[run.case["id"], run.method].append(
                    "record evidence invalid: input_record_sha256 missing or mismatched for "
                    f"{run.case['id']}/{run.method}")
    for rid, dependent in grouped.items():
        try:
            invalid = retirement_result_errors(root / "records" / rid,
                {f"{r.case['id']}/{r.method}": r.path for r in dependent})
            for run in dependent:
                key = f"{run.case['id']}/{run.method}"
                if key in invalid:
                    result_errors[run.case["id"], run.method].append(invalid[key])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors[rid].append(f"Invalid sensor-retirement evidence: {exc}")
    return errors, result_errors


def _discover(root, manifest):
    runs, seen = [], set()
    failures = [dict(kind="runner_failure", **failure) for failure in manifest.get("failures", [])]
    cases_by_id = {case["id"]: case for case in manifest["cases"]}
    cases_by_record = defaultdict(set)
    for case in manifest["cases"]:
        if case.get("record_id") is not None:
            cases_by_record[case["record_id"]].add(case["id"])
    record_errors, replay_errors = defaultdict(list), defaultdict(list)
    # Only the current attempt's failures invalidate evidence. Historical
    # attempt failures may have been repaired by a successful resume.
    for failure in manifest.get("failures", []):
        cid = failure.get("case", failure.get("case_id"))
        if failure.get("phase") == "record":
            rid = failure.get("record_id", cases_by_id.get(cid, {}).get("record_id"))
            for affected in cases_by_record.get(rid, set()) | ({cid} if cid is not None else set()):
                record_errors[affected].append("record evidence invalid: " + str(failure.get("error", "record failed")))
        elif failure.get("phase") == "replay":
            replay_errors[cid, failure.get("method")].append(
                "replay evidence invalid: " + str(failure.get("error", "replay failed")))
    for case in manifest["cases"]:
        cid = case["id"]
        if not isinstance(cid, str) or cid in seen or Path(cid).name != cid or cid in (".", ".."):
            raise ValueError(f"invalid or duplicate case id: {cid!r}")
        seen.add(cid)
        folder = root / "results" / cid
        discovered = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.exists() else []
        expected = case.get("methods", manifest.get("methods"))
        if expected is None:
            expected = METHODS if case.get("suite", "main") == "main" else ("N3",)
        methods = sorted(set(expected) | set(discovered))
        if not discovered:
            failures.append(dict(case_id=cid, method=None, kind="generation_or_missing_results",
                                 status=case.get("status", "missing_results"),
                                 error=case.get("error"), case_metadata=case))
        for method in methods:
            if not isinstance(method, str) or Path(method).name != method or method in (".", ".."):
                raise ValueError(f"invalid method: {method!r}")
            path, issues = folder / method, []
            try:
                info = json.loads((path / "run.json").read_text())
                if not isinstance(info, dict):
                    raise ValueError("run.json is not an object")
            except (OSError, ValueError) as exc:
                info = dict(status="missing", error=str(exc))
                issues.append("missing or unreadable run.json")
            invalid = list(dict.fromkeys(record_errors[cid] + replay_errors[cid, method]))
            issues.extend(invalid)
            planned, completed = _number(info.get("planned_duration_s")), _number(info.get("completed_duration_s"))
            complete = (not invalid and info.get("status") == "complete" and planned is not None and planned > 0
                        and completed is not None and completed >= planned - 1e-9
                        and (path / "steps.jsonl").exists())
            excluded = (info.get("excluded_from_formal") is True or case.get("excluded_from_formal") is True
                        or manifest.get("excluded_from_formal") is True
                        or case.get("suite") == "smoke")
            cohort = ("excluded_" if excluded else "") + ("complete" if complete else "incomplete")
            run = Run(case, method, path, info, cohort, issues)
            runs.append(run)
    marker_errors, binding_errors = _record_check_errors(root, runs)
    for run in runs:
        invalid = (marker_errors[run.case.get("record_id", run.case["id"])]
                   + binding_errors[run.case["id"], run.method])
        if invalid:
            run.issues.extend(invalid)
            run.cohort = "excluded_incomplete" if run.cohort.startswith("excluded_") else "incomplete"
        if run.cohort.endswith("incomplete"):
            failures.append(dict(case_id=run.case["id"], method=run.method, kind="run_incomplete", cohort=run.cohort,
                                 status=run.info.get("status", "unknown"), error=run.info.get("error"),
                                 issues=list(run.issues), planned_duration_s=run.planned,
                                 completed_duration_s=run.completed))
    return runs, failures


def _run_resources(info, manifest):
    """Resolve only runtime evidence explicitly linked to this result's origin."""
    attempt_id = info.get("attempt_id")
    runtime = info.get("runtime")
    source = "run" if isinstance(runtime, dict) and runtime else "unknown"
    if source == "unknown" and attempt_id is not None:
        attempt = next((a for a in manifest.get("attempts", [])
                        if a.get("attempt_id") == attempt_id), None)
        if attempt is not None:
            runtime = attempt.get("runtime")
            source = "manifest_attempt"
        elif manifest.get("attempt_id") == attempt_id:
            runtime = manifest.get("runtime")
            source = "manifest_current_attempt"
    if not isinstance(runtime, dict) or not runtime:
        runtime, source = None, "unknown"
    aliases = dict(precision="inference_precision", gpu_synchronization="gpu_timing")
    resources = {key: info.get(key, (runtime or {}).get(aliases.get(key, key)))
                 for key in ("device", "torch_threads", "render_threads", "resolution", "precision",
                             "gpu_synchronization", "execution", "gpu", "cuda", "torch", "python",
                             "platform", "executable")}
    provenance = dict(attempt_id=attempt_id, runtime=runtime, runtime_source=source,
                      input_record_sha256=info.get("input_record_sha256"),
                      recording_runtime=info.get("recording_runtime"),
                      recording_attempt_id=info.get("recording_attempt_id"))
    return resources, provenance


def _provenance_summary(run_records):
    attempts = sorted({r["attempt_id"] for r in run_records if r["attempt_id"] is not None})
    recording_attempts = sorted({r["recording_attempt_id"] for r in run_records if r["recording_attempt_id"] is not None})
    unknown_attempts = sum(r["attempt_id"] is None for r in run_records)
    unknown_recording_attempts = sum(r["recording_attempt_id"] is None for r in run_records)
    groups = {}
    for run in run_records:
        for kind, attempt, runtime in (("replay", run["attempt_id"], run["runtime"]),
                                       ("recording", run["recording_attempt_id"], run["recording_runtime"])):
            key = (kind, attempt, json.dumps(runtime, sort_keys=True))
            group = groups.setdefault(key, dict(kind=kind, attempt_id=attempt, runtime=runtime, runs=0))
            group["runs"] += 1
    return dict(result_attempt_ids=attempts, recording_attempt_ids=recording_attempts,
                mixed_attempts=True if len(set(attempts + recording_attempts)) > 1
                else None if unknown_attempts or unknown_recording_attempts else False,
                unknown_attempt_runs=unknown_attempts,
                unknown_recording_attempt_runs=unknown_recording_attempts,
                unknown_runtime_runs=sum(r["runtime"] is None for r in run_records),
                unknown_recording_runtime_runs=sum(not r["recording_runtime"] for r in run_records),
                originating_runtimes=list(groups.values()))


def _provenance_report(summary):
    provenance = summary["provenance"]
    mixed = {True: "yes", False: "no", None: "unknown"}[provenance["mixed_attempts"]]
    lines = ["## Originating runtime provenance", "",
             f"Replay attempt IDs: `{json.dumps(provenance['result_attempt_ids'])}`. Mixed attempts: {mixed}. "
             f"Unknown attempt: {provenance['unknown_attempt_runs']} runs; "
             f"unknown recording attempt: {provenance['unknown_recording_attempt_runs']} runs; "
             f"unknown replay runtime: {provenance['unknown_runtime_runs']} runs; "
             f"unknown recording runtime: {provenance['unknown_recording_runtime_runs']} runs.", "",
             "The current manifest runtime describes the latest coordinator attempt, not every retained measurement. "
             "runs.json retains each result's attempt_id, originating runtime and its source, input_record_sha256, "
             "and separate recording_runtime for generation timings. Attempt history, including runtime and workers, "
             "is retained in summary.json. Unlinked legacy results remain unknown; they never inherit the latest hardware.", ""]
    for group in provenance["originating_runtimes"]:
        lines.append(f"- {group['kind'].capitalize()} attempt `{group['attempt_id'] or 'unknown'}`: "
                     f"{group['runs']} result(s); "
                     f"originating runtime `{json.dumps(group['runtime'], ensure_ascii=False)}`.")
    lines.extend(["", "Timing aggregates may mix attempts or unknown environments. Use per-run provenance to select "
                  "comparable measurements; a pooled value does not describe a single machine or isolated latency.", ""])
    return lines


def _eligible(row, method, ref):
    return not (method == "N1" and (row.get("camera_sample") is not True or ref == "cg_shaped"))


def _values(row, method, ref):
    cache = row.setdefault("_analysis_values", {})
    if (method, ref) in cache:
        return cache[method, ref]
    truth = _vector(row.get("cg_truth_ned" if ref == "cg_shaped" else ref + "_truth_ned"))
    estimate = _vector(row.get("cg_shaped_ned" if ref == "cg_shaped" else ref + "_estimate_ned"))
    if (ref != "cg_shaped" and row.get("estimate_available") is not True) or not _eligible(row, method, ref):
        estimate = None
    error = estimate - truth if estimate is not None and truth is not None else None
    flag = "cg_signal_valid" if ref == "cg_shaped" else "navigation_valid"
    valid = row.get(flag) is True
    covariance = _covariance(row.get(ref + "_covariance_ned")) if ref != "cg_shaped" else None
    cache[method, ref] = truth, estimate, error, valid, covariance
    return cache[method, ref]


def _common_times(case_runs):
    by_method = {}
    for run in case_runs:
        times = {ref: set() for ref in REFERENCES}
        for row in _rows(run):
            if row.get("camera_sample") is True:
                for ref in REFERENCES:
                    if _values(row, run.method, ref)[2] is not None:
                        times[ref].add(row["timestamp"])
        by_method[run.method] = times
    groups = {"common_camera": sorted(set(by_method) & {"N0", "N1", "N2", "N3"})}
    if set(groups["common_camera"]) != set(by_method):
        groups["common_camera_all_methods"] = sorted(by_method)
    for method in sorted(set(by_method)-{"N0", "N1", "N2", "N3"}):
        if "N3" in by_method:
            groups["common_camera_pair_"+method] = [method, "N3"]
    result = {}
    for name, methods in groups.items():
        if methods:
            result[name] = (methods, {ref: set.intersection(*(by_method[m][ref] for m in methods)) for ref in REFERENCES})
    return result


def _accuracy(rows, method, ref, sample_set, common):
    selected, errors, cov_errors, jumps, angles = [], [], [], [], []
    previous = None
    for index, row in enumerate(rows):
        _, _, error, valid, covariance = _values(row, method, ref)
        take = error is not None
        if sample_set == "declared_valid":
            take = take and valid
        if sample_set.startswith("common_camera"):
            take = take and row.get("camera_sample") is True and row["timestamp"] in common
        if not take:
            # N1's noncamera rows are not samples. Missing camera outputs and
            # invalid samples break the increment chain; neither is bridged.
            if _eligible(row, method, ref):
                previous = None
            continue
        selected.append(index)
        errors.append(error)
        if previous is not None:
            jumps.append(float(np.linalg.norm(error - previous)))
        previous = error
        if covariance is not None:
            nees = float(error @ np.linalg.solve(covariance, error))
            if np.isfinite(nees):
                cov_errors.append(nees)
        angle = _number(row.get("attitude_error_rad"))
        if angle is not None and angle >= 0:
            angles.append(angle)
    e = np.asarray(errors).reshape(-1, 3)
    distances = np.linalg.norm(e, axis=1)
    metrics = dict(samples=len(e), rmse_3d_m=float(np.sqrt(np.mean(distances**2))) if len(e) else None,
                   median_3d_m=float(np.median(distances)) if len(e) else None,
                   p95_3d_m=float(np.percentile(distances, 95)) if len(e) else None,
                   max_3d_m=float(distances.max()) if len(e) else None,
                   attitude_samples=len(angles),
                   attitude_rmse_rad=float(np.sqrt(np.mean(np.square(angles)))) if angles else None,
                   nees_samples=len(cov_errors), nees_mean=float(np.mean(cov_errors)) if cov_errors else None,
                   chi2_95_coverage=float(np.mean(np.asarray(cov_errors) <= CHI2_95)) if cov_errors else None,
                   **_stats(jumps, "J_m"))
    for axis, name in enumerate("ned"):
        metrics[f"bias_{name}_m"] = float(e[:, axis].mean()) if len(e) else None
        metrics[f"rmse_{name}_m"] = float(np.sqrt(np.mean(e[:, axis]**2))) if len(e) else None
    return metrics


def _segments(rows, run, stage):
    """Partition planned time; unknown stages are never assigned to descent."""
    planned, completed = run.planned, run.completed
    if planned is None or planned <= 0 or completed is None or completed < 0:
        return None
    end = min(planned, completed)
    cells, cursor = [], 0.
    for i, row in enumerate(rows):
        start = min(row["timestamp"], end)
        stop = min(row["timestamp"] + row["dt"], end)
        if i + 1 < len(rows):
            stop = min(stop, rows[i+1]["timestamp"])
        if start > cursor and stage == "all":
            cells.append((cursor, start, None))
        if stop > start:
            if stage == "all" or row.get("stage") == stage:
                cells.append((start, stop, row))
            cursor = stop
    if cursor < planned and stage == "all":
        cells.append((cursor, planned, None))
    return cells


def _episodes(cells, predicate):
    episodes = []
    for start, stop, row in cells:
        if predicate(row):
            if episodes and abs(episodes[-1][1] - start) < 1e-9:
                episodes[-1][1] = stop
            else:
                episodes.append([start, stop])
    return episodes


def _availability(rows, run, ref, stage, evaluation):
    camera = [r for r in rows if r.get("camera_sample") is True and (stage == "all" or r.get("stage") == stage)]
    scored = sum(_values(r, run.method, ref)[2] is not None for r in camera)
    metrics = dict(camera_samples=len(camera), camera_scored_samples=scored,
                   camera_coverage=scored / len(camera) if camera else None)
    fields = ("duration_denominator_s", "observed_duration_s", "covered_duration_s", "uncovered_duration_s",
              "valid_duration_s", "availability", "first_valid_s", "outage_count", "longest_outage_s",
              "false_valid_events", "false_valid_duration_s", "false_valid_scored_samples",
              "outage_limit_exceedances")
    metrics.update(dict.fromkeys(fields))
    cells = _segments(rows, run, stage)
    if cells is None:
        return metrics, []
    def output(row):
        return row is not None and _values(row, run.method, ref)[1] is not None
    def valid(row):
        return output(row) and _values(row, run.method, ref)[3]
    duration = sum(b-a for a, b, _ in cells)
    covered = sum(b-a for a, b, r in cells if output(r))
    valid_time = sum(b-a for a, b, r in cells if valid(r))
    outages = _episodes(cells, lambda r: not valid(r))
    events = [dict(kind="outage", start_s=a, end_s=b, duration_s=b-a) for a, b in outages]
    metrics.update(duration_denominator_s=duration, observed_duration_s=sum(b-a for a, b, r in cells if r is not None),
                   covered_duration_s=covered, uncovered_duration_s=duration-covered,
                   valid_duration_s=valid_time, availability=valid_time/duration if duration else None,
                   first_valid_s=next((a for a, _, r in cells if valid(r)), None),
                   outage_count=len(outages), longest_outage_s=max((b-a for a, b in outages), default=0.) if duration else None)
    threshold = evaluation.get("false_valid_error_m")
    if threshold is not None:
        def false_valid(row):
            if row is None:
                return False
            _, _, e, flag, _ = _values(row, run.method, ref)
            return e is not None and flag and np.linalg.norm(e) > threshold
        false_events = _episodes(cells, false_valid)
        metrics.update(false_valid_events=len(false_events), false_valid_duration_s=sum(b-a for a, b in false_events),
                       false_valid_scored_samples=sum(r is not None and _values(r, run.method, ref)[2] is not None
                                                     and _values(r, run.method, ref)[3] for _, _, r in cells))
        events.extend(dict(kind="false_valid", start_s=a, end_s=b, duration_s=b-a) for a, b in false_events)
    threshold = evaluation.get("max_outage_s")
    if threshold is not None:
        metrics["outage_limit_exceedances"] = sum(b-a > threshold for a, b in outages)
    return metrics, events


def _recoveries(rows, run):
    """Start at first accepted camera after loss, including confirmation frames."""
    if run.method == "N1":
        return []
    events, seen_valid, lost, accepted, previous_end = [], False, None, None, None
    for row in rows:
        t = row["timestamp"]
        if previous_end is not None and t > previous_end + 1e-9 and lost is None:
            lost, accepted = previous_end, None
        _, output, _, valid, _ = _values(row, run.method, "cg_shaped")
        valid = output is not None and valid
        if not valid and lost is None:
            lost = t
        if lost is not None and accepted is None and row.get("camera_sample") is True and row["diagnostics"].get("vision_accepted") is True:
            accepted = t
        if valid:
            if lost is not None:
                events.append(dict(kind="recovery" if seen_valid else "initialization",
                                   start_s=lost, first_qualified_vision_s=accepted, restored_s=t,
                                   recovery_time_s=t-accepted if accepted is not None else None, censored=False,
                                   stage=row.get("stage", "unknown")))
            seen_valid, lost, accepted = True, None, None
        previous_end = t + row["dt"]
    if lost is not None:
        events.append(dict(kind="recovery" if seen_valid else "initialization", start_s=lost,
                           first_qualified_vision_s=accepted, restored_s=None, recovery_time_s=None,
                           censored=True, stage=rows[-1].get("stage", "unknown")))
    return events


def _lag(rows, method):
    """Velocity cross-correlation diagnostic on longest uninterrupted uniform run."""
    segments, current, previous = [], [], None
    for row in rows:
        raw = _values(row, method, "cg")[1]
        shaped = _values(row, method, "cg_shaped")[1]
        if raw is None or shaped is None:
            if current:
                segments.append(current)
            current, previous = [], None
            continue
        if previous is not None and (not np.isclose(row["timestamp"]-previous["timestamp"], previous["dt"])
                                     or not np.isclose(row["dt"], previous["dt"])):
            segments.append(current)
            current = []
        current.append((row["timestamp"], raw, shaped))
        previous = row
    if current:
        segments.append(current)
    segment = max(segments, key=len, default=[])
    result = dict(lag_s=None, lag_correlation=None, lag_samples=len(segment))
    if len(segment) < 5:
        return result
    a = np.diff(np.array([s[1] for s in segment]), axis=0)
    b = np.diff(np.array([s[2] for s in segment]), axis=0)
    a, b = a-a.mean(axis=0), b-b.mean(axis=0)
    if np.sum(a*a) <= np.finfo(float).eps or np.sum(b*b) <= np.finfo(float).eps:
        return result
    n = len(a)
    cross = sum(correlate(b[:, i], a[:, i], mode="full", method="fft")[n-1:] for i in range(3))
    candidates = []
    for lag in range(n//2 + 1):
        denominator = np.sqrt(np.sum(a[:n-lag]**2) * np.sum(b[lag:]**2))
        candidates.append(cross[lag]/denominator if denominator else -np.inf)
    best = int(np.argmax(candidates))
    result.update(lag_s=best*(segment[1][0]-segment[0][0]), lag_correlation=float(candidates[best]))
    return result


def _diagnostics(rows, run, recovery):
    camera = [r for r in rows if r.get("camera_sample") is True]
    lidar = [r for r in rows if r.get("lidar_sample") is True]
    known_lidar = any(isinstance(r.get("lidar_sample"), bool) for r in rows)
    result = dict(camera_events=len(camera), lidar_events=len(lidar) if known_lidar else None)
    solved = [r["diagnostics"]["pnp_solved"] for r in camera if isinstance(r["diagnostics"].get("pnp_solved"), bool)]
    result.update(pnp_solve_observations=len(solved), pnp_solved_count=sum(solved) if solved else None,
                  pnp_solve_fraction=float(np.mean(solved)) if solved else None)
    for sensor, samples in (("vision", camera), ("lidar", lidar)):
        known = [r for r in samples if isinstance(r["diagnostics"].get(sensor+"_accepted"), bool)]
        result[sensor+"_acceptance_fraction"] = (sum(r["diagnostics"][sensor+"_accepted"] for r in known)/len(known)) if known else None
        result[sensor+"_acceptance_observations"] = len(known) if sensor == "vision" or known_lidar else None
        # The denominator is explicit gate evaluations, not all sensor packets:
        # geometry failures and dropped packets never reach the innovation gate.
        gated = [r for r in known if _number(r["diagnostics"].get(sensor+"_mahalanobis")) is not None]
        rejected = sum(r["diagnostics"].get(sensor+"_rejection") == "innovation_gate" for r in gated)
        result[sensor+"_innovation_rejection_count"] = rejected if gated else None
        result[sensor+"_innovation_rejection_observations"] = len(gated)
        result[sensor+"_innovation_rejection_fraction"] = rejected/len(gated) if gated else None
        for key in ("reprojection_error", "vision_mahalanobis", "vision_position_correction_m") if sensor == "vision" else (
                "lidar_mahalanobis", "lidar_position_correction_m", "lidar_association_rejection_fraction"):
            result.update(_stats([r["diagnostics"].get(key) for r in samples], key))
        for outcome in (True, False):
            key = sensor+"_mahalanobis"
            result.update(_stats([r["diagnostics"].get(key) for r in samples
                                  if r["diagnostics"].get(sensor+"_accepted") is outcome],
                                 key + ("_accepted" if outcome else "_rejected")))
    point_pairs = [(r["quality"].get("lidar_points"), r["diagnostics"].get("lidar_associated_points")) for r in lidar]
    point_pairs = [(n, a) for n, a in point_pairs if _number(n) is not None and _number(a) is not None and 0 <= a <= n]
    point_total = sum(n for n, _ in point_pairs)
    result.update(lidar_association_input_points=point_total if point_pairs else None,
                  lidar_association_rejected_points=sum(n-a for n, a in point_pairs) if point_pairs else None,
                  lidar_association_rejection_fraction_pooled=sum(n-a for n, a in point_pairs)/point_total if point_total else None)
    for key in ("vision_roi_attempted", "vision_reacquired", "vision_recovery_pending"):
        known = [r["diagnostics"][key] for r in camera if isinstance(r["diagnostics"].get(key), bool)]
        result[key+"_count"] = sum(known) if known else None
        result[key+"_fraction"] = float(np.mean(known)) if known else None
    for key in QUALITY:
        samples = lidar if key in ("lidar_points", "plane_inlier_ratio", "plane_rms") else camera
        values = [int(r["quality"][key]) if isinstance(r["quality"].get(key), bool) else r["quality"].get(key) for r in samples]
        result.update(_stats(values, key))
    result.update(_stats([r["diagnostics"].get("cg_tracking_error_m") for r in rows], "cg_tracking_error_m"))
    limited = [r["diagnostics"]["cg_correction_limited"] for r in rows
               if isinstance(r["diagnostics"].get("cg_correction_limited"), bool)]
    result["cg_correction_limited_count"] = sum(limited) if limited else None
    residual = [np.linalg.norm(a-b) for r in rows
                if (a := _values(r, run.method, "cg")[1]) is not None
                and (b := _values(r, run.method, "cg_shaped")[1]) is not None]
    result.update(_stats(residual, "shaping_residual_m"))
    result.update(_lag(rows, run.method))
    recoveries = [e for e in recovery if e["kind"] == "recovery"]
    result.update(recovery_events=len(recoveries), recovery_censored=sum(e["censored"] for e in recoveries),
                  **_stats([e["recovery_time_s"] for e in recoveries], "recovery_time_s"))
    return result


def _timing_metrics(rows):
    records = []
    for component in sorted(set(TIMINGS) | {key for r in rows for key in r["timing"]}):
        calls = []
        for row in rows:
            if component in ("inference_s", "geometry_s", "camera_s") and row.get("camera_sample") is not True:
                continue
            if component == "lidar_s" and row.get("lidar_sample") is False:
                continue
            value = _number(row["timing"].get(component))
            if value is not None and value >= 0:
                calls.append((row, value))
        for stage in STAGES:
            for phase, values in (("first_recorded_call", calls[:1]), ("subsequent_calls", calls[1:])):
                samples = [v for r, v in values if stage == "all" or r.get("stage") == stage]
                records.append(dict(reference="timing:"+component, stage=stage, sample_set=phase,
                                    **_stats(samples, "seconds")))
    return records


def _bootstrap(records, samples, seed):
    """Resample paired differences/flight metrics, never individual frames.

    Reused source recordings are first collapsed into one independent unit.
    A source assigned conflicting strata cannot support stratified inference.
    Singleton strata remain fixed and are explicitly reported.
    """
    units = defaultdict(list)
    for r in records:
        if _number(r["value"]) is not None:
            units[r["independent_unit"]].append(r)
    grouped, conflicts = defaultdict(list), False
    for unit in sorted(units):
        entries = units[unit]
        strata = {(r["weather"], r["motion"], r["gimbal_mode"]) for r in entries}
        conflicts |= len(strata) != 1
        # Imported aliases do not gain weight by appearing under more ids.
        grouped[sorted(strata)[0]].append((float(np.mean([r["value"] for r in entries])),
                                          float(np.mean([r["weight"] for r in entries]))))
    arrays = [np.asarray(grouped[k]) for k in sorted(grouped)]
    n = sum(len(a) for a in arrays)
    if not n:
        return []
    values = np.concatenate(arrays)
    estimates = dict(flight_equal=float(values[:, 0].mean()),
                     scenario_equal=float(np.mean([a[:, 0].mean() for a in arrays])),
                     duration_weighted=float(np.average(values[:, 0], weights=values[:, 1]))
                     if values[:, 1].sum() > 0 else None)
    eligible = all(r["cohort"] in ("complete", "complete/complete") for rs in units.values() for r in rs)
    can_resample = eligible and not conflicts and any(len(a) > 1 for a in arrays)
    intervals = {}
    if can_resample:
        rng = np.random.default_rng(seed)
        totals = np.zeros(samples)
        weighted = np.zeros(samples)
        weights = np.zeros(samples)
        means = np.zeros(samples)
        for a in arrays:
            # Batch the replicate axis to bound memory for large experiments.
            for start in range(0, samples, 128):
                stop = min(start+128, samples)
                draw = a[rng.integers(len(a), size=(stop-start, len(a)))]
                totals[start:stop] += draw[:, :, 0].sum(axis=1)
                means[start:stop] += draw[:, :, 0].mean(axis=1)
                weighted[start:stop] += (draw[:, :, 0]*draw[:, :, 1]).sum(axis=1)
                weights[start:stop] += draw[:, :, 1].sum(axis=1)
        distributions = dict(flight_equal=totals/n, scenario_equal=means/len(arrays))
        if np.all(weights > 0):
            distributions["duration_weighted"] = weighted/weights
        intervals = {key: np.percentile(a, [2.5, 97.5]).tolist() for key, a in distributions.items()}
    reason = ("percentile_flight_bootstrap" if can_resample else
              "incomplete_or_excluded" if not eligible else
              "shared_source_crosses_strata" if conflicts else "insufficient_independent_replicates")
    return [dict(estimand=key, estimate=value, ci_low=intervals.get(key, [None, None])[0],
                 ci_high=intervals.get(key, [None, None])[1], n_flights=sum(len(v) for v in units.values()),
                 n_units=n, n_strata=len(arrays), singleton_strata=sum(len(a) == 1 for a in arrays),
                 interval_status=reason) for key, value in estimates.items()]


def _long_metrics(records):
    dimensions = set(DIMENSIONS) | {"planned_duration_s"}
    for record in records:
        meta = {key: record.get(key) for key in DIMENSIONS}
        for metric, value in record.items():
            if metric not in dimensions:
                yield dict(meta, metric=metric, value=value,
                           weight=record.get("planned_duration_s") or 0.,
                           sample_count=record.get("samples", record.get("camera_samples", record.get("seconds_n"))))


def _table_metric(metric):
    """Counts describing evidence stay in raw files, not hundreds of count CIs."""
    excluded = {"samples", "camera_events", "lidar_events", "recovery_events", "recovery_censored",
                "duration_denominator_s", "observed_duration_s", "covered_duration_s", "valid_duration_s",
                "lidar_association_input_points", "lidar_association_rejected_points"}
    return (metric not in excluded and not metric.endswith(("_n", "_samples", "_observations"))
            and (not metric.endswith("_count") or metric == "outage_count"))


def _aggregate(records, samples, seed, views):
    groups = defaultdict(list)
    for row in records:
        if row["metric"] not in REPORT_METRICS or _number(row["value"]) is None:
            continue
        compact = {key: row[key] for key in ("value", "independent_unit", "weather", "motion", "gimbal_mode", "weight", "cohort")}
        compact["sample_count"] = row.get("sample_count")
        for view, factors in views.items():
            group = dict(view=view, suite=row["suite"], condition=row["condition"], method=row["method"],
                         cohort=row["cohort"], reference=row["reference"], stage=row["stage"],
                         sample_set=row["sample_set"], common_methods=row.get("common_methods", ""),
                         metric=row["metric"])
            for key in ("weather", "motion", "gimbal_mode"):
                group[key] = row[key] if key in factors else "all"
            for key in ("comparison", "comparator"):
                if key in row:
                    group[key] = row[key]
            groups[tuple(group.items())].append(compact)
    for key, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        digest = hashlib.sha256(repr(key).encode()).digest()
        group_seed = seed + int.from_bytes(digest[:8], "little")
        results = _bootstrap(entries, samples, group_seed)
        if not results:
            results = [dict(estimand=w, estimate=None, ci_low=None, ci_high=None,
                            n_flights=0, n_units=0, n_strata=0, singleton_strata=0, interval_status="no_finite_evidence")
                       for w in ("flight_equal", "scenario_equal", "duration_weighted")]
        for r in results:
            counts = defaultdict(list)
            for entry in entries:
                if entry["sample_count"] is not None:
                    counts[entry["independent_unit"]].append(entry["sample_count"])
            yield dict(key, **r, n_observations=sum(max(v) for v in counts.values()) if counts else None)


def _table_records(records):
    for record in records:
        if record["sample_set"] == "timeline" and not record.get("duration_denominator_s"):
            continue
        yield record


def _wide_tables(root, metrics, pairs, samples, seed):
    """Small thesis review tables; full flight evidence stays in raw artifacts."""
    views = dict(overall=(), weather=("weather",), motion=("motion",), gimbal=("gimbal_mode",))
    group_keys = ("view", "suite", "condition", "method", "cohort", "reference", "stage", "sample_set",
                  "common_methods", "weather", "motion", "gimbal_mode", "comparison", "comparator", "estimand")
    tables = {13: {}, 14: {}}
    stats_fields = list(group_keys) + ["metric", "estimate", "ci_low", "ci_high", "n_flights", "n_units", "n_strata",
                                      "singleton_strata", "interval_status", "n_observations"]
    def collect(number, result):
        key = tuple(result.get(k, "") for k in group_keys)
        wide = tables[number].setdefault(key, dict(zip(group_keys, key)))
        statuses = set(wide.get("interval_status", "").split(";")) - {""}
        statuses.add(result["interval_status"])
        wide["interval_status"] = ";".join(sorted(statuses))
        for field in ("estimate", "ci_low", "ci_high", "n_flights", "n_units", "n_observations"):
            wide[result["metric"]+"_"+field] = result.get(field)
    with (root/"statistics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fields)
        writer.writeheader()
        for source in (_long_metrics(_table_records(metrics)), iter(pairs)):
            for result in _aggregate(source, samples, seed, views):
                # Overall estimates retain all three weighting policies, without
                # multiplying the human-review tables by redundant estimands.
                if result["view"] == "overall":
                    writer.writerow(result)
                if result["estimand"] != "scenario_equal":
                    continue
                if result["reference"].startswith("timing:"):
                    continue  # Resource results are in statistics_summary.csv.
                paired = "comparison" in result
                if (result["suite"] == "main" and result["method"] in ("N0", "N1", "N2", "N3")
                        and result["view"] == "overall" and not paired
                        and result["sample_set"] in ("finite", "declared_valid", "common_camera", "timeline", "events")
                        and (result["reference"] in REFERENCES or result["metric"] == "recovery_time_s_mean")):
                    collect(13, result)
                if (paired and result["view"] == "overall") or (not paired and (result["view"] != "overall" or result["method"].startswith("A") or result["suite"] != "main")):
                    collect(14, result)
    for number, rows in tables.items():
        _csv(root/f"table_5_{number}.csv", list(rows.values()))


def _paired(records, runs, warnings):
    lookup = {(r["case_id"], r["method"], r["reference"], r["stage"], r["sample_set"]): r for r in records}
    baseline_cases = defaultdict(set)
    for run in runs:
        m = run.meta
        if m["suite"] == "main" and m["condition"] == "baseline" and run.method == "N3":
            baseline_cases[(m["pair_id"], m["weather"], m["motion"], m["seed"], m["gimbal_mode"])].add(m["case_id"])
    output = []
    for record in records:
        if record["sample_set"] == "timeline" and not record.get("duration_denominator_s"):
            continue
        pairs = []
        if record["method"] != "N3":
            pairs.append((record["case_id"], "method", "N3"))
        elif record["suite"] == "stress" or (record["suite"] == "gimbal" and record["gimbal_mode"] == "fixed"):
            mode = "tracking" if record["suite"] == "gimbal" else record["gimbal_mode"]
            key = (record["pair_id"], record["weather"], record["motion"], record["seed"], mode)
            candidates = baseline_cases.get(key, set())
            if len(candidates) == 1:
                pairs.append((next(iter(candidates)), record["suite"], "N3_main_baseline_"+mode))
            else:
                warnings.add(f"{record['case_id']}: no unique main N3 baseline for {record['suite']} pair")
        for cid, comparison, comparator in pairs:
            # Different physical sensor recordings do not share the same
            # finite-camera intersection. Compare their per-flight metrics.
            if cid != record["case_id"] and record["sample_set"].startswith("common_camera"):
                continue
            base = lookup.get((cid, "N3", record["reference"], record["stage"], record["sample_set"]))
            if base is None:
                continue
            meta = {key: record.get(key) for key in DIMENSIONS}
            meta.update(cohort=record["cohort"]+"/"+base["cohort"], comparison=comparison, comparator=comparator,
                        baseline_case_id=cid)
            for metric in sorted(record.keys() - set(DIMENSIONS) - {"planned_duration_s"}):
                if not _table_metric(metric):
                    continue
                a, b = _number(record[metric]), _number(base.get(metric))
                output.append(dict(meta, metric=metric, value=a-b if a is not None and b is not None else None,
                                   subject_value=a, baseline_value=b, weight=record.get("planned_duration_s") or 0.))
    return output


def _sample_fields():
    return list(META) + ["timestamp", "dt", "stage", "camera_sample", "lidar_sample", "estimate_available",
                         "navigation_valid", "cg_signal_valid", "attitude_error_rad"] + [
        f"{ref}_{kind}_{axis}" for ref in REFERENCES for kind in ("truth", "estimate", "error", "band95") for axis in "ned"
    ] + [f"{ref}_J_m" for ref in REFERENCES] + ["shaping_residual_m"] + [
        "diagnostics."+key for key in DIAGNOSTICS
    ] + ["quality."+key for key in QUALITY] + ["timing."+key for key in TIMINGS]


def _sample_records(rows, run):
    previous = dict.fromkeys(REFERENCES)
    for row in rows:
        flat = dict(run.meta, **{k: row.get(k) for k in ("timestamp", "dt", "stage", "camera_sample", "lidar_sample",
                                                       "estimate_available", "navigation_valid", "cg_signal_valid", "attitude_error_rad")})
        for ref in REFERENCES:
            truth, estimate, error, _, covariance = _values(row, run.method, ref)
            band = 1.96*np.sqrt(np.diag(covariance)) if covariance is not None and estimate is not None else None
            for kind, value in (("truth", truth), ("estimate", estimate), ("error", error), ("band95", band)):
                for i, axis in enumerate("ned"):
                    flat[f"{ref}_{kind}_{axis}"] = float(value[i]) if value is not None else None
            flat[ref+"_J_m"] = float(np.linalg.norm(error-previous[ref])) if error is not None and previous[ref] is not None else None
            if _eligible(row, run.method, ref):
                previous[ref] = error
        a, b = _values(row, run.method, "cg")[1], _values(row, run.method, "cg_shaped")[1]
        flat["shaping_residual_m"] = float(np.linalg.norm(a-b)) if a is not None and b is not None else None
        for key, fields in (("diagnostics", DIAGNOSTICS), ("quality", QUALITY), ("timing", TIMINGS)):
            flat.update({key+"."+field: row[key].get(field) for field in fields})
        yield _clean(flat)


def _select_representative(runs):
    by_case = defaultdict(list)
    for run in runs:
        by_case[run.case["id"]].append(run)
    # Selection uses metadata only, never observed error or how good a curve is.
    candidates = [cid for cid, group in by_case.items() if any(r.method == "N3" for r in group)
                  and all(r.cohort in ("complete", "excluded_complete") and (r.path/"steps.jsonl").exists() for r in group)]
    candidates.sort(key=lambda cid: (any(r.cohort != "complete" for r in by_case[cid]),
                                     by_case[cid][0].case.get("suite", "main") != "main", cid))
    return candidates[0] if candidates else None


def _zoom_window(samples):
    n3 = [r for r in samples if r["method"] == "N3"]
    candidates = [r for r in n3 if r["camera_sample"] is True and
                  (_number(r.get("diagnostics.vision_position_correction_m")) or 0) > 0]
    if candidates:
        event = min(candidates, key=lambda r: (-r["diagnostics.vision_position_correction_m"], r["timestamp"]))
        reason = "largest_camera_correction"
    else:
        candidates = [r for r in n3 if r["camera_sample"] is True and r.get("diagnostics.vision_reacquired") is True]
        if not candidates:
            return None
        event, reason = candidates[0], "first_vision_reacquisition"
    start, stop = min(r["timestamp"] for r in n3), max(r["timestamp"]+r["dt"] for r in n3)
    # A display window, not a performance/evaluation threshold.
    radius = max((stop-start)*.05, event["dt"]*2)
    return dict(start_s=max(start, event["timestamp"]-radius), end_s=min(stop, event["timestamp"]+radius),
                event_s=event["timestamp"], reason=reason)


def _plots(root, representative, flight_metrics, zoom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def numbers(rows, key):
        return np.array([np.nan if _number(r.get(key)) is None else r[key] for r in rows], dtype=float)

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(root/(name+".png"), dpi=150)
        plt.close(fig)

    def annotations(axes, rows, flag="navigation_valid"):
        if not rows:
            return
        previous = None
        for row in rows:
            t = row["timestamp"]
            if row.get("stage") != previous:
                for axis in axes:
                    axis.axvline(t, color=".5", lw=.6, ls=":")
                axes[0].text(t, .98, str(row.get("stage", "unknown")), transform=axes[0].get_xaxis_transform(),
                             va="top", fontsize=7)
                previous = row.get("stage")
        # Draw merged invalid intervals, not tens of thousands of patches.
        intervals = _episodes([(r["timestamp"], r["timestamp"]+r["dt"], r) for r in rows],
                              lambda r: r.get(flag) is not True)
        for a, b in intervals:
            for axis in axes:
                axis.axvspan(a, b, color="tab:red", alpha=.07)
        stop = max(r["timestamp"]+r["dt"] for r in rows)
        for axis in axes:
            axis.axvline(stop, color="k", lw=.6, ls="--")

    samples = representative
    n3 = [r for r in samples if r["method"] == "N3"]
    banner = "SMOKE/EXCLUDED; not formal — " if n3 and n3[0]["cohort"] != "complete" else ""
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    if not n3:
        fig.text(.5, .5, "No complete representative case available", ha="center")
    for i, axis_name in enumerate("ned"):
        left, right = axes[i]
        if n3:
            left.plot(numbers(n3, "timestamp"), numbers(n3, "gear_truth_"+axis_name), color="k", label="truth")
        for method in sorted({r["method"] for r in samples}):
            rows = [r for r in samples if r["method"] == method and (method != "N1" or r["camera_sample"] is True)]
            t = numbers(rows, "timestamp")
            style = dict(marker=".", linestyle="none") if method == "N1" else {}
            line, = left.plot(t, numbers(rows, "gear_estimate_"+axis_name), label=method, lw=.9, **style)
            right.plot(t, numbers(rows, "gear_error_"+axis_name), label=method, color=line.get_color(), lw=.9, **style)
            band = numbers(rows, "gear_band95_"+axis_name)
            if method == "N1":
                right.errorbar(t, np.zeros(len(t)), yerr=band, fmt="none", alpha=.15, color=line.get_color())
            else:
                right.fill_between(t, -band, band, color=line.get_color(), alpha=.08)
        left.set_ylabel(axis_name.upper()+" position (m)")
        right.set_ylabel(axis_name.upper()+" error (m)")
        right.axhline(0, color=".5", lw=.6)
    annotations(list(axes.flat), n3)
    if n3:
        axes[0, 0].legend(ncol=4, fontsize=7)
    axes[2, 0].set_xlabel("Elapsed time (s)")
    axes[2, 1].set_xlabel("Elapsed time (s)")
    fig.suptitle(banner+"5-8 Gear NED; error bounds ±1.96 sqrt(Pii); shading: N3 invalid")
    finish(fig, "figure_5_8")

    fig, axes = plt.subplots(5, 2, figsize=(14, 15))
    t = numbers(n3, "timestamp")
    for i, axis_name in enumerate("ned"):
        for ref, label in (("cg", "raw"), ("cg_shaped", "shaped")):
            axes[i, 0].plot(t, numbers(n3, ref+"_estimate_"+axis_name), label=label, lw=1)
            axes[i, 1].plot(t, numbers(n3, ref+"_error_"+axis_name), label=label, lw=1)
        axes[i, 0].plot(t, numbers(n3, "cg_truth_"+axis_name), color="k", label="truth", lw=.8)
        band = numbers(n3, "cg_band95_"+axis_name)
        axes[i, 1].fill_between(t, -band, band, alpha=.15, label="raw covariance only")
        axes[i, 0].set_ylabel(axis_name.upper()+" CG (m)")
        axes[i, 1].set_ylabel(axis_name.upper()+" error (m)")
    for ref in ("cg", "cg_shaped"):
        axes[3, 0].plot(t, numbers(n3, ref+"_J_m"), label=ref+" error increment", lw=.8)
    axes[3, 0].plot(t, numbers(n3, "shaping_residual_m"), label="raw-shaped residual", lw=.8)
    for flag in ("navigation_valid", "cg_signal_valid"):
        axes[3, 1].step(t, [float(r[flag]) if isinstance(r.get(flag), bool) else np.nan for r in n3], where="post", label=flag)
    axes[3, 0].set_ylabel("Increment / residual (m)")
    axes[3, 1].set_ylabel("Declared validity")
    axes[4, 0].plot(t, [r.get("diagnostics.vision_position_correction_m") if r["camera_sample"] is True else np.nan
                       for r in n3], marker=".", linestyle="none", label="vision correction")
    axes[4, 0].plot(t, numbers(n3, "shaping_residual_m"), label="shaping residual")
    for ref in ("cg", "cg_shaped"):
        axes[4, 1].plot(t, numbers(n3, ref+"_J_m"), label=ref+" error increment")
    for axis in axes.flat:
        axis.set_xlabel("Elapsed time (s)")
        if n3:
            axis.legend(fontsize=7)
    annotations(list(axes.flat), n3, "cg_signal_valid")
    for axis in axes[4]:
        if zoom:
            axis.set_xlim(zoom["start_s"], zoom["end_s"])
            axis.axvline(zoom["event_s"], color="tab:red", ls="--", lw=.8)
            axis.set_title("Event zoom: "+zoom["reason"], fontsize=9)
        else:
            axis.set_title("No qualified correction/recovery zoom event", fontsize=9)
    if not n3:
        fig.text(.5, .5, "No complete representative case available", ha="center")
    fig.suptitle(banner+"5-9 N3 CG raw / shaped; shading: CG invalid; no shaped covariance")
    finish(fig, "figure_5_9")

    # Every plotted point is a flight statistic. No frame independence implied.
    data = []
    for record in flight_metrics:
        if record["stage"] == "all" and ((record["reference"] == "gear" and record["sample_set"] == "finite")
                                        or record["reference"] == "diagnostics"):
            data.append(record)
    _csv(root/"figure_5_10.csv", data)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    complete = [r for r in data if r["cohort"] == "complete" and r["suite"] == "main"]
    smoke_distribution = not complete
    if smoke_distribution:
        complete = [r for r in data if r["cohort"] == "excluded_complete" and r["suite"] == "main"]
    for method in sorted({r["method"] for r in complete}):
        units = defaultdict(list)
        for r in complete:
            if r["method"] == method and _number(r.get("rmse_3d_m")) is not None:
                units[r["independent_unit"]].append(r["rmse_3d_m"])
        values = sorted(float(np.mean(v)) for v in units.values())
        if values:
            axes[0, 0].step(values, np.arange(1, len(values)+1)/len(values), where="post", marker=".", label=method)
    axes[0, 0].set(xlabel="Per-flight gear RMSE (m)", ylabel="ECDF (SMOKE/EXCLUDED)" if smoke_distribution else "ECDF (complete main flights)")
    axes[0, 0].set_ylim(-.02, 1.02)
    panels = ((axes[0, 1], "target_scale_px_mean", "edge_margin_px_mean", "Mean scale (px)", "Mean edge margin (px)"),
              (axes[0, 2], "gimbal_saturated_mean", "gimbal_reference_error_rad_mean", "Saturation fraction", "Reference error (rad)"),
              (axes[1, 2], "truncation_fraction_mean", "vision_acceptance_fraction", "Truncation fraction", "Vision acceptance fraction"))
    for axis, x, y, xlabel, ylabel in panels:
        for cohort in sorted({r["cohort"] for r in data}):
            rows = [r for r in data if r["cohort"] == cohort and _number(r.get(x)) is not None and _number(r.get(y)) is not None]
            if rows:
                axis.scatter([r[x] for r in rows], [r[y] for r in rows], s=12, alpha=.6, label=cohort)
        axis.set(xlabel=xlabel, ylabel=ylabel)
    for sensor, axis in (("vision", axes[1, 0]), ("lidar", axes[1, 1])):
        for cohort in sorted({r["cohort"] for r in data}):
            for outcome in ("accepted", "rejected"):
                key = sensor+"_mahalanobis_"+outcome+"_mean"
                values = sorted(r[key] for r in data if r["cohort"] == cohort and _number(r.get(key)) is not None)
                if values:
                    axis.step(values, np.arange(1, len(values)+1)/len(values), where="post", marker=".", label=cohort+" "+outcome)
        axis.set(xlabel=sensor+" flight-mean innovation d²", ylabel="Empirical CDF")
        axis.set_ylim(-.02, 1.02)
    for axis in axes.flat:
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, "Insufficient recorded evidence", transform=axis.transAxes, ha="center", fontsize=8)
    fig.suptitle(("SMOKE/EXCLUDED; not formal — " if smoke_distribution and complete else "")+
                 "5-10 Flight distributions, FOV and innovations; cohorts labelled")
    finish(fig, "figure_5_10")

    innovations = []
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for sensor, axis in zip(("vision", "lidar"), axes):
        sample_flag = "camera_sample" if sensor == "vision" else "lidar_sample"
        for method in sorted({r["method"] for r in samples}):
            rows = [r for r in samples if r["method"] == method and r.get(sample_flag) is True]
            for row in rows:
                innovations.append(dict(case_id=row["case_id"], method=method, sensor=sensor,
                                        timestamp=row["timestamp"], stage=row["stage"],
                                        accepted=row.get("diagnostics."+sensor+"_accepted"),
                                        mahalanobis=row.get("diagnostics."+sensor+"_mahalanobis"),
                                        threshold=row.get("diagnostics."+sensor+"_mahalanobis_threshold"),
                                        rejection=row.get("diagnostics."+sensor+"_rejection")))
            color = None
            for outcome, marker in ((True, "."), (False, "x")):
                selected = [r for r in rows if r.get("diagnostics."+sensor+"_accepted") is outcome
                            and _number(r.get("diagnostics."+sensor+"_mahalanobis")) is not None]
                if selected:
                    line, = axis.plot(numbers(selected, "timestamp"), numbers(selected, "diagnostics."+sensor+"_mahalanobis"),
                                      linestyle="none", marker=marker, color=color,
                                      label=method+(" accepted" if outcome else " rejected"), markersize=4)
                    color = line.get_color()
            thresholds = numbers(rows, "diagnostics."+sensor+"_mahalanobis_threshold")
            if np.isfinite(thresholds).any():
                axis.step(numbers(rows, "timestamp"), thresholds, where="post", ls=":", color=color, label=method+" logged gate")
        axis.set_ylabel(sensor+" innovation d²")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7, ncol=3)
        else:
            axis.text(.5, .5, "No recorded innovations / gates", transform=axis.transAxes, ha="center")
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle(banner+"5-10 Innovation detail: representative complete case, actual logged gates")
    _csv(root/"figure_5_10_innovations.csv", innovations,
         ["case_id", "method", "sensor", "timestamp", "stage", "accepted", "mahalanobis", "threshold", "rejection"])
    finish(fig, "figure_5_10_innovations")


def _evaluation(manifest, bootstrap_samples):
    evaluation = dict(false_valid_error_m=1., max_outage_s=.5,
                      false_valid_sensitivity_m=[.5, 1., 2.], outage_sensitivity_s=[.2, .5, 1.],
                      bootstrap_seed=0)
    evaluation.update(manifest.get("evaluation", {}))
    for name in ("false_valid_error_m", "max_outage_s"):
        value = evaluation.get(name)
        if value is not None and (_number(value) is None or value <= 0):
            raise ValueError(f"evaluation.{name} must be null or positive finite")
    for name in ("false_valid_sensitivity_m", "outage_sensitivity_s"):
        values = evaluation[name]
        if not isinstance(values, list) or any(_number(v) is None or v <= 0 for v in values):
            raise ValueError(f"evaluation.{name} must be a list of positive finite values")
        evaluation[name] = sorted(set(values))
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    seed = evaluation["bootstrap_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    evaluation["manifest_bootstrap_samples"] = evaluation.get("bootstrap_samples")
    evaluation["bootstrap_samples"] = (evaluation.get("bootstrap_samples", bootstrap_samples)
                                       if bootstrap_samples == 2000 else bootstrap_samples)
    effective = evaluation["bootstrap_samples"]
    if isinstance(effective, bool) or not isinstance(effective, int) or effective < 1:
        raise ValueError("manifest bootstrap_samples must be a positive integer")
    return evaluation


def _report(root, summary):
    counts = Counter(r["cohort"] for r in summary["runs"])
    lines = ["# 导航测试分析 — 论文 5.2.3", "",
             *_numeric_findings(root, summary), "",
             "This is navigation evidence only. Evaluation limits are not health gates, touchdown criteria or certification.", "",
             f"Manifest cases: {summary['case_count']}. Run cohorts: {dict(counts)}. "
             f"Failure records (case and run records may overlap): {len(summary['failures'])}.", "",
             "## Sensor storage", "",
             f"Intentionally retired records: {summary.get('storage', {}).get('pruned_records', 0)}. "
             "All numerical method results remain available. storage_summary.json reports logical sensor bytes only. "
             "Diagnostic clips are selected examples, not an unbiased sample for statistics; all numerical flights are scored. "
             "Pruned observations cannot support full sensor replay without regenerating the entire dependent group. "
             "Analysis checks retired-result integrity without needing image/point-cloud payloads.", "",
             "## Evaluation and sample definitions", "",
             f"Frozen evaluation configuration: `{json.dumps(summary['evaluation'], ensure_ascii=False)}`.", "",
             "The primary false-valid criterion is strict 3D error > false_valid_error_m, separately for gear, CG and shaped CG. "
             "The outage limit is exceeded only when duration > max_outage_s. Null primary limits remain unscored. "
             "Sensitivity results are in sensitivity.csv; they do not tune the navigation health gates.", "",
             "finite uses every produced finite estimate with matching finite truth; declared_valid additionally requires "
             "navigation_valid for raw gear/CG or cg_signal_valid for shaped CG. common_camera intersects exact finite "
             "camera timestamps across expected main methods N0–N3. common_camera_all_methods separately includes ablations, "
             "and common_camera_pair_A* uses an ablation and N3 on identical samples. Membership is stored in common_methods. "
             "It is not an intersection of valid samples. N1 has camera samples only, and no shaped output. "
             "No hold, interpolation, zero substitution or time-shifted accuracy is used.", "",
             "Sample accuracy is unweighted within flights. Availability sums logged dt cells over the entire planned duration, "
             "clipped to the next timestamp, completed duration and planned end. Gaps and uncompleted tails are unavailable. "
             "For N1 this measures only explicitly logged camera-cell support; it does not claim continuous PnP output. "
             "Stage denominators contain only recorded stage cells: unknown gaps/tails cannot be assigned a stage. "
             "camera_coverage counts finite scored outputs over camera samples, including unsolved samples in the denominator. "
             "covered_duration requires finite output, not truth; valid duration additionally requires the declared flag.", "",
             "J is the norm of consecutive error increments, not position increments. Missing outputs break its chain. "
             "Correction diagnostics remain separate from raw-minus-shaped residual and logged CG tracking residual. "
             "The lag diagnostic uses velocity cross-correlation on the longest contiguous uniform segment, searches "
             "nonnegative lags up to half its length, and requires changing velocity. It is exploratory, can be ambiguous "
             "for periodic motion, and never adjusts accuracy timestamps.", "",
             "Recovery starts at the first accepted camera frame during a CG outage, including accepted confirmation frames "
             "with vision_recovery_pending=true, and ends at finite shaped CG with cg_signal_valid=true. Initialization is "
             "separate. No accepted frame or no restoration produces null latency; censored events remain in events.csv. "
             "False-valid counts are contiguous episodes; observed sample counts and durations are also retained.", "",
             "NEES uses only finite symmetric positive-definite 3x3 covariance at the same reference in NED. "
             f"The 95% ellipsoid empirical coverage compares NEES to chi-square(3)={CHI2_95:.8g}. "
             "There is no shaped NEES or shaped uncertainty band. Plot bands are ±1.96 sqrt(Pii), marginal bounds about "
             "zero error, not confidence intervals for a mean. Rejected and declared-invalid finite estimates remain in raw statistics.", "",
             "## Independent units and paired comparisons", "",
             "Compact wide table 5-13 contains main N0–N3, overall scenario-equal estimates; table 5-14 contains "
             "weather/motion/gimbal views, ablations and paired differences. Each metric has its own estimate, CI and sample counts. "
             "statistics_summary.csv retains the three overall weighting policies. Empty groups are omitted from review tables; "
             "null evidence and every diagnostic/count remain in flight_metrics.csv and statistical_raw.csv. "
             "All tables retain suite, condition, stage, reference, sample set and complete/incomplete/excluded cohort. "
             "A complete run must report complete status and completed duration covering planned duration. Short complete-labelled "
             "runs are incomplete. Current record-phase failures invalidate all cases sharing that record; replay-phase "
             "failures invalidate only the named case/method. Their old rows remain descriptive incomplete evidence even "
             "when run.json says complete. If record_checks/<record_id>.json exists, it must report passed and stored "
             "results must match its input_record_sha256. An unfinished/failed check invalidates the shared record's "
             "results; a mismatched binding invalidates only that stale case/method. "
             "Outputs without markers retain legacy behavior. "
             "Smoke/excluded data never enter formal complete inference.", "",
             "Bootstrap resamples flight-level scalars within weather/motion/gimbal strata. Method differences are paired to N3 "
             "on the same case. Fixed gimbal N3 is paired with tracking main N3 using pair_id/weather/motion/seed; stress "
             "N3 is paired with main baseline at the same pair and gimbal mode. Cross-recording comparisons do not pretend "
             "to share a common finite-camera set. Paired values and signs (subject minus comparator) are in paired_differences.csv.", "",
             "Source identifiers collapse reused recordings into a single independent unit. Otherwise pair_id identifies the unit. "
             "A recording assigned conflicting strata disables its confidence interval. Aliases cannot create replication. "
             "Single-unit strata are fixed in bootstrap draws, their count is reported, and entirely singleton designs have no CI. "
             "Incomplete and excluded cohorts have descriptive point estimates only. Flight-equal, scenario-equal and planned-duration "
             "weighted means are distinct estimands. CIs are percentile 95% intervals; there are no p-values, multiplicity-adjusted "
             "claims or assertions of practical significance. Small batches cannot establish low failure rates.", "",
             "## Figures and resource evidence", "",
             f"Representative case: `{summary['representative_case']}`. Selection is lexicographic among cases whose expected "
             "methods are all complete, preferring main; it never ranks accuracy. If no formal complete case exists, "
             "excluded complete runs may be illustrated with a prominent SMOKE/EXCLUDED label and no formal inference. "
             "All figure source rows are saved as CSV.", "",
             f"Figure 5-9 zoom: `{json.dumps(summary['zoom_window'])}`. It selects the largest recorded camera correction "
             "(earliest tie), falling back to first reacquisition. The window radius is 5% of flight length or two dt cells. "
             "These are display choices, not evaluation thresholds. Figure 5-10 uses flight statistics; its distribution "
             "panel contains complete main-suite runs only, with an explicitly excluded smoke fallback if necessary. "
             "Source aliases are collapsed for its ECDF. FOV/innovation panels label incomplete/excluded runs.", "",
             "Timing is reported in seconds as mean/p95/max, separately for the first recorded call per component and "
             "subsequent calls. First call is not proof of warm-up completion or captured model loading. render_s is shared "
             "generation cost, not navigation compute; matlab_s is simulation cost and null means no measurement. "
             "Timing components may overlap and are not summed. LiDAR timing without lidar_sample can include idle rows. "
             "Device/thread/resolution/precision/synchronization metadata is retained in runs.json; missing metadata precludes "
             "hardware-normalized or GPU-synchronized real-time claims.", "",
             f"Current coordinator execution: `{json.dumps((summary.get('runtime') or {}).get('execution', {}))}`. "
             "Each retained measurement uses its own originating execution metadata below. "
             "When timing_mode is shared_resources, other cases may use the same GPU/CPU during a measured block. "
             "Those times describe concurrent throughput conditions, not isolated algorithm latency. "
             "Use a separate --workers 1 experiment for latency comparisons; scheduling does not change accuracy cohorts.", "",
             *_provenance_report(summary),
             "## Missing evidence and limitations", "",
             "No sensor generation, inference, replay, training or controller evaluation is performed by this module. "
             "The 240-flight design and seven methods are planned coverage, not generated results. Missing/generation-failed "
             "cases and every failed run remain in failure_manifest.json. Raw steps preserve rejection reasons and native diagnostics. "
             "A held diagnostic is not a new event: vision requires camera_sample, and LiDAR event statistics require lidar_sample. "
             "PnP solve rate uses explicit pnp_solved booleans and their known denominator, separately from vision acceptance. "
             "Innovation rejection rates use known gate evaluations (finite d²); association rejection is reported both "
             "as per-packet fractions and as rejected/input point counts. Missing telemetry leaves these rates unscored. "
             "Figure 5-10 innovation detail uses actual recorded *_mahalanobis_threshold fields; missing gates remain absent. "
             "Originating runtime provenance and attempt history are retained in summary.json; current manifest model loading "
             "and network input size describe the latest attempt. "
             "Logged covariance alone does not establish Gaussian consistency or independence.", ""]
    lines.extend("- "+warning for warning in summary["warnings"])
    (root/"analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def _numeric_findings(root, summary):
    """Describe only computed review-table values, never synthesize evidence."""
    def read(name):
        with (root/name).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    main, grouped = read("table_5_13.csv"), read("table_5_14.csv")
    formal = [r for r in main if r.get("cohort") == "complete" and r.get("condition") == "baseline"]
    if not formal:
        excluded = sum(r["cohort"].startswith("excluded") for r in summary["runs"])
        return [f"本批没有可纳入正式比较的完整架次；排除的运行数为 {excluded}。"
                "smoke/短测仅用于接口与绘图检查，不能据此报告正式导航精度或算法优劣。"
                "失败和未完成记录保留在失败清单中。"]
    def value(record, metric):
        try:
            return float(record.get(metric+"_estimate", ""))
        except (ValueError, TypeError):
            return None
    def formatted(number, percent=False):
        return "缺少可评分输出" if number is None else (f"{100*number:.1f}%" if percent else f"{number:.4g}")
    lines = ["以下数值来自紧凑汇总表，仅使用完整、未排除的 main baseline 架次，并按场景等权。"
             "它们描述本批记录中的观测结果，不构成因果归因、着舰成功或工程认证结论。", ""]
    for method in ("N0", "N1", "N2", "N3"):
        rows = [r for r in formal if r.get("method") == method and r.get("reference") == "gear" and r.get("stage") == "all"]
        if not rows:
            lines.append(f"- {method}：没有完整且可评分的正式数据。")
            continue
        by_set = {r["sample_set"]: r for r in rows}
        finite = value(by_set.get("finite", {}), "rmse_3d_m")
        valid = value(by_set.get("declared_valid", {}), "rmse_3d_m")
        fraction = value(by_set.get("timeline", {}), "availability")
        lines.append(f"- {method}：起落架三维 RMSE（全部有限 / 声明有效）为 {formatted(finite)} / {formatted(valid)} m；"
                     f"计划时长内有效比例 {formatted(fraction, True)}。")
    for label, records, view, dimension in (("天气", grouped, "weather", "weather"), ("阶段", formal, "overall", "stage")):
        candidates = [r for r in records if r.get("method") == "N3" and r.get("suite") == "main"
                      and r.get("condition") == "baseline" and r.get("cohort") == "complete"
                      and r.get("reference") == "gear" and r.get("sample_set") == "finite" and r.get("view") == view
                      and (r.get("stage") == "all" if dimension == "weather" else r.get("stage") in STAGES[1:])
                      and value(r, "rmse_3d_m") is not None]
        if candidates:
            worst = min(candidates, key=lambda r: (-value(r, "rmse_3d_m"), r[dimension]))
            lines.append(f"- N3 在本批已观测{label}中 RMSE 最高的是 {worst[dimension]}："
                         f"{value(worst, 'rmse_3d_m'):.4g} m（{worst.get('rmse_3d_m_n_units', '?')} 个独立记录）。")
    pairs = [r for r in grouped if r.get("comparison") == "method" and r.get("cohort") == "complete/complete"
             and r.get("reference") == "gear" and r.get("stage") == "all" and r.get("condition") == "baseline"
             and value(r, "rmse_3d_m") is not None]
    for method in sorted({r["method"] for r in pairs}):
        preferred = "common_camera_pair_"+method if method.startswith("A") else "common_camera"
        candidates = [r for r in pairs if r["method"] == method and r["sample_set"] in (preferred, "finite")]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda r: r["sample_set"] != preferred)
        interval = (chosen.get("rmse_3d_m_ci_low"), chosen.get("rmse_3d_m_ci_high"))
        text = f"- {method} − N3 的架次配对 RMSE 差值为 {value(chosen, 'rmse_3d_m'):.4g} m（{chosen['sample_set']}）。"
        if all(interval):
            low, high = map(float, interval)
            direction = "区间全为正" if low > 0 else "区间全为负" if high < 0 else "区间包含零"
            text += f" 95% 分层架次 bootstrap 区间 [{low:.4g}, {high:.4g}] m，{direction}；不作多重检验显著性声明。"
        else:
            text += " 独立重复不足，未给出置信区间。"
        lines.append(text)
    lines.extend(["", "全部有限与有效输出采用不同条件集合；共同相机集合、覆盖率、失效与缺失证据需结合查看。"
                  "天气/阶段的误差峰值是观测关联，不能单独归因于某个模块。"])
    return lines


def analyze(root: Path, bootstrap_samples=2000) -> dict:
    """Analyze existing benchmark records, writing tables, raw statistics and plots.

    Only data already in the manifest/results tree is evidence. Outputs replace
    this module's named artifacts; input files are never edited. The function
    nondefault argument overrides manifest.bootstrap_samples; otherwise the
    manifest count (or 2000) is used. The manifest seed is always used.
    """
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    if not isinstance(manifest.get("cases"), list):
        raise ValueError("manifest.cases must be a list")
    evaluation = _evaluation(manifest, bootstrap_samples)
    bootstrap_samples = evaluation["bootstrap_samples"]
    runs, failures = _discover(root, manifest)
    representative_id = _select_representative(runs)
    by_case = defaultdict(list)
    for run in runs:
        by_case[run.case["id"]].append(run)
    metrics, events, sensitivities, run_records, categorical = [], [], [], [], []
    representative, warnings = [], set()
    fields = _sample_fields()
    with (root/"samples.csv").open("w", newline="", encoding="utf-8") as csvfile, (root/"raw_steps.jsonl").open("w", encoding="utf-8") as raw:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for cid in sorted(by_case):
            common = _common_times(by_case[cid])
            for run in by_case[cid]:
                rows = list(_rows(run, audit=True))  # At most one flight in memory.
                meta = run.meta
                if run.planned is None or run.planned <= 0 or run.completed is None:
                    warnings.add(f"{cid}/{run.method}: missing/invalid duration metadata; time availability unscored")
                if rows and not any(isinstance(r.get("lidar_sample"), bool) for r in rows):
                    warnings.add(f"{cid}/{run.method}: no lidar_sample; LiDAR event statistics unscored")
                if rows and not any(_values(r, run.method, "cg")[2] is not None for r in rows):
                    warnings.add(f"{cid}/{run.method}: no finite matching CG estimate/truth")
                if rows and all(r["stage"] not in STAGES[1:] for r in rows if "stage" in r):
                    warnings.add(f"{cid}/{run.method}: no recognized task stages")
                if not rows:
                    warnings.add(f"{cid}/{run.method}: no usable step rows")
                for row in rows:
                    raw.write(json.dumps(_clean(dict(meta, row={k: v for k, v in row.items() if not k.startswith("_analysis_")})),
                                         ensure_ascii=False, allow_nan=False)+"\n")
                for flat in _sample_records(rows, run):
                    writer.writerow(flat)
                    if cid == representative_id:
                        representative.append(flat)
                recovery = _recoveries(rows, run)
                events.extend(dict(meta, reference="cg_shaped", **e) for e in recovery)
                for stage in STAGES:
                    stage_rows = rows if stage == "all" else [r for r in rows if r.get("stage") == stage]
                    for ref in REFERENCES:
                        base = dict(meta, reference=ref, stage=stage, planned_duration_s=run.planned, common_methods="")
                        selections = {s: ([], set()) for s in SETS}
                        selections.update({s: (members, times[ref]) for s, (members, times) in common.items() if run.method in members})
                        for sample_set, (members, times) in selections.items():
                            metrics.append(dict(base, sample_set=sample_set,
                                                **_accuracy(stage_rows, run.method, ref, sample_set, times),
                                                **({"common_methods": ",".join(members)} if members else {})))
                        availability, intervals = _availability(rows, run, ref, stage, evaluation)
                        metrics.append(dict(base, sample_set="timeline", **availability))
                        if stage == "all":
                            events.extend(dict(meta, reference=ref, stage="all", **event) for event in intervals)
                        for key, thresholds in (("false_valid_error_m", evaluation["false_valid_sensitivity_m"]),
                                                ("max_outage_s", evaluation["outage_sensitivity_s"])):
                            for threshold in thresholds:
                                sensitivity, _ = _availability(rows, run, ref, stage, dict(evaluation, **{key: threshold}))
                                sensitivities.append(dict(base, threshold_kind=key, threshold=threshold,
                                                          false_valid_error_m=threshold if key == "false_valid_error_m" else evaluation["false_valid_error_m"],
                                                          max_outage_s=threshold if key == "max_outage_s" else evaluation["max_outage_s"], **sensitivity))
                    stage_recovery = recovery if stage == "all" else [e for e in recovery if e["stage"] == stage]
                    metrics.append(dict(meta, reference="diagnostics", stage=stage, sample_set="events", common_methods="",
                                        planned_duration_s=run.planned, **_diagnostics(stage_rows, run, stage_recovery)))
                    for sensor in ("vision", "lidar"):
                        sample_key = "camera_sample" if sensor == "vision" else "lidar_sample"
                        for key in (sensor+"_rejection",) + (("vision_source", "vision_path") if sensor == "vision" else ()):
                            counts = Counter(str(r["diagnostics"][key]) for r in stage_rows
                                             if r.get(sample_key) is True and r["diagnostics"].get(key) is not None)
                            categorical.extend(dict(meta, stage=stage, field=key, value=value, count=count) for value, count in sorted(counts.items()))
                metrics.extend(dict(meta, common_methods="", planned_duration_s=run.planned, **record) for record in _timing_metrics(rows))
                for issue in run.issues:
                    warnings.add(f"{cid}/{run.method}: {issue}")
                resources, provenance = _run_resources(run.info, manifest)
                run_records.append(dict(meta, status=run.info.get("status"), planned_duration_s=run.planned,
                                        completed_duration_s=run.completed, usable_rows=len(rows), issues=run.issues,
                                        resources=resources, metadata=run.info, **provenance))
    pairs = _paired(metrics, runs, warnings)
    _csv(root/"flight_metrics.csv", metrics)
    _csv(root/"statistical_raw.csv", _long_metrics(metrics), list(DIMENSIONS)+["metric", "value", "weight", "sample_count"])
    _csv(root/"paired_differences.csv", pairs)
    _csv(root/"events.csv", events)
    _csv(root/"categorical_diagnostics.csv", categorical)
    _csv(root/"sensitivity.csv", sensitivities)
    _wide_tables(root, metrics, pairs, bootstrap_samples, evaluation["bootstrap_seed"])
    _json(root/"flight_metrics.json", metrics)
    _json(root/"paired_differences.json", pairs)
    _json(root/"runs.json", run_records)
    _json(root/"failure_manifest.json", failures)
    zoom = _zoom_window(representative)
    _csv(root/"figure_5_8.csv", representative, fields)
    cg_samples = [dict(r, zoom_selected=bool(zoom and zoom["start_s"] <= r["timestamp"] <= zoom["end_s"]))
                  for r in representative if r["method"] == "N3"]
    _csv(root/"figure_5_9.csv", cg_samples, fields+["zoom_selected"])
    _plots(root, representative, metrics, zoom)
    provenance = _provenance_summary(run_records)
    if provenance["mixed_attempts"]:
        warnings.add("Results span multiple attempts; timing aggregates may mix runtime environments")
    if provenance["unknown_attempt_runs"] or provenance["unknown_recording_attempt_runs"] or provenance["unknown_runtime_runs"]:
        warnings.add("Some originating attempts/runtimes are unknown; current manifest resources are not attributed to them")
    storage = storage_summary(root)
    _json(root/"storage_summary.json", storage)
    summary = dict(schema_version=1, case_count=len(manifest["cases"]), evaluation=evaluation, storage=storage,
                   runtime=manifest.get("runtime"), model_load_s=manifest.get("model_load_s"),
                   runtime_scope="current manifest attempt; originating runtime is recorded per result",
                   attempts=manifest.get("attempts", []), provenance=provenance,
                   model_load_note=manifest.get("model_load_note"), workers=manifest.get("workers",[]),
                   network_input_size=manifest.get("network_input_size"),
                   runs=run_records, flight_metrics=metrics, failures=failures,
                   representative_case=representative_id, zoom_window=zoom, warnings=sorted(warnings),
                   representative_cohort=next((r.cohort for r in runs if r.case["id"] == representative_id), None),
                   pairing_counts=dict(Counter(p["comparison"] for p in pairs)),
                   artifacts=["table_5_13.csv", "table_5_14.csv", "flight_metrics.csv", "flight_metrics.json",
                              "statistical_raw.csv", "statistics_summary.csv", "paired_differences.csv", "paired_differences.json", "sensitivity.csv",
                              "samples.csv", "raw_steps.jsonl", "events.csv", "categorical_diagnostics.csv", "runs.json",
                              "failure_manifest.json", "figure_5_8.csv", "figure_5_8.png", "figure_5_9.csv", "figure_5_9.png",
                              "figure_5_10.csv", "figure_5_10.png", "figure_5_10_innovations.csv",
                              "figure_5_10_innovations.png", "analysis.md", "summary.json", "storage_summary.json"])
    _report(root, summary)
    _json(root/"summary.json", summary)
    return _clean(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.root, args.bootstrap_samples)
    print(json.dumps(dict(cases=result["case_count"], failures=len(result["failures"]), artifacts=result["artifacts"])))
