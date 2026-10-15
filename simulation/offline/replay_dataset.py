"""Replay RGB, synchronized gimbal encoder and LiDAR, without plant/inner loop.

Truth is read only for scoring/plotting and the explicitly selected initial prior.
No truth poses, velocities or target points are fed into subsequent updates.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from navigation.common.contracts import CameraCalibration, CameraFrame, GimbalSample, LidarFrame
from navigation.common.reference_points import body_point_estimate
from navigation.common.frames import invert, transform_points
from navigation.fusion.eskf_state import ESKFState
from navigation.pipeline import NavigationPipeline
from navigation.vision.infer_h import HDetector
from visual_training.data.manifest import WEATHERS, load_manifest
from visual_training.metrics import grouped_metrics


class FixedDetection:
    detection = None
    def __call__(self, image):
        return copy.deepcopy(self.detection)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def position_metrics(rows):
    e = np.array([r["error_deck_m"] for r in rows])
    norms = np.linalg.norm(e, axis=1)
    ages = [r["age_since_vision_s"] for r in rows]
    return dict(frames=len(rows), rmse_3d_m=float(np.sqrt(np.mean(norms**2))),
                p95_3d_m=float(np.percentile(norms, 95)), max_3d_m=float(norms.max()),
                rmse_xyz_m=np.sqrt(np.mean(e**2, axis=0)).tolist(),
                last_error_3d_m=float(norms[-1]),
                vision_accepted=sum(r["vision_accepted"] for r in rows),
                lidar_accepted=sum(r["lidar_accepted"] for r in rows),
                mask_recoveries=sum(r["diagnostics"].get("vision_source") == "mask_recovery" and r["vision_accepted"] for r in rows),
                max_observed_vision_gap_s=max(ages),
                max_update_correction_m=max(r["update_correction_m"] for r in rows))


def plot_sequence(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = {r["stage"]: r["timestamp"] for r in reversed(payload["variants"]["current"])}
    for zoom in (False, True):
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout="constrained")
        current = payload["variants"]["current"]
        t = np.array([r["timestamp"] for r in current])
        truth = np.array([r["truth_deck_m"] for r in current])
        for j in range(3):
            axes[j].plot(t, truth[:, j], "k--", lw=1.6, label="Truth")
        for name, frames in payload["variants"].items():
            x = np.array([r["estimate_deck_m"] for r in frames])
            for j in range(3):
                axes[j].plot(t, x[:, j], ".-", lw=1.1, ms=2, label=name)
            axes[3].plot(t, [np.linalg.norm(r["error_deck_m"]) for r in frames], label=f"{name} error")
        failed = np.array([not r["vision_accepted"] for r in current])
        axes[3].plot(t[failed], np.zeros(failed.sum()), "rx", ms=4, label="Vision rejected")
        for ax, label in zip(axes, ("X / bow (m)", "Y / port (m)", "Z / up (m)", "3D error (m)")):
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            for boundary in list(stages.values())[1:]:
                ax.axvline(boundary, c="gray", ls=":", lw=.8)
            ax.legend(loc="upper right", fontsize=8)
        if zoom and "touchdown" in stages:
            start = stages["touchdown"]
            axes[-1].set_xlim(start, t[-1])
            active = t >= start
            for j in range(3):
                values = [truth[active, j]] + [np.array([r["estimate_deck_m"][j] for r in f])[active] for f in payload["variants"].values()]
                lo, hi = np.min(values), np.max(values)
                pad = max(.03, (hi-lo)*.08)
                axes[j].set_ylim(lo-pad, hi+pad)
            error_max = max(np.linalg.norm(r["error_deck_m"]) for f in payload["variants"].values() for r in f if r["timestamp"] >= start)
            axes[3].set_ylim(-.03, max(.1, error_max*1.1))
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(payload["sequence"] + (" / touchdown" if zoom else " / full trajectory"))
        for extension in ("png", "pdf"):
            fig.savefig(Path(output) / f"{payload['sequence']}_{'touchdown' if zoom else 'full'}.{extension}", dpi=150)
        plt.close(fig)


def replay_sequence(records, root, detector, config, args):
    first = records[0][1]
    calibration = CameraCalibration(first["K"], first["width"], first["height"], first["distortion"],
                                    first["T_camera_lidar"], first["lidar_mount"], first["T_body_lidar"])
    cfg = dict(config, checkpoint=None, filter_point_body=first["geometry"]["gear_body_m"])
    if args.initialization == "truth":
        p, R = np.array(first["gear_relative_deck_m"]), np.array(first["T_deck_camera"])[:3, :3]
    else:
        p, R = np.array(cfg["initial_position"]), np.array(cfg["initial_rotation"])
    p = p + args.initial_position_error
    R = R @ Rotation.from_euler("xyz", args.initial_rotation_error, degrees=True).as_matrix()
    configs = {"current": cfg}
    if args.compare_baseline:
        configs["baseline"] = dict(cfg, mask_geometry=False, adaptive_pixel_noise=False, minimum_keypoint_visibility=0., refine_pnp=False, acceleration_noise=.5)
    pipelines = {name: NavigationPipeline(calibration, first["points_deck"], c,
                 ESKFState(position=p.copy(), rotation=R.copy(), timestamp=first["timestamp"])) for name, c in configs.items()}
    fixed = FixedDetection()
    for pipeline in pipelines.values():
        pipeline.detector = fixed
    results = {name: [] for name in pipelines}
    last_vision = {name: first["timestamp"] for name in pipelines}
    errors, visible, metric_rows = [], [], []
    world_frame_difference = 0.
    for index, (path, row) in enumerate(records):
        # Coordinate audit is evaluation only; this value never enters fusion.
        truth_world = transform_points(invert(np.asarray(row["T_world_deck"])) @ np.asarray(row["T_world_body"]),
                                       np.asarray(row["geometry"]["gear_body_m"])[None])[0]
        world_frame_difference = max(world_frame_difference, float(np.max(np.abs(truth_world - row["gear_relative_deck_m"]))))
        if world_frame_difference > 1e-7:
            raise ValueError(f"Truth reference frame mismatch: {path}")
        image = cv2.imread(str(root / row["image"]))
        if image is None:
            raise FileNotFoundError(root / row["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fixed.detection = detector(image)
        encoder = GimbalSample(**row["gimbal"])
        visibility = np.asarray(row.get("keypoint_train_visibility", row["visibility"]), bool)
        # Missing detections fail the four-point success rate. They have no
        # meaningful pixel error; report their count, never a fabricated value.
        predicted = fixed.detection.keypoints if fixed.detection else np.full((4, 2), np.nan)
        errors.append(np.linalg.norm((predicted - row["keypoints"]) / [row["width"], row["height"]] * detector.image_size, axis=1))
        visible.append(visibility)
        metric_rows.append(row)
        with np.load(path.with_name(path.stem + "_lidar.npz"), allow_pickle=False) as data:
            if abs(float(data["timestamp"]) - row["timestamp"]) > 1e-8:
                raise ValueError(f"LiDAR timestamp mismatch: {path}")
            lidar = LidarFrame(row["timestamp"], data["points"].copy(), encoder)
        for name, pipeline in pipelines.items():
            pipeline.advance(row["timestamp"])
            predicted_position = pipeline.state.position.copy()
            pipeline.process_camera(CameraFrame(row["timestamp"], image, encoder))
            estimate = pipeline.process_lidar(lidar)
            position, velocity, covariance = body_point_estimate(estimate, first["geometry"]["gear_body_m"])
            if not np.allclose(position, pipeline.state.position, atol=1e-8):
                raise ValueError("Camera-to-gear roundtrip failed")
            if not np.isfinite(position).all():
                raise ValueError("Nonfinite estimate")
            diagnostics = clean(estimate.diagnostics)
            accepted = diagnostics.get("vision_accepted", False)
            if accepted:
                last_vision[name] = row["timestamp"]
            truth = np.asarray(row["gear_relative_deck_m"])
            results[name].append(dict(timestamp=row["timestamp"], stage=row["stage"], frame_id=row["frame_id"],
                truth_deck_m=truth.tolist(), estimate_deck_m=position.tolist(), error_deck_m=(position-truth).tolist(),
                velocity_deck_m_s=velocity.tolist(), position_std_m=np.sqrt(np.maximum(np.diag(covariance), 0)).tolist(),
                vision_accepted=accepted, lidar_accepted=diagnostics.get("lidar_accepted", False), healthy=estimate.healthy,
                age_since_vision_s=row["timestamp"]-last_vision[name],
                update_correction_m=float(np.linalg.norm(position-predicted_position)), diagnostics=diagnostics))
        if (index + 1) % 60 == 0:
            print(json.dumps(dict(event="replay_progress", sequence=first["sequence_id"], frames=index+1)), flush=True)
    metrics = {name: {stage: position_metrics(frames if stage == "all" else [r for r in frames if r["stage"] == stage])
                    for stage in ["all"] + list(dict.fromkeys(r["stage"] for r in frames))} for name, frames in results.items()}
    return dict(sequence=first["sequence_id"], weather=first["weather"], split=args.split,
                initialization=dict(mode=args.initialization, position=p.tolist(), camera_rotation=R.tolist(), velocity=[0,0,0]),
                navigation_configs=configs, variants=results, metrics=metrics,
                consistency_checks=dict(world_to_deck_max_difference_m=world_frame_difference),
                vision_metrics=grouped_metrics(np.array(errors), np.array(visible), metric_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/landing_four_weather")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="configs/navigation.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--mode", choices=("test", "full"), default="test", help="test: one complete sortie per weather; full: all selected sorties")
    parser.add_argument("--weathers", nargs="+", choices=WEATHERS, default=list(WEATHERS))
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--initialization", choices=("truth", "config"), default="truth")
    parser.add_argument("--initial-position-error", type=float, nargs=3, default=[0,0,0])
    parser.add_argument("--initial-rotation-error", type=float, nargs=3, default=[0,0,0])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    root = Path(args.data_root).resolve()
    records = load_manifest(root / "splits" / f"{args.split}.json", root, args.weathers)
    sequences = {}
    for path, row in records:
        if args.sequences and row["sequence_id"] not in args.sequences:
            continue
        sequences.setdefault(row["sequence_id"], []).append((path, row))
    if args.sequences and set(args.sequences) - set(sequences):
        raise ValueError("Requested sequences are absent from selected split/weathers")
    if args.mode == "test" and not args.sequences:
        chosen = {}
        for seq, items in sorted(sequences.items()):
            chosen.setdefault(items[0][1]["weather"], seq)
        sequences = {seq: sequences[seq] for seq in chosen.values()}
    output = Path(args.output or time.strftime("outputs/replay/%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", dict(status="running", arguments=vars(args),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        frame="deck origin=landing center; +X bow, +Y port, +Z up; vector landing center -> gear center",
        limitation="Sparse sampled sensor replay. Truth used only for scoring and optional initial prior; no inner loop or smoothing."))
    detector = HDetector(args.checkpoint, args.device)
    run = json.loads((output / "run.json").read_text())
    run["runtime"] = dict(torch=str(torch.__version__), numpy=np.__version__, opencv=cv2.__version__,
                          device=args.device, image_size=detector.image_size, precision="fp32")
    write_json(output / "run.json", run)
    config = yaml.safe_load(Path(args.config).read_text())
    summaries = []
    aggregate_frames = {}
    for seq, items in sorted(sequences.items()):
        items.sort(key=lambda r: r[1]["timestamp"])
        if any(b[1]["timestamp"] <= a[1]["timestamp"] for a, b in zip(items, items[1:])):
            raise ValueError(f"Non-increasing timestamps: {seq}")
        payload = replay_sequence(items, root, detector, config, args)
        write_json(output / f"{seq}.json", payload)
        for variant, frames in payload["variants"].items():
            aggregate_frames.setdefault(variant, []).extend(frames)
            with (output / f"{seq}_{variant}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s", "stage", "truth_x", "truth_y", "truth_z", "estimate_x", "estimate_y", "estimate_z", "error_x", "error_y", "error_z", "vision_accepted", "lidar_accepted", "vision_age_s", "update_correction_m", "vision_rejection", "vision_source"])
                for r in frames:
                    writer.writerow([r["timestamp"], r["stage"], *r["truth_deck_m"], *r["estimate_deck_m"], *r["error_deck_m"], r["vision_accepted"], r["lidar_accepted"], r["age_since_vision_s"], r["update_correction_m"], r["diagnostics"].get("vision_rejection", ""), r["diagnostics"].get("vision_source", "")])
        plot_sequence(payload, output)
        summaries.append({k: v for k,v in payload.items() if k not in ("variants", "navigation_configs")})
        print(json.dumps(dict(event="sequence_complete", sequence=seq,
              touchdown={name: m.get("touchdown", m["all"]) for name, m in payload["metrics"].items()})), flush=True)
    write_json(output / "summary.json", summaries)
    stages = ["all"] + list(dict.fromkeys(r["stage"] for r in aggregate_frames["current"]))
    aggregate = {stage: {name: position_metrics([r for r in frames if stage == "all" or r["stage"] == stage])
                        for name, frames in aggregate_frames.items()} for stage in stages}
    # An aggregate has no unique last frame; report terminal errors by sortie.
    for stage in aggregate.values():
        for metrics in stage.values():
            metrics.pop("last_error_3d_m", None)
    write_json(output / "aggregate_metrics.json", aggregate)
    lines = ["# 离线导航回放", "", "起落架中心相对着舰中心，甲板坐标 +X舰艏、+Y左舷、+Z向上；米。", "",
             f"初始化：{args.initialization}；只在首帧使用指定先验，后续不使用真值更新。稀疏采样回放，不运行内环。", "",
             "|航次|算法|着舰RMSE m|P95 m|最大误差 m|末帧 m|最长视觉间隔 s|最大更新校正 m|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, d in aggregate.get("touchdown", aggregate["all"]).items():
        lines.append(f"|全部|{name}|{d['rmse_3d_m']:.4f}|{d['p95_3d_m']:.4f}|{d['max_3d_m']:.4f}|—|{d['max_observed_vision_gap_s']:.2f}|{d['max_update_correction_m']:.4f}|")
    for p in summaries:
        for name, m in p["metrics"].items():
            d=m.get("touchdown", m["all"])
            lines.append(f"|{p['sequence']}|{name}|{d['rmse_3d_m']:.4f}|{d['p95_3d_m']:.4f}|{d['max_3d_m']:.4f}|{d['last_error_3d_m']:.4f}|{d['max_observed_vision_gap_s']:.2f}|{d['max_update_correction_m']:.4f}|")
    lines += ["", "逐帧JSON/CSV记录拒绝原因、几何恢复来源、协方差与视觉间隔。图中的红叉表示视觉拒绝。雷达平面不直接观测水平位置或偏航；长时间视觉不可用仍会退化。", ""]
    for p in summaries:
        lines += [f"![{p['sequence']}]({p['sequence']}_touchdown.png)", ""]
    (output / "analysis.md").write_text("\n".join(lines))
    run = json.loads((output / "run.json").read_text())
    run.update(status="complete", sequences=len(summaries))
    write_json(output / "run.json", run)
    print(f"Replay complete: {output}", flush=True)


if __name__ == "__main__":
    main()
