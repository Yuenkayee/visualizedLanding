"""Analyze closed-loop state/feedback logs without MATLAB or the training set."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np


def ned_comparison(rows):
    """Use logged conversion outputs, or reconstruct NED truth for old logs."""
    truth, estimate = [], []
    ned_from_world = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    for row in rows:
        if "gear_relative_truth_ned" in row:
            truth.append(np.asarray(row["gear_relative_truth_ned"]).reshape(3))
        elif "T_world_deck" in row["truth"]:
            rotation = np.asarray(row["truth"]["T_world_deck"])[:3, :3]
            truth.append(ned_from_world @ rotation @ row["truth"]["gear_position_deck"])
        else:
            # A deck attitude is needed; do not silently assume a level deck.
            return None
        estimate.append(np.asarray(row.get(
            "gear_relative_estimate_ned", row["feedback"]["gear_relative_ned_m"]
        )).reshape(3))
    return np.asarray(truth), np.asarray(estimate)


def error_statistics(error, mask):
    if not mask.any():
        return None
    norm = np.linalg.norm(error[mask], axis=1)
    return dict(
        samples=int(mask.sum()),
        rmse_xyz_m=np.sqrt(np.mean(error[mask] ** 2, axis=0)).tolist(),
        rmse_3d_m=float(np.sqrt(np.mean(norm ** 2))),
        p95_3d_m=float(np.percentile(norm, 95)),
        maximum_3d_m=float(norm.max()),
    )


def cg_analysis(rows, root):
    """Raw conversion vs causal 50 Hz signal, with separate validity masks."""
    import matplotlib.pyplot as plt
    keys = ("cg_relative_truth_ned", "cg_relative_estimate_ned", "cg_relative_estimate_50hz_ned")
    if not all(all(k in row for k in keys) for row in rows):
        return None  # Legacy logs cannot supply CG truth without extra geometry.
    t = np.array([r["timestamp"] for r in rows])
    truth, raw, signal = (np.array([r[k] for r in rows]).reshape(-1, 3) for k in keys)
    valid = np.array([r["feedback"]["valid"] for r in rows], bool)
    signal_valid = np.array([r["cg_signal_valid"] for r in rows], bool)
    common = valid & signal_valid
    all_rows = np.ones(len(rows), bool)
    result = dict(direction="helicopter CG minus ship CG", axes=["north", "east", "down"],
                  raw_valid_fraction=float(valid.mean()), signal_valid_fraction=float(signal_valid.mean()),
                  clock_50hz_checked=bool(np.allclose(np.diff(t), .02, atol=1e-8, rtol=0)),
                  raw=dict(all=error_statistics(raw-truth, all_rows), valid=error_statistics(raw-truth, valid),
                           common_valid=error_statistics(raw-truth, common)),
                  signal_50hz=dict(all=error_statistics(signal-truth, all_rows),
                                  valid=error_statistics(signal-truth, signal_valid),
                                  common_valid=error_statistics(signal-truth, common)),
                  maximum_signal_raw_difference_m=float(np.linalg.norm(signal-raw, axis=1).max()),
                  max_navigation_packet_age_s=max(r["cg_signal_age_s"] for r in rows),
                  note="Signal validity does not certify control performance. Signal age is navigation packet age; sensor age is recorded separately.")
    velocities = np.array([r["navigation_velocity_cg_ned_mps"] for r in rows]).reshape(-1, 3)
    dt = np.diff(t)
    for name, values in (("raw", raw), ("signal_50hz", signal)):
        if len(t) > 1:
            # Position change unexplained by estimated velocity, not total flight motion.
            innovations = np.diff(values, axis=0) - .5 * (velocities[1:]+velocities[:-1]) * dt[:, None]
            result[name]["max_step_residual_m"] = float(np.linalg.norm(innovations, axis=1).max())
    with (root / "cg_relative_ned.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "raw_valid", "signal_valid", "navigation_packet_age_s", *[
            f"{kind}_{axis}_m" for kind in ("truth", "raw", "signal_50hz", "raw_error", "signal_error")
            for axis in ("north", "east", "down")], "truth_distance_m", "raw_distance_m", "signal_distance_m"])
        for i, row in enumerate(rows):
            writer.writerow([t[i], int(valid[i]), int(signal_valid[i]), row["cg_signal_age_s"],
                *truth[i], *raw[i], *signal[i], *(raw[i]-truth[i]), *(signal[i]-truth[i]),
                np.linalg.norm(truth[i]), np.linalg.norm(raw[i]), np.linalg.norm(signal[i])])
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for k, axis in enumerate(("North", "East", "Down")):
        axes[k].plot(t, truth[:, k], color="black", lw=1.7, label="CG relative truth")
        axes[k].plot(t, raw[:, k], color="tab:orange", lw=.9, alpha=.8, label="Raw navigation")
        axes[k].plot(t, signal[:, k], color="tab:blue", lw=1.1, label="Causal 50 Hz signal")
        if (~signal_valid).any():
            axes[k].scatter(t[~signal_valid], signal[~signal_valid, k], color="red", s=8, label="Invalid 50 Hz signal")
        axes[k].set_ylabel(f"{axis} (m)")
    axes[3].plot(t, np.linalg.norm(raw-truth, axis=1), color="tab:orange", label="Raw 3D error")
    axes[3].plot(t, np.linalg.norm(signal-truth, axis=1), color="tab:blue", label="50 Hz 3D error")
    axes[3].set_ylabel("Error (m)")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend(loc="best"); axes[3].legend(loc="best")
    axes[-1].set_xlabel("Simulation time (s)")
    fig.savefig(root / "cg_relative_ned_comparison.png", dpi=160)
    fig.savefig(root / "cg_relative_ned_comparison.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    for value, label, color in ((truth, "Truth", "black"), (raw, "Raw navigation", "tab:orange"),
                                (signal, "Causal 50 Hz signal", "tab:blue")):
        ax.plot(t, np.linalg.norm(value, axis=1), label=label, color=color)
    ax.set(xlabel="Simulation time (s)", ylabel="CG separation norm (m)")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(root / "cg_distance_comparison.png", dpi=160)
    plt.close(fig)
    return result


def recovery_metrics(rows):
    """Count camera packets, not diagnostic values held over host iterations."""
    invalid = []
    start = None
    for i, row in enumerate(rows):
        if not row["feedback"]["valid"] and start is None:
            start = row["timestamp"]
        if start is not None and (row["feedback"]["valid"] or i == len(rows)-1):
            end = row["timestamp"] if row["feedback"]["valid"] else row["next_timestamp"]
            invalid.append(dict(start_s=start, end_s=end, duration_s=end-start))
            start = None
    camera = [r.get("estimate", {}).get("diagnostics", {}) for r in rows if r.get("sensors", {}).get("camera")]
    ages = [r.get("estimate", {}).get("diagnostics", {}).get("measurement_age") for r in rows]
    ages = [age for age in ages if age is not None]
    attempts = [attempt for d in camera for attempt in d.get("vision_attempts", [])]
    return dict(invalid_intervals=invalid, invalid_duration_s=sum(v["duration_s"] for v in invalid),
        max_invalid_duration_s=max((v["duration_s"] for v in invalid), default=0.),
        max_pose_age_s=max(ages, default=None),
        accepted_paths=dict(Counter(d.get("vision_path", "legacy") for d in camera if d.get("vision_accepted"))),
        roi_attempted_frames=sum(bool(d.get("vision_roi_attempted")) for d in camera),
        retry_rejections=dict(Counter(a.get("vision_rejection", "unspecified") for a in attempts
                                      if a.get("vision_path") != "full" and not a.get("vision_accepted"))),
        reacquisitions=sum(bool(d.get("vision_reacquired")) for d in camera),
        max_vision_correction_m=max((d.get("vision_position_correction_m", 0.) for d in camera), default=0.))


def plot_recovery(rows, root):
    import matplotlib.pyplot as plt
    camera = [r for r in rows if r.get("sensors", {}).get("camera")]
    if not camera:
        return
    t = [r["timestamp"] for r in camera]
    diagnostics = [r.get("estimate", {}).get("diagnostics", {}) for r in camera]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, [d.get("measurement_age", np.nan) for d in diagnostics], label="Accepted pose age")
    axes[0].set_ylabel("Age (s)")
    axes[1].plot(t, [d.get("vision_position_correction_m", 0.) for d in diagnostics], label="Vision position correction")
    axes[1].set_ylabel("Correction (m)")
    paths = ["full", "roi", "roi_mask", "full_small_mask"]
    for i, p in enumerate(paths):
        times = [s for s, d in zip(t, diagnostics) if d.get("vision_accepted") and d.get("vision_path", "full") == p]
        axes[2].scatter(times, [i]*len(times), s=15, label=p)
    axes[2].set_yticks(range(len(paths)), paths)
    axes[2].set_ylabel("Accepted path")
    for ax in axes:
        for interval in recovery_metrics(rows)["invalid_intervals"]:
            ax.axvspan(interval["start_s"], interval["end_s"], color="red", alpha=.15)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[1].legend()
    axes[-1].set_xlabel("Simulation time (s)")
    fig.savefig(root / "navigation_recovery.png", dpi=160)
    plt.close(fig)
    with (root / "vision_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "accepted", "path", "primary_rejection", "final_rejection", "pose_age_s",
                         "position_correction_m", "recovery_pending", "roi", "attempts"])
        for s, d in zip(t, diagnostics):
            writer.writerow([s, d.get("vision_accepted"), d.get("vision_path"), d.get("vision_primary_rejection"),
                d.get("vision_rejection"), d.get("measurement_age"), d.get("vision_position_correction_m"),
                d.get("vision_recovery_pending"), json.dumps(d.get("vision_roi", {})), json.dumps(d.get("vision_attempts", []))])


def gimbal_analysis(rows, metadata, root):
    """Commands and geometric FOV coverage are evaluation, not navigation input."""
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    from navigation.common.frames import invert, transform_points
    from visual_modeling.blender.build_h_marker import marker_points

    selected = [r for r in rows if r.get("estimate", {}).get("gimbal_reference")]
    if not selected:
        return None
    t = np.array([r["timestamp"] for r in selected])
    commands = [r["estimate"]["gimbal_reference"] for r in selected]
    rates = np.array([c.get("angular_velocity_deck", [0, 0, 0]) for c in commands])
    error = [Rotation.from_matrix(np.asarray(c["R_deck_camera"]).T @
             np.asarray(r["truth"]["T_deck_camera"])[:3, :3]).magnitude()
             for c, r in zip(commands, selected)]
    states = [r["estimate"]["diagnostics"].get("gimbal_tracking", {}).get("status", "unknown") for r in selected]
    result = dict(status_counts=dict(Counter(states)),
                  max_reference_rate_rad_s=float(np.max(np.linalg.norm(rates, axis=1))),
                  max_camera_reference_error_rad=float(max(error)))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, np.rad2deg(error), label="Actual camera vs command")
    axes[0].set_ylabel("Angle error (deg)")
    axes[1].plot(t, np.rad2deg(np.linalg.norm(rates, axis=1)), label="Reference rate")
    axes[1].set_ylabel("Rate (deg/s)")
    camera = metadata.get("camera")
    if camera:
        K = np.array(camera["K"])
        marker = metadata.get("marker", {})
        points = marker_points(marker.get("width", 6), marker.get("length", 6))
        targets = np.vstack(([0, 0, .025], points))
        centres, coverage = [], []
        for r in selected:
            p = transform_points(invert(r["truth"]["T_deck_camera"]), targets)
            pixels = p @ K.T
            uv = pixels[:, :2] / np.where(abs(pixels[:, 2:]) > 1e-12, pixels[:, 2:], np.nan)
            normalized = 2 * uv / [camera["width"], camera["height"]] - 1
            centres.append(np.where(p[0, 2] > 0, normalized[0], np.nan))
            coverage.append(bool(np.all(p[1:, 2] > 0) and np.all(abs(normalized[1:]) <= 1)))
        centres = np.asarray(centres)
        axes[2].plot(t, centres[:, 0], label="H centre x / half width")
        axes[2].plot(t, centres[:, 1], label="H centre y / half height")
        axes[2].axhline(1, color="red", ls=":")
        axes[2].axhline(-1, color="red", ls=":")
        axes[2].set_ylabel("Normalized pixel")
        result["four_corners_in_frame_fraction"] = float(np.mean(coverage))
        result["fov_note"] = "Geometric projection only; does not test occlusion/detection."
    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(loc="best")
    fig.savefig(root / "gimbal_tracking.png", dpi=160)
    plt.close(fig)
    return result


def analyze(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(directory)
    rows = [
        json.loads(line)
        for line in (root / "steps.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No completed iterations to analyze")
    t = np.array([r["timestamp"] for r in rows])
    truth = np.array([r["truth"]["gear_position_deck"] for r in rows])
    estimated = np.array([r["gear_estimate_deck_m"] for r in rows])
    error = estimated - truth
    valid = np.array([r["feedback"]["valid"] for r in rows], bool)
    norm = np.linalg.norm(error, axis=1)

    metadata = json.loads((root / "run.json").read_text()) if (root / "run.json").exists() else {}
    metrics = dict(
        iterations=len(rows),
        duration_s=rows[-1]["next_timestamp"] - t[0],
        valid_fraction=float(valid.mean()),
        all=error_statistics(error, np.ones(len(rows), bool)),
        valid=error_statistics(error, valid),
        next_state_clock_checked=all(
            (r["timestamp"]-1e-8 <= r["next_timestamp"] <= r["timestamp"]+r["dt"]+1e-8
             if r.get("plant_stopped") else
             abs(r["next_timestamp"] - r["timestamp"] - r["dt"]) < 1e-8)
            for r in rows
        ),
        termination_reason=metadata.get("termination_reason"),
        stop_reason=metadata.get("stop_reason"),
        navigation_applied_to_control=metadata.get("navigation_applied_to_control"),
        final_truth_gear_deck_m=rows[-1]["next_truth"]["gear_position_deck"],
        note="Position error is wheel-centre minus H-centre in bow/port/up deck axes; completion is not a touchdown certification.",
    )
    metrics["observations"] = {}
    metrics["navigation_recovery"] = recovery_metrics(rows)
    plot_recovery(rows, root)
    metrics["gimbal_tracking"] = gimbal_analysis(rows, metadata, root)
    metrics["cg_relative"] = cg_analysis(rows, root)
    for sensor, prefix in (("camera", "vision"), ("lidar", "lidar")):
        fresh = [r.get("estimate", {}).get("diagnostics", {}) for r in rows
                 if r.get("sensors", {}).get(sensor)]
        rejected = [d for d in fresh if not d.get(f"{prefix}_accepted", False)]
        metrics["observations"][sensor] = dict(
            frames=len(fresh), accepted=len(fresh)-len(rejected),
            rejections=dict(Counter(d.get(f"{prefix}_rejection", "unspecified") for d in rejected)),
        )
    with (root / "comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_s",
                "valid",
                *[
                    f"{kind}_{axis}_m"
                    for kind in ("truth", "estimate", "error")
                    for axis in ("bow", "port", "up")
                ],
            ]
        )
        for i in range(len(rows)):
            w.writerow([t[i], int(valid[i]), *truth[i], *estimated[i], *error[i]])
    fig, axs = plt.subplots(
        4, 1, figsize=(11, 10), sharex=True, constrained_layout=True
    )
    for i, axis in enumerate(("Bow", "Port", "Up")):
        axs[i].plot(t, truth[:, i], label="Truth", linewidth=1.7)
        axs[i].plot(t, estimated[:, i], label="Navigation", linewidth=1.2)
        axs[i].set_ylabel(f"{axis} (m)")
        axs[i].grid(alpha=0.25)
        if (~valid).any():
            axs[i].scatter(
                t[~valid], estimated[~valid, i], c="red", s=10, label="Invalid feedback"
            )
    axs[0].legend(loc="best")
    axs[3].plot(t, norm, label="3D error")
    axs[3].set_ylabel("Error (m)")
    axs[3].set_xlabel("Simulation time (s)")
    axs[3].grid(alpha=0.25)
    fig.savefig(root / "position_comparison.png", dpi=160)
    fig.savefig(root / "position_comparison.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    feedback = np.array([r["feedback"]["gear_relative_ned_m"] for r in rows]).reshape(
        -1, 3
    )
    for i, axis in enumerate(("North", "East", "Down")):
        axs[i].plot(t, feedback[:, i])
        axs[i].set_ylabel(f"{axis} (m)")
        axs[i].grid(alpha=0.25)
    axs[-1].set_xlabel("Simulation time (s)")
    fig.savefig(root / "controller_feedback.png", dpi=160)
    plt.close(fig)
    ned = ned_comparison(rows)
    metrics["ned_comparison_available"] = ned is not None
    if ned is not None:
        truth_ned, estimate_ned = ned
        error_ned = estimate_ned - truth_ned
        metrics["ned"] = dict(
            axes=["north", "east", "down"],
            all=error_statistics(error_ned, np.ones(len(rows), bool)),
            valid=error_statistics(error_ned, valid),
        )
        with (root / "gear_relative_ned.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "valid", *[
                f"gear_relative_{kind}_ned_{axis}_m"
                for kind in ("truth", "estimate", "error")
                for axis in ("north", "east", "down")
            ]])
            for i in range(len(rows)):
                writer.writerow([t[i], int(valid[i]), *truth_ned[i], *estimate_ned[i], *error_ned[i]])
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        for i, axis in enumerate(("North", "East", "Down")):
            axs[i].plot(t, truth_ned[:, i], label="gear_relative_truth_ned", linewidth=1.7)
            axs[i].plot(t, estimate_ned[:, i], label="gear_relative_estimate_ned", linewidth=1.2)
            if (~valid).any():
                axs[i].scatter(t[~valid], estimate_ned[~valid, i], color="red", s=10, label="Invalid estimate")
            axs[i].set_ylabel(f"{axis} (m)")
            axs[i].grid(alpha=.25)
        axs[0].legend(loc="best")
        axs[-1].set_xlabel("Simulation time (s)")
        fig.savefig(root / "gear_relative_ned_comparison.png", dpi=160)
        fig.savefig(root / "gear_relative_ned_comparison.pdf")
        plt.close(fig)
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    (root / "analysis.md").write_text(
        f"# 离线闭环结果\n\n完成迭代：{len(rows)}；有效反馈比例：{valid.mean():.1%}。\n\n"
        f"全时段三维 RMSE：{metrics['all']['rmse_3d_m']:.3f} m；P95：{metrics['all']['p95_3d_m']:.3f} m。\n\n"
        f"相机更新：{metrics['observations']['camera']}；LiDAR更新：{metrics['observations']['lidar']}。\n\n"
        f"连续失效区间：{metrics['navigation_recovery']['invalid_intervals']}；"
        f"最大完整位姿观测年龄：{metrics['navigation_recovery']['max_pose_age_s']} s。\n\n"
        "`navigation_recovery.png` 展示观测年龄、视觉纠偏量和接受路径；"
        "`vision_diagnostics.csv` 保存每张相机图像的主路径拒绝、ROI 框及各级重试结果。\n\n"
        + ("`gimbal_tracking.png` 展示参考角速度、实际朝向与指令的差值，以及 H 中心的视场位置；"
           "视场投影统计不代表没有遮挡或视觉识别成功。\n\n" if metrics["gimbal_tracking"] else "")
        + ("本段没有有效导航估计；曲线包含先验外推，不能用较小误差认定导航已收敛。\n\n" if not valid.any() else "") +
        "`position_comparison.png/pdf` 为起落架中心相对 H 中心的位置真值与估计；红点为无效反馈。"
        "`controller_feedback.png` 是转换后的 NED 导航位置；旁路模式中不作用于控制器。`comparison.csv` 保存甲板系数据。\n\n"
        + ("`gear_relative_ned_comparison.png/pdf` 对比起落架减 H 中心的 North/East/Down 真值与估计；"
           "`gear_relative_ned.csv` 保存对应数据，`metrics.json` 的 `ned` 给出三个方向误差。\n\n"
           if ned is not None else "旧日志缺少甲板姿态，无法恢复 NED 真值。\n\n") +
        ("`cg_relative_ned_comparison.png/pdf` 对比直升机质心减舰船质心的 N/E/D 真值、原始导航值与 50 Hz 因果平滑值。"
         "`cg_distance_comparison.png` 为三维距离模长，`cg_relative_ned.csv` 保存逐步数据。"
         "`metrics.json` 的 `cg_relative` 分别统计两种输出及其共同有效区间误差；"
         "平滑不会保证精度必然提高，信号有效标志也不代表已验证控制可用性。\n\n" if metrics["cg_relative"] else "") +
        f"结束原因：{metrics['termination_reason']}；模型停止事件：{metrics['stop_reason']}；"
        f"导航接入控制：{metrics['navigation_applied_to_control']}。最后状态单独保存在 `terminal_state.json`（模型 STOP 时）。\n\n"
        "请结合 `run.json` 的运行类型、渲染器、权重和完成状态解读。接口验收模型不代表直升机着舰性能；"
        "结束仿真不等于已触舰。无效反馈时的控制策略由接入的控制器负责。\n"
    )
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
