"""Analyze closed-loop state/feedback logs without MATLAB or the training set."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np


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

    def statistics(mask):
        if not mask.any():
            return None
        return dict(
            samples=int(mask.sum()),
            rmse_xyz_m=np.sqrt(np.mean(error[mask] ** 2, axis=0)).tolist(),
            rmse_3d_m=float(np.sqrt(np.mean(norm[mask] ** 2))),
            p95_3d_m=float(np.percentile(norm[mask], 95)),
            maximum_3d_m=float(norm[mask].max()),
        )

    metrics = dict(
        iterations=len(rows),
        duration_s=rows[-1]["next_timestamp"] - t[0],
        valid_fraction=float(valid.mean()),
        all=statistics(np.ones(len(rows), bool)),
        valid=statistics(valid),
        next_state_clock_checked=all(
            abs(r["next_timestamp"] - r["timestamp"] - r["dt"]) < 1e-8 for r in rows
        ),
        final_truth_gear_deck_m=rows[-1]["next_truth"]["gear_position_deck"],
        note="Position error is wheel-centre minus H-centre in bow/port/up deck axes; completion is not a touchdown certification.",
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
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    (root / "analysis.md").write_text(
        f"# 离线闭环结果\n\n完成迭代：{len(rows)}；有效反馈比例：{valid.mean():.1%}。\n\n"
        f"全时段三维 RMSE：{metrics['all']['rmse_3d_m']:.3f} m；P95：{metrics['all']['p95_3d_m']:.3f} m。\n\n"
        "`position_comparison.png/pdf` 为起落架中心相对 H 中心的位置真值与估计；红点为无效反馈。"
        "`controller_feedback.png` 是实际写入控制器的 NED 位置。`comparison.csv` 保存数据。\n\n"
        "请结合 `run.json` 的运行类型、渲染器、权重和完成状态解读。接口验收模型不代表直升机着舰性能；"
        "结束仿真不等于已触舰。无效反馈时的控制策略由接入的控制器负责。\n"
    )
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
