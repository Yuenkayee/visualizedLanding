"""Tail-aware metrics in network-input pixels, plus automatic training artifacts."""
import json
from pathlib import Path
import numpy as np


def keypoint_metrics(errors, visible):
    errors, visible = np.asarray(errors), np.asarray(visible, bool)
    values = errors[visible]
    missing = int((~np.isfinite(values)).sum())
    values = values[np.isfinite(values)]
    full = visible.all(axis=1)
    return dict(frames=len(errors), visible_keypoints=int(visible.sum()),
                unlocalized_visible_keypoints=missing,
                mean_px=float(values.mean()) if len(values) else None,
                p95_px=float(np.percentile(values, 95)) if len(values) else None,
                p99_px=float(np.percentile(values, 99)) if len(values) else None,
                max_px=float(values.max()) if len(values) else None,
                points_over_5px=int((values > 5).sum()), fully_visible_frames=int(full.sum()),
                four_point_success_5px=float((errors[full].max(axis=1) <= 5).mean()) if full.any() else None)


def grouped_metrics(errors, visible, rows):
    result = {"all": keypoint_metrics(errors, visible), "by_condition": {}}
    keys = {}
    for i, r in enumerate(rows):
        stage, weather = r.get("stage", "unspecified"), r.get("weather", "unspecified")
        for key in (f"stage/{stage}", f"weather/{weather}", f"stage_weather/{stage}/{weather}"):
            keys.setdefault(key, []).append(i)
    for key, indices in keys.items():
        result["by_condition"][key] = keypoint_metrics(errors[indices], visible[indices])
    return result


def training_report(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for name in ("train", "val"):
        axes[0, 0].plot(epochs, [r[f"{name}_loss"] for r in history], label=name)
        axes[1, 0].plot(epochs, [r["phases"][name]["loss_components"]["heatmap"] for r in history], label=name)
    for key in ("mean_px", "p95_px", "p99_px"):
        axes[0, 1].plot(epochs, [r["keypoint_metrics"]["all"][key] for r in history], label=key)
    axes[1, 1].plot(epochs, [r["keypoint_metrics"]["all"]["four_point_success_5px"] for r in history], label="four-point success <=5px")
    for ax, title in zip(axes.flat, ("Loss (soft targets change absolute scale)", "Validation keypoint error (input px)", "Heatmap loss", "Validation complete target frames")):
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)
    best = min(history, key=lambda r: r["val_loss"])
    tail = best["keypoint_metrics"]["all"]
    (out / "analysis.md").write_text(
        f"训练完成 {len(history)} 轮；最低验证损失在第 {best['epoch']} 轮。\n\n"
        f"该轮关键点指标（输入图像像素）：\n\n```json\n{json.dumps(tail, indent=2)}\n```\n\n"
        "![训练曲线](training_curves.png)\n\n"
        "epochs.json 包含每轮分项损失、梯度范数、学习率、分阶段/天气尾部指标。"
        "best.pt 与 best_loss.pt 按验证损失选取；best_keypoint.pt 优先按四点成功率、再按P99和均值选取。"
        "这些视觉指标不能代替位姿评价，请用离线回放比较候选权重；未自动宣称 best_pose。\n")
