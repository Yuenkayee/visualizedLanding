"""Draw a side projection of the configured proxy and belly-camera layout."""

import argparse
from pathlib import Path
import numpy as np
import yaml
from visual_modeling.sensors.airframe_visibility import airframe_primitives


def render_camera_mount(config, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon

    primitives = airframe_primitives(config["geometry"])
    mount = np.array(config["camera"]["position_body_m"])
    angle = np.deg2rad(config["camera"]["optical_down_deg"])
    gear = np.array(config["geometry"]["gear_body_m"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.1), layout="constrained")
    for ax in axes:
        for obj in primitives:
            p = obj["centre"]
            if "strut" in obj["name"]:
                extent = obj["R"][:, 2] * obj["axes"][2]
                start, end = p - extent, p + extent
                ax.plot([start[0], end[0]], [start[2], end[2]], color="#596373", lw=3)
            else:
                covariance = obj["R"] @ np.diag(obj["axes"] ** 2) @ obj["R"].T
                values, vectors = np.linalg.eigh(covariance[np.ix_([0, 2], [0, 2])])
                semiaxes = np.sqrt(values)
                tilt = np.rad2deg(np.arctan2(vectors[1, 0], vectors[0, 0]))
                ax.add_patch(
                    Ellipse(
                        (p[0], p[2]),
                        2 * semiaxes[0],
                        2 * semiaxes[1],
                        angle=tilt,
                        facecolor="#c3cbd4" if obj["name"] == "fuselage" else "#45505f",
                        edgecolor="#45505f",
                        alpha=0.85,
                    )
                )
        ax.scatter([0], [0], s=30, c="black", zorder=4)
        ax.text(-0.1, -0.2, "CG", ha="right", fontsize=9)
        ax.scatter(
            [mount[0]], [mount[2]], s=65, c="#c0392b", edgecolor="white", zorder=5
        )
        ax.annotate(
            "",
            xy=(mount[0] + 0.8 * np.cos(angle), mount[2] + 0.8 * np.sin(angle)),
            xytext=(mount[0], mount[2]),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#c0392b"},
            zorder=5,
        )
        ax.axhline(gear[2], color="#596373", ls="--", lw=1)
        ax.scatter([gear[0]], [gear[2]], marker="x", s=40, color="black", zorder=5)
        ax.set(
            xlim=(-7, 7),
            ylim=(4, -1.8),
            xlabel="Body x: forward (m)",
            ylabel="Body z: down (m)",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
    axes[0].set_title("Camera at forward belly surface", fontsize=11)
    axes[0].annotate(
        f"Optical centre {mount.tolist()} m",
        xy=(mount[0], mount[2]),
        xytext=(-0.6, -1.45),
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "#c0392b"},
    )
    axes[0].text(
        -6.6,
        3.55,
        f"Wheel-bottom centre: {gear.tolist()} m\nOptical down angle: {config['camera']['optical_down_deg']:g} degrees",
        fontsize=9,
    )
    half = np.deg2rad(config["camera"]["vertical_fov_deg"] / 2)
    height = gear[2] - mount[2]
    ends = [
        [mount[0] + height / np.tan(angle + half), gear[2]],
        [mount[0] + height / np.tan(angle - half), gear[2]],
    ]
    axes[1].add_patch(
        Polygon([mount[[0, 2]], *ends], facecolor="#64b5db", alpha=0.17, zorder=0)
    )
    span = config["marker"]["length"] / 2
    axes[1].plot([-span, span], [gear[2], gear[2]], lw=5, color="#248550")
    for x in [-span, 0, span]:
        axes[1].plot([mount[0], x], [mount[2], gear[2]], color="#248550", lw=1.2)
    axes[1].annotate(
        "",
        xy=(5.5, gear[2]),
        xytext=(5.5, mount[2]),
        arrowprops={"arrowstyle": "<->", "color": "#c0392b"},
    )
    axes[1].text(
        5.7, (gear[2] + mount[2]) / 2, f"{height:.2f} m", fontsize=9, color="#c0392b"
    )
    axes[1].set_title("Nominal touchdown and vertical FOV", fontsize=11)
    axes[1].text(
        -6.6,
        3.55,
        f"H span in side view: {2 * span:g} m\nRectified FOV: {config['camera']['horizontal_fov_deg']:g} / {config['camera']['vertical_fov_deg']:g} degrees",
        fontsize=9,
    )
    fig.suptitle(
        "Provisional belly camera layout: replace with measured UH-60 geometry\nSide projection only; main wheels are offset sideways. Assess occlusion with the 3D ray report.",
        fontsize=11,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/landing_dataset.yaml")
    parser.add_argument("--output", default="output/figure/landing_camera_mount.png")
    args = parser.parse_args()
    render_camera_mount(yaml.safe_load(Path(args.config).read_text()), args.output)


if __name__ == "__main__":
    main()
