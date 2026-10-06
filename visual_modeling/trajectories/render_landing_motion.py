"""Plot the continuous large-angle Euler oscillations saved by the landing plan."""

import argparse
import json
from pathlib import Path
import numpy as np


def render_landing_motion(sortie, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sortie["rows"]
    t = np.array([r["timestamp"] for r in rows])
    angles = np.array([r["attitude_oscillation_rad"] for r in rows])
    vibration = np.array([r["attitude_vibration_rad"] for r in rows])
    amplitudes = np.array([r["attitude_amplitude_rad"] for r in rows])
    boundaries = np.r_[0, np.cumsum(sortie["stage_durations_s"])]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout="constrained")
    colours = ["#d5e7f5", "#daf0dd", "#f5e3cf"]
    for axis, ax in enumerate(axes):
        for k, stage in enumerate(("Approach", "Window hold", "Descent")):
            ax.axvspan(
                boundaries[k], boundaries[k + 1], facecolor=colours[k], alpha=0.5
            )
            if axis == 0:
                ax.text(
                    (boundaries[k] + boundaries[k + 1]) / 2,
                    0.66,
                    stage,
                    ha="center",
                    fontsize=10,
                )
        ax.plot(
            t,
            angles[:, axis] + vibration[:, axis],
            color="#205b8f",
            lw=1,
            label="Euler offset incl. vibration",
        )
        ax.plot(
            t,
            amplitudes[:, axis],
            color="#c0392b",
            ls="--",
            lw=1,
            label="Single-sided harmonic envelope",
        )
        ax.plot(t, -amplitudes[:, axis], color="#c0392b", ls="--", lw=1)
        ax.set(
            ylabel=["phi / roll (rad)", "theta / pitch (rad)", "psi / yaw (rad)"][axis],
            ylim=(-0.7, 0.72),
            xlim=(0, t[-1]),
        )
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=9, ncol=2)
    axes[-1].set_xlabel("Time (s)")
    periods = ", ".join(f"{p:.2f}" for p in sortie["attitude_model"]["period_s"])
    fig.suptitle(
        f"Stage-dependent helicopter Euler offsets: 0.60 → 0.05 → 0.30 rad\nPeriods [phi, theta, psi]: {periods} s; continuous phase, fixed body camera",
        fontsize=12,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="data/landing/plan.json")
    parser.add_argument(
        "--output", default="output/figure/landing_attitude_profile.png"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    sortie = json.loads(Path(plan["sorties"][0]["trajectory"]).read_text())
    render_landing_motion(sortie, args.output)


if __name__ == "__main__":
    main()
