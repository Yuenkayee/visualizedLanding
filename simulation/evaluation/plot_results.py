import argparse
from pathlib import Path
import numpy as np
from simulation.offline.logger import read_log


def plot_results(log_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_log(log_path)
    if not rows:
        raise ValueError("empty log")
    t = np.array([r["timestamp"] for r in rows])
    truth = np.array([r["truth"]["T_deck_camera"] for r in rows])[:, :3, 3]
    estimate = np.array([r["estimate"]["T_deck_camera"] for r in rows])[:, :3, 3]
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for i, label in enumerate(["deck x [m]", "deck y [m]", "height [m]"]):
        axes[i].plot(t, truth[:, i], label="truth")
        axes[i].plot(t, estimate[:, i], "--", label="estimate")
        axes[i].set_ylabel(label)
        axes[i].grid(True)
    axes[0].legend()
    axes[3].plot(t, np.linalg.norm(estimate - truth, axis=1))
    axes[3].set_ylabel("position error [m]")
    axes[3].set_xlabel("simulation time [s]")
    axes[3].grid(True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--output", default="outputs/figures/closed_loop.png")
    a = p.parse_args()
    print(plot_results(a.log, a.output))


if __name__ == "__main__":
    main()
