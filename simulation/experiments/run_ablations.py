import argparse
import copy
import json
from pathlib import Path
import yaml
from simulation.offline.run_closed_loop import run_closed_loop


def run_ablations(config, output="outputs/metrics/ablations"):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, changes in [
        ("vision_lidar", {}),
        ("vision_only", {"disable_lidar": True}),
        ("lidar_only", {"disable_vision": True}),
        ("prediction_only", {"disable_vision": True, "disable_lidar": True}),
    ]:
        c = copy.deepcopy(config)
        c.update(changes)
        c.update(log=str(path / f"{name}.jsonl"), metrics=str(path / f"{name}.json"))
        results[name] = run_closed_loop(c)
    (path / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/offline_simulation.yaml")
    p.add_argument("--output", default="outputs/metrics/ablations")
    a = p.parse_args()
    print(
        json.dumps(
            run_ablations(yaml.safe_load(Path(a.config).read_text()), a.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
