import argparse
import copy
import json
from pathlib import Path
import yaml
from simulation.offline.run_closed_loop import run_closed_loop


def run_batch(config, seeds=(1, 2, 3), output="outputs/metrics/batch"):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        c = copy.deepcopy(config)
        c.update(
            seed=int(seed),
            log=str(path / f"seed_{seed}.jsonl"),
            metrics=str(path / f"seed_{seed}.json"),
        )
        results.append(dict(seed=int(seed), **run_closed_loop(c)))
    (path / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/offline_simulation.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--output", default="outputs/metrics/batch")
    a = p.parse_args()
    print(
        json.dumps(
            run_batch(yaml.safe_load(Path(a.config).read_text()), a.seeds, a.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
