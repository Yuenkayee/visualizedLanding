"""Persistent isolated bpy process; stdout protocol is distinct from Blender logs."""

import json
import sys
import traceback
import numpy as np
from main.sensors import SensorGenerator, PREFIX
from simulation.bridge.exchange_contracts import SensorTruth


def reply(value):
    print(PREFIX + json.dumps(value, allow_nan=False), flush=True)


def main():
    try:
        c = json.loads(sys.stdin.readline())
        generator = SensorGenerator(
            c["config"], c["landing"], c["lidar"], c["directory"]
        )
        reply(generator.metadata)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("stop"):
                break
            result = generator.observe(
                SensorTruth.from_dict(request["truth"]),
                request["camera"],
                request["lidar"],
                request["index"],
            )
            np.savez_compressed(request["output"], **result)
            reply(dict(timestamp=float(result["timestamp"])))
    except BaseException as exc:
        traceback.print_exc()
        reply(dict(error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
