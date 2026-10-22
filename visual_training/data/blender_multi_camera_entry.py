"""Blender-only entry; never import the training environment's site-packages."""

import argparse
import os
from pathlib import Path
import sys


def main():
    sys.path.insert(0, os.environ["LANDING_RENDER_SITE"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cycles-device", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--setup-only", action="store_true")
    args, remaining = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
    # Import all native dependencies here as well as in the standalone Python check.
    import numpy  # noqa: F401
    import scipy.spatial.transform  # noqa: F401
    import cv2  # noqa: F401
    import yaml  # noqa: F401
    from visual_modeling.blender.cycles_device import smoke_render

    runtime = smoke_render(args.cycles_device, args.gpu_index)
    if not args.setup_only:
        from visual_training.data.generate_multi_camera_dataset import main as generate

        generate(remaining, render_runtime=runtime)


if __name__ == "__main__":
    main()
