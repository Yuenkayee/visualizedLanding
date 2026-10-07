"""Entry point for Blender's embedded interpreter."""

import os
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if os.environ.get("LANDING_SITE_PACKAGES"):
    sys.path.append(os.environ["LANDING_SITE_PACKAGES"])
if __name__ == "__main__":
    main = importlib.import_module(
        os.environ.get(
            "LANDING_DATASET_MODULE", "visual_training.data.generate_landing_dataset"
        )
    ).main
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
