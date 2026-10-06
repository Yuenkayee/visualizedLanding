"""Entry point for Blender's embedded interpreter."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if os.environ.get("LANDING_SITE_PACKAGES"):
    sys.path.append(os.environ["LANDING_SITE_PACKAGES"])
from visual_training.data.generate_landing_dataset import main

if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
