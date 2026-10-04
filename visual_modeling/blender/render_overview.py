"""Render a review image of the entire frigate, separate from training cameras."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml


def main(argv=None):
    import bpy
    from mathutils import Vector
    from visual_modeling.blender.build_ship import build_ship
    from visual_modeling.blender.build_sea_sky import build_sea_sky
    from visual_modeling.blender.render_frame import render_frame

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/scene.yaml"))
    p.add_argument(
        "--output", default=str(ROOT / "outputs/figures/frigate_overview.png")
    )
    a = p.parse_args(argv)
    c = yaml.safe_load(Path(a.config).read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    build_ship(c.get("ship"))
    build_sea_sky(c.get("sea"))
    bpy.ops.object.camera_add(location=(-105, -140, 120))
    camera = bpy.context.object
    camera.rotation_euler = (
        (Vector((38, 0, 3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 180.0
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = 1200
    bpy.context.scene.render.resolution_y = 800
    bpy.context.scene.render.resolution_percentage = 100
    render_frame(a.output, samples=24)
    print(a.output)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
