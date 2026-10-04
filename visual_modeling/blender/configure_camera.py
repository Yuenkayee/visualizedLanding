"""Pinhole intrinsics: Blender local camera -Z/Y becomes OpenCV +Z/-Y."""


def configure_camera(config):
    import bpy
    import numpy as np

    if np.any(config.get("distortion", [0] * 5)):
        raise ValueError(
            "Blender renderer uses zero-distortion intrinsics; degrade/remap separately"
        )
    fx, fy = config["fx"], config["fy"]
    scene = bpy.context.scene
    scene.render.resolution_x = config["width"]
    scene.render.resolution_y = config["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "NavigationCamera"
    data = camera.data
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = fx * data.sensor_width / config["width"]
    data.shift_x = (config["width"] / 2 - config["cx"]) / config["width"]
    data.shift_y = (config["cy"] - config["height"] / 2) * (fx / fy) / config["width"]
    data.clip_start = 0.05
    data.clip_end = 3000.0
    scene.camera = camera
    return camera
