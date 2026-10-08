"""Select an explicit Cycles device and validate it with an actual render."""

import json
from pathlib import Path
import tempfile


def configure_cycles(backend="OPTIX", gpu_index=0):
    import bpy

    backend = backend.upper()
    if backend not in ("OPTIX", "CUDA", "CPU", "METAL"):
        raise ValueError("Cycles backend must be OPTIX, CUDA, CPU or METAL")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if backend == "CPU":
        scene.cycles.device = "CPU"
        devices = ["CPU"]
    else:
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
        except (TypeError, RuntimeError) as error:
            raise RuntimeError(f"Cycles {backend} unavailable: {error}") from error
        available = [d for d in prefs.devices if d.type == backend]
        if gpu_index < 0 or gpu_index >= len(available):
            found = [(d.name, d.type) for d in prefs.devices]
            raise RuntimeError(
                f"Cycles {backend} GPU index {gpu_index} unavailable; devices={found}. "
                "Check the Blender/NVIDIA driver combination. No CPU fallback was used."
            )
        selected = available[gpu_index]
        for device in prefs.devices:
            device.use = device.id == selected.id and device.type == backend
        scene.cycles.device = "GPU"
        devices = [selected.name]
    report = dict(backend=backend, devices=devices, blender=bpy.app.version_string)
    print(json.dumps(dict(event="cycles_device", **report)), flush=True)
    return report


def smoke_render(backend="OPTIX", gpu_index=0):
    """Compile GPU kernels and render a tiny scene before planning 60 sorties."""
    import bpy
    from mathutils import Vector

    report = configure_cycles(backend, gpu_index)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(4, -4, 3))
    camera = bpy.context.object
    camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(2, -2, 4))
    light = bpy.context.object
    light.rotation_euler = (
        (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
    )
    light.data.energy = 600
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 1
    denoising = scene.cycles.use_denoising
    scene.cycles.use_denoising = False
    scene.render.image_settings.file_format = "PNG"
    try:
        with tempfile.TemporaryDirectory(prefix="landing-cycles-") as folder:
            path = Path(folder) / "probe.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("Cycles probe did not produce an image")
    finally:
        scene.cycles.use_denoising = denoising
    # The dataset rebuilds geometry, lighting and camera; keep device settings.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    print(json.dumps(dict(event="cycles_smoke_ok", **report)), flush=True)
    return report
