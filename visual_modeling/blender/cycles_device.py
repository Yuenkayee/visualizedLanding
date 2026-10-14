"""Explicit Cycles device selection; a requested GPU never falls back to CPU."""


def configure_cycles(backend="OPTIX", threads=8, device_index=0):
    import bpy

    backend = backend.upper()
    if backend not in ("OPTIX", "CUDA", "CPU") or threads < 1 or device_index < 0:
        raise ValueError("invalid Cycles backend, thread count or device index")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.use_persistent_data = True
    # Preserve RGB denoising; sensor noise is applied afterwards. The emission
    # mask pass temporarily disables denoising to preserve label boundaries.
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = backend != "CPU"
    if backend == "CPU":
        scene.cycles.device = "CPU"
        return dict(
            backend=backend,
            devices=["CPU"],
            threads=threads,
            blender=bpy.app.version_string,
        )
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = backend
        prefs.refresh_devices()
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(f"Cycles {backend} unavailable: {exc}") from exc
    devices = [d for d in prefs.devices if d.type == backend]
    if device_index >= len(devices):
        raise RuntimeError(
            f"Cycles {backend} GPU index {device_index} unavailable; found "
            f"{[(d.name, d.type) for d in prefs.devices]}. "
            "Check NVIDIA driver/container GPU and graphics library access."
        )
    selected = devices[device_index]
    for device in prefs.devices:
        device.use = device == selected
    scene.cycles.device = "GPU"
    return dict(
        backend=backend,
        devices=[selected.name],
        threads=threads,
        blender=bpy.app.version_string,
    )


def probe(backend, threads, device_index):
    """Compile/use the GPU kernels before spending time planning trajectories."""
    import bpy

    result = configure_cycles(backend, threads, device_index)
    bpy.ops.wm.read_factory_settings(use_empty=False)
    configure_cycles(backend, threads, device_index)
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 1
    bpy.ops.render.render()
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["OPTIX", "CUDA", "CPU"], default="OPTIX")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(probe(args.device, args.threads, args.gpu_index)), flush=True)
