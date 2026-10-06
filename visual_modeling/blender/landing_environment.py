"""Moving waves/wake, local fog and illuminated deck for landing datasets."""

import numpy as np


def prepare_weather(weather, ship, camera):
    import bpy
    from visual_modeling.blender.configure_weather import configure_weather

    # Fog enclosure follows camera per frame to cover all absolute ship travel.
    configure_weather(
        dict(
            sky_strength=weather["sky_strength"],
            sun_energy=weather["sun_energy"],
            fog_density=weather["fog_density"],
            sun_pitch=90 - weather["sun_elevation_deg"],
        )
    )
    sky = next(
        (n for n in bpy.context.scene.world.node_tree.nodes if n.type == "TEX_SKY"),
        None,
    )
    if sky:
        sky.sun_elevation = np.deg2rad(weather["sun_elevation_deg"])
    if weather.get("deck_lights"):
        # Real lights provide geometry-dependent illumination; no fake bright H.
        for location in [(7, -5, 4), (7, 5, 4), (-7, -5, 3), (-7, 5, 3)]:
            bpy.ops.object.light_add(type="AREA", location=location)
            light = bpy.context.object
            light.name = "DeckFloodlight"
            light.parent = ship
            light.data.energy = 250.0
            light.data.shape = "DISK"
            light.data.size = 3.0
        from visual_modeling.blender.build_ship import material

        lamp = material("DeckLamp", (0.2, 0.8, 0.3))
        bsdf = lamp.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Emission Color"].default_value = (0.15, 0.8, 0.25, 1)
        bsdf.inputs["Emission Strength"].default_value = 8
        for x in [-10, -5, 0, 5, 10]:
            for y in [-6.6, 6.6]:
                bpy.ops.mesh.primitive_uv_sphere_add(
                    segments=8, ring_count=4, radius=0.08, location=(x, y, 0.1)
                )
                obj = bpy.context.object
                obj.name = "DeckLamp"
                obj.parent = ship
                obj.data.materials.append(lamp)


def make_visual_wake(ship):
    import bpy
    from visual_modeling.blender.build_ship import material

    mesh = bpy.data.meshes.new("WakeMesh")
    # Narrow wash astern, growing wider further behind; independent of airwake.
    vertices = []
    faces = []
    for i in range(41):
        x = -27 - i * 2.5
        half = 1.5 + i * 0.12
        vertices.extend([(x, -half, -5.65), (x, half, -5.65)])
        if i:
            faces.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("SurfaceWake", mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = ship
    mesh.materials.append(material("WakeFoam", (0.32, 0.48, 0.50), roughness=0.9))
    return obj


def update_environment(ocean, ship, camera, timestamp, sea_config):
    import bpy

    # Re-centre finite ocean patch while evaluating travelling waves in local
    # world coordinates. Absolute vessel translation remains in saved poses.
    ocean.location.x = ship.matrix_world.translation.x
    ocean.location.y = ship.matrix_world.translation.y
    amplitude = sea_config.get("wave_amplitude_m", 0.4)
    for v in ocean.data.vertices:
        x, y = v.co.x + ocean.location.x, v.co.y + ocean.location.y
        v.co.z = sea_config.get("sea_level_m", -6) + amplitude * (
            np.sin(0.12 * x + 0.04 * y - 0.9 * timestamp)
            + 0.5 * np.sin(0.2 * y - 0.03 * x - 1.2 * timestamp)
        )
    ocean.data.update()
    fog = bpy.data.objects.get("FogVolume")
    if fog:
        fog.location = camera.matrix_world.translation
