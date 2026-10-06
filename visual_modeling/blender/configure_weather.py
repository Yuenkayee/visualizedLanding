def configure_weather(config=None):
    import bpy
    import math

    c = config or {}
    scene = bpy.context.scene
    if scene.world:
        bg = scene.world.node_tree.nodes.get("Background")
        bg.inputs["Strength"].default_value = c.get("sky_strength", 0.3)
    sun = bpy.data.objects.get("Sun")
    if sun:
        sun.data.energy = c.get("sun_energy", 3.0)
        sun.rotation_euler = (
            math.radians(c.get("sun_pitch", 25)),
            0,
            math.radians(c.get("sun_yaw", -30)),
        )
    if c.get("fog_density", 0) > 0:
        from visual_modeling.blender.build_ship import box

        m = bpy.data.materials.new("fog")
        m.use_nodes = True
        nodes = m.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        fog = nodes.new("ShaderNodeVolumePrincipled")
        fog.inputs["Density"].default_value = c["fog_density"]
        m.node_tree.links.new(fog.outputs["Volume"], output.inputs["Volume"])
        box("FogVolume", (0, 0, 100), (600, 600, 250), m)
