import math
from visual_modeling.blender.build_ship import material


def build_sea_sky(config=None):
    import bpy

    c = config or {}
    size = c.get("size", 1000.0)
    n = c.get("grid", 80)
    amp = c.get("wave_amplitude", 0.3)
    verts = []
    for i in range(n + 1):
        for j in range(n + 1):
            x = (i / n - 0.5) * size
            y = (j / n - 0.5) * size
            z = -6 + amp * (
                math.sin(x * 0.12 + y * 0.04) + 0.5 * math.sin(y * 0.2 - x * 0.03)
            )
            verts.append((x, y, z))
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    mesh = bpy.data.meshes.new("OceanMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Ocean", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(material("ocean", (0.015, 0.08, 0.12), 0.25, 0.22))
    for p in mesh.polygons:
        p.use_smooth = True
    world = bpy.data.worlds.new("MarineSky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(c.get("sun_elevation", 30.0))
    world.node_tree.links.new(
        sky.outputs["Color"], nodes.get("Background").inputs["Color"]
    )
    nodes.get("Background").inputs["Strength"].default_value = 0.3
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 100))
    sun = bpy.context.object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (0.4, -0.5, -0.5)
    return obj
