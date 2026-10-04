"""Original procedural frigate; optional licensed GLB/OBJ/Blend visual hull import.
Deck origin is the H centre; bow is +X, port +Y, up +Z. VRX references
inform the asset/physics separation; no proprietary game mesh is redistributed.
"""

from pathlib import Path


def material(name, color, metallic=0.0, roughness=0.7):
    import bpy

    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1.0)
    m.use_nodes = True
    node = m.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    return m


def box(name, location, dimensions, mat, parent=None, bevel=0.0):
    import bpy

    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    obj.parent = parent
    if bevel:
        mod = obj.modifiers.new("edge_bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    return obj


def import_ship(path, parent, length=140.0):
    import bpy
    from mathutils import Vector

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = set(bpy.data.objects)
    if path.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif path.suffix.lower() == ".blend":
        with bpy.data.libraries.load(str(path), link=False) as (src, dst):
            dst.objects = src.objects
        for obj in dst.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError("ship asset must be .glb, .gltf, .obj or .blend")
    objects = list(set(bpy.data.objects) - before)
    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        raise ValueError("asset contains no mesh")
    bounds = [o.matrix_world @ Vector(v) for o in meshes for v in o.bound_box]
    xmin, xmax = min(v.x for v in bounds), max(v.x for v in bounds)
    if xmax - xmin < 1e-5:
        raise ValueError("asset X axis must point along ship length")
    root = bpy.data.objects.new("ImportedHull", None)
    bpy.context.collection.objects.link(root)
    # Preserve imported hierarchies and world transforms before normalising.
    for obj in objects:
        if obj.parent not in objects:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
    scale = length / (xmax - xmin)
    root.scale = (scale,) * 3
    root.location = (
        -26 - xmin * scale,
        -(max(v.y for v in bounds) + min(v.y for v in bounds)) * 0.5 * scale,
        -6 - min(v.z for v in bounds) * scale,
    )
    root.parent = parent
    for obj in objects:
        if obj.type in ("CAMERA", "LIGHT"):
            obj.hide_render = True
    return root


def build_ship(config=None):
    import bpy
    from visual_modeling.blender.build_h_marker import build_h_marker

    c = config or {}
    if c.get("length", 140.0) <= 0:
        raise ValueError("ship length must be positive")
    root = bpy.data.objects.new("DeckFrame", None)
    bpy.context.collection.objects.link(root)
    steel = material("naval_grey", (0.24, 0.29, 0.32), 0.3)
    dark = material("deck", (0.065, 0.09, 0.09))
    yellow = material("deck_yellow", (0.9, 0.63, 0.05))
    if c.get("asset_path"):
        import_ship(c["asset_path"], root, c.get("length", 140.0))
    else:
        # Hull stations: stern, parallel body, tapered bow. Closed manifold rings.
        stations = [
            (-26, 7, 0),
            (-20, 8, 0),
            (50, 8, 0),
            (85, 7, 0),
            (105, 4, 1),
            (114, 0.15, 2),
        ]
        verts = []
        for x, b, z in stations:
            verts.extend([(x, -b, z), (x, b, z), (x, b * 0.7, -5), (x, -b * 0.7, -5)])
        faces = [(3, 2, 1, 0)]
        for i in range(len(stations) - 1):
            for j in range(4):
                faces.append(
                    (
                        i * 4 + j,
                        i * 4 + (j + 1) % 4,
                        (i + 1) * 4 + (j + 1) % 4,
                        (i + 1) * 4 + j,
                    )
                )
        faces.append(tuple(range((len(stations) - 1) * 4, len(stations) * 4)))
        faces = [tuple(reversed(face)) for face in faces]
        mesh = bpy.data.meshes.new("frigate_hull")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("FrigateHull", mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = root
        mesh.materials.append(steel)
        box("Hangar", (24, 0, 4), (23, 13, 8), steel, root, 0.6)
        box("Superstructure", (54, 0, 5.5), (32, 12, 11), steel, root, 1.0)
        box("Bridge", (65, 0, 12), (15, 10, 4), steel, root, 0.7)
        glass = material("bridge_windows", (0.015, 0.055, 0.075), 0.5, 0.15)
        for y in (-5.06, 5.06):
            box("BridgeWindows", (65, y, 12.5), (12, 0.08, 1.4), glass, root)
        box("Mast", (52, 0, 19), (2, 2, 17), steel, root)
        box("Radar", (52, 0, 27), (7, 0.8, 2), steel, root)
        box("Funnel", (40, 0, 12), (5, 5, 9), steel, root, 0.4)
        box("BowTurret", (93, 0, 3), (7, 6, 5), steel, root, 1.0)
        box("Barrel", (99, 0, 4), (8, 0.5, 0.5), steel, root)
        length_scale = c.get("length", 140.0) / 140.0
        for obj in root.children:
            obj.location.x *= length_scale
            obj.scale.x *= length_scale
    box("Helideck", (0, 0, -0.12), (24, 14, 0.24), dark, root)
    # Yellow landing circle and edge lines leave the white H separable.
    bpy.ops.mesh.primitive_torus_add(
        major_radius=5.2,
        minor_radius=0.055,
        major_segments=96,
        minor_segments=6,
        location=(0, 0, 0.02),
    )
    ring = bpy.context.object
    ring.name = "LandingCircle"
    ring.data.materials.append(yellow)
    ring.parent = root
    for y in (-6.5, 6.5):
        box("DeckEdge", (0, y, 0.015), (22, 0.1, 0.015), yellow, root)
    for x in (-11, 11):
        box("DeckEdge", (x, 0, 0.015), (0.1, 13, 0.015), yellow, root)
    build_h_marker(c.get("marker"), root)
    return root
