"""H has four outer rectangle landmarks in fixed deck-frame order."""

import numpy as np


def marker_points(width=6.0, length=6.0, height=0.025):
    return np.array(
        [
            [-width / 2, -length / 2, height],
            [width / 2, -length / 2, height],
            [width / 2, length / 2, height],
            [-width / 2, length / 2, height],
        ]
    )


def marker_polygons(width=6.0, length=6.0, stroke=1.2, height=0.025):
    if not 0 < stroke < min(width, length) / 2:
        raise ValueError("invalid H stroke")
    boxes = [
        (-width / 2, -width / 2 + stroke, -length / 2, length / 2),
        (width / 2 - stroke, width / 2, -length / 2, length / 2),
        (-width / 2, width / 2, -stroke / 2, stroke / 2),
    ]
    return [
        np.array(
            [[x0, y0, height], [x1, y0, height], [x1, y1, height], [x0, y1, height]]
        )
        for x0, x1, y0, y1 in boxes
    ]


def build_h_marker(config=None, parent=None):
    import bpy

    c = config or {}
    objects = []
    mat = bpy.data.materials.get("H_white") or bpy.data.materials.new("H_white")
    mat.diffuse_color = (0.98, 0.98, 0.98, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.98, 0.98, 0.98, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    for i, p in enumerate(
        marker_polygons(c.get("width", 6.0), c.get("length", 6.0), c.get("stroke", 1.2))
    ):
        mesh = bpy.data.meshes.new(f"H_bar_{i}")
        mesh.from_pydata(p.tolist(), [], [(0, 1, 2, 3)])
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)
        obj.parent = parent
        obj.pass_index = 1
        objects.append(obj)
    return objects
