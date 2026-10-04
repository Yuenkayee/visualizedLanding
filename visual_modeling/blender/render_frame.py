"""RGB render and geometry-visible binary H mask using an emission label pass."""

from pathlib import Path


def render_frame(image_path, mask_path=None, samples=32):
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.render.film_transparent = False
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(Path(image_path).resolve())
    bpy.ops.render.render(write_still=True)
    if mask_path is None:
        return

    def emission(name, value):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        node = nodes.new("ShaderNodeEmission")
        node.inputs["Color"].default_value = (value, value, value, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(node.outputs[0], output.inputs["Surface"])
        return mat

    black = emission("LabelBlack", 0)
    white = emission("LabelWhite", 1)
    saved = []
    hidden = []
    bg = scene.world.node_tree.nodes.get("Background") if scene.world else None
    strength = bg.inputs["Strength"].default_value if bg else None
    old_samples = scene.cycles.samples
    old_path = scene.render.filepath
    old_view = scene.view_settings.view_transform
    try:
        if bg:
            bg.inputs["Strength"].default_value = 0
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            if obj.name.startswith("FogVolume"):
                hidden.append((obj, obj.hide_render))
                obj.hide_render = True
                continue
            saved.append((obj, list(obj.data.materials)))
            obj.data.materials.clear()
            obj.data.materials.append(white if obj.pass_index == 1 else black)
        scene.cycles.samples = 1
        scene.view_settings.view_transform = "Standard"
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(Path(mask_path).resolve())
        bpy.ops.render.render(write_still=True)
    finally:
        for obj, mats in saved:
            obj.data.materials.clear()
            for mat in mats:
                obj.data.materials.append(mat)
        for obj, hidden_value in hidden:
            obj.hide_render = hidden_value
        if bg:
            bg.inputs["Strength"].default_value = strength
        scene.cycles.samples = old_samples
        scene.render.filepath = old_path
        scene.view_settings.view_transform = old_view
        bpy.data.materials.remove(black)
        bpy.data.materials.remove(white)
