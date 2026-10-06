"""Render the same provisional fuselage/wheel/strut solids used by ray checks."""

from visual_modeling.sensors.airframe_visibility import airframe_primitives


def build_airframe_proxy(geometry):
    import bpy
    from mathutils import Matrix
    from visual_modeling.blender.build_ship import material

    root = bpy.data.objects.new("HelicopterBodyFrame", None)
    bpy.context.collection.objects.link(root)
    skin = material("AircraftProxySkin", (0.08, 0.11, 0.08), roughness=0.8)
    rubber = material("AircraftProxyRubber", (0.015, 0.018, 0.018), roughness=0.95)
    for primitive in airframe_primitives(geometry):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1)
        obj = bpy.context.object
        obj.name = "AircraftProxy_" + primitive["name"]
        obj.parent = root
        obj.location = primitive["centre"].tolist()
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Matrix(primitive["R"].tolist()).to_quaternion()
        obj.scale = primitive["axes"].tolist()
        obj.data.materials.append(rubber if "wheel" in primitive["name"] else skin)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return root
