"""CPU analytic deck ray caster plus Blender scene ray caster."""

import numpy as np


def beam_directions(horizontal_fov=50.0, vertical_fov=40.0, columns=64, rows=32):
    x = np.tan(
        np.deg2rad(np.linspace(-horizontal_fov / 2, horizontal_fov / 2, columns))
    )
    y = np.tan(np.deg2rad(np.linspace(-vertical_fov / 2, vertical_fov / 2, rows)))
    xx, yy = np.meshgrid(x, y)
    d = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
    return d / np.linalg.norm(d, axis=1)[:, None]


def raycast_deck(
    T_deck_lidar, directions=None, half_extent=(12.0, 7.0), max_range=150.0
):
    directions = (
        beam_directions() if directions is None else np.asarray(directions, float)
    )
    world_d = directions @ T_deck_lidar[:3, :3].T
    origin = T_deck_lidar[:3, 3]
    denominator = world_d[:, 2]
    distance = np.divide(
        -origin[2],
        denominator,
        out=np.full(len(directions), np.inf),
        where=np.abs(denominator) > 1e-9,
    )
    hits = origin + world_d * distance[:, None]
    valid = (
        (distance > 0)
        & (distance < max_range)
        & (np.abs(hits[:, 0]) < half_extent[0])
        & (np.abs(hits[:, 1]) < half_extent[1])
    )
    return directions[valid] * distance[valid, None]


def raycast_scene(T_world_lidar, directions=None, max_range=150.0):
    import bpy
    from mathutils import Vector

    directions = beam_directions() if directions is None else directions
    graph = bpy.context.evaluated_depsgraph_get()
    points = []
    origin = Vector(T_world_lidar[:3, 3].tolist())
    for d in directions:
        direction = Vector((T_world_lidar[:3, :3] @ d).tolist())
        hit, loc, *_ = bpy.context.scene.ray_cast(
            graph, origin, direction, distance=max_range
        )
        if hit:
            points.append(
                (T_world_lidar[:3, :3].T @ (np.array(loc) - np.array(origin))).tolist()
            )
    return np.asarray(points, float).reshape(-1, 3)
