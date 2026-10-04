import json
from pathlib import Path
import numpy as np
from navigation.pose.solve_pnp import project_points
from navigation.common.frames import transform_points


def make_annotation(
    timestamp,
    sequence_id,
    frame_id,
    image,
    mask,
    points,
    T_deck_camera,
    calibration,
    visibility=None,
):
    pixels = project_points(points, T_deck_camera, calibration)
    Xc = transform_points(np.linalg.inv(T_deck_camera), points)
    visible = (
        (Xc[:, 2] > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < calibration.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < calibration.height)
    )
    if visibility is not None:
        visible &= np.asarray(visibility, dtype=bool)
    return dict(
        schema_version=1,
        timestamp=float(timestamp),
        sequence_id=str(sequence_id),
        frame_id=int(frame_id),
        image=str(image),
        mask=str(mask),
        keypoints=pixels.tolist(),
        visibility=visible.tolist(),
        points_deck=np.asarray(points).tolist(),
        T_deck_camera=np.asarray(T_deck_camera).tolist(),
        K=calibration.K.tolist(),
        distortion=calibration.distortion.tolist(),
        width=calibration.width,
        height=calibration.height,
    )


def export_annotations(path, annotation):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(annotation, indent=2, allow_nan=False) + "\n")


def keypoint_visibility(points, ship, camera):
    import bpy
    from mathutils import Vector

    graph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation
    visible = []
    for p in points:
        # Cast slightly inside painted bars: exact outer mesh vertices may miss
        # because of triangle boundary floating-point precision.
        sample = np.array(p, dtype=float).copy()
        sample[:2] -= np.sign(sample[:2]) * 0.02
        target = ship.matrix_world @ Vector(sample)
        direction = target - origin
        hit, loc, _, _, obj, _ = bpy.context.scene.ray_cast(
            graph, origin, direction.normalized(), distance=direction.length + 0.01
        )
        visible.append(
            bool(hit and obj.pass_index == 1 and (loc - target).length < 0.15)
        )
    return visible
