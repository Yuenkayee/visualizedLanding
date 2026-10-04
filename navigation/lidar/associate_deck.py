import numpy as np
from navigation.common.frames import transform_points


def associate_deck(
    points_camera, T_deck_camera, bounds=(12.0, 8.0), height_tolerance=2.0
):
    p = transform_points(T_deck_camera, points_camera)
    return (
        (np.abs(p[:, 0]) <= bounds[0])
        & (np.abs(p[:, 1]) <= bounds[1])
        & (np.abs(p[:, 2]) <= height_tolerance)
    )


def plane_matches_deck(
    plane, T_deck_camera, max_angle=np.deg2rad(25), max_height_error=5.0
):
    expected = T_deck_camera[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return (
        float(plane.normal @ expected) >= np.cos(max_angle)
        and abs(plane.offset - T_deck_camera[2, 3]) <= max_height_error
    )
