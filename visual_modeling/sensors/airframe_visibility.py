"""Shared ellipsoid airframe proxy for mounting checks, rays and Blender meshes.

Dimensions are configurable design proxies, not measured UH-60 geometry.
All coordinates use the aircraft forward/right/down body frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def airframe_primitives(geometry):
    proxy = geometry.get("airframe_proxy", {})
    primitives = []
    for item in proxy.get("ellipsoids", []):
        centre = np.asarray(item["centre_body_m"], float)
        axes = np.asarray(item["semiaxes_m"], float)
        rotation = Rotation.from_euler(
            "xyz", item.get("euler_deg", [0, 0, 0]), degrees=True
        ).as_matrix()
        if (
            centre.shape != (3,)
            or axes.shape != (3,)
            or not np.isfinite(centre).all()
            or not np.isfinite(axes).all()
            or (axes <= 0).any()
        ):
            raise ValueError("invalid airframe ellipsoid centre/semiaxes")
        primitives.append(dict(name=item["name"], centre=centre, axes=axes, R=rotation))
    for item in proxy.get("struts", []):
        start = np.asarray(item["start_body_m"], float)
        end = np.asarray(item["end_body_m"], float)
        radius = float(item["radius_m"])
        length = np.linalg.norm(end - start)
        if (
            start.shape != (3,)
            or end.shape != (3,)
            or not np.isfinite([*start, *end, radius]).all()
            or length <= 0
            or radius <= 0
        ):
            raise ValueError("invalid airframe strut")
        z = (end - start) / length
        helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        rotation = np.column_stack([x, np.cross(z, x), z])
        primitives.append(
            dict(
                name=item["name"],
                centre=(start + end) / 2,
                axes=np.array([radius, radius, length / 2 + radius]),
                R=rotation,
            )
        )
    return primitives


def validate_external_camera(position, primitives):
    position = np.asarray(position, float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("camera body position must be a finite 3-vector")
    for primitive in primitives:
        q = (position - primitive["centre"]) @ primitive["R"] / primitive["axes"]
        if np.dot(q, q) <= 1:
            raise ValueError(f"camera optical centre is inside/on {primitive['name']}")


def airframe_occlusion(origin, targets, primitives):
    """Return rays blocked BEFORE each target, including an origin inside a solid."""
    origin = np.asarray(origin, float)
    targets = np.atleast_2d(np.asarray(targets, float))
    blocked = np.zeros(len(targets), dtype=bool)
    for primitive in primitives:
        q = (origin - primitive["centre"]) @ primitive["R"] / primitive["axes"]
        d = (targets - origin) @ primitive["R"] / primitive["axes"]
        a = np.einsum("ij,ij->i", d, d)
        b = 2 * (d @ q)
        c = np.dot(q, q) - 1
        if c <= 0:
            return np.ones(len(targets), dtype=bool)
        discriminant = b * b - 4 * a * c
        root = np.sqrt(np.maximum(discriminant, 0))
        near = (-b - root) / (2 * np.maximum(a, 1e-20))
        far = (-b + root) / (2 * np.maximum(a, 1e-20))
        blocked |= (discriminant >= 0) & (a > 1e-20) & (far > 1e-6) & (near < 1 - 1e-6)
    return blocked


def projected_airframe_mask(calibration, T_deck_camera, T_body_camera, primitives):
    """CPU smoke silhouette: trace rays to the marker plane, including occlusion."""
    v, u = np.indices((calibration.height, calibration.width))
    pixels = np.column_stack([u.ravel(), v.ravel(), np.ones(u.size)])
    rays = pixels @ np.linalg.inv(calibration.K).T
    pose = np.asarray(T_deck_camera)
    denominator = rays @ pose[2, :3]
    forward = np.abs(denominator) > 1e-10
    distance = np.zeros(len(rays))
    distance[forward] = (0.025 - pose[2, 3]) / denominator[forward]
    forward &= distance > 0
    mount = np.asarray(T_body_camera)
    targets = (rays * distance[:, None]) @ mount[:3, :3].T + mount[:3, 3]
    blocked = airframe_occlusion(mount[:3, 3], targets, primitives) & forward
    return blocked.reshape(calibration.height, calibration.width)
