import numpy as np
from scipy.spatial.transform import Rotation


def resolve_planar_ambiguity(
    candidates, prior=None, position_scale=10.0, angle_scale=0.5
):
    if not candidates:
        return None

    def score(item):
        T, e = item
        if prior is None:
            return e
        angle = np.linalg.norm(
            Rotation.from_matrix(prior[:3, :3].T @ T[:3, :3]).as_rotvec()
        )
        return (
            e
            + np.linalg.norm(T[:3, 3] - prior[:3, 3]) / position_scale
            + angle / angle_scale
        )

    return min(candidates, key=score)
