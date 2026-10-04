"""H symmetry needs a heading prior; enumerate rectangle dihedral permutations."""

import numpy as np
from navigation.pose.solve_pnp import solve_pnp
from navigation.pose.resolve_planar_ambiguity import resolve_planar_ambiguity


def associate_keypoints(points_deck, pixels, calibration, prior=None, ordered=False):
    pixels = np.asarray(pixels, dtype=float)
    candidates = []
    permutations = (
        [pixels]
        if ordered
        else [
            np.roll(pixels[::-1] if flip else pixels, k, axis=0)
            for flip in (False, True)
            for k in range(4)
        ]
    )
    for u in permutations:
        for T, e in solve_pnp(points_deck, u, calibration):
            candidates.append((T, e, u))
    best = resolve_planar_ambiguity([(T, e) for T, e, _ in candidates], prior)
    if best is None:
        return None
    return next((T, e, u) for T, e, u in candidates if T is best[0])
