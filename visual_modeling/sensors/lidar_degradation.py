import numpy as np


def degrade_lidar(
    points, rng=None, range_std=0.02, dropout=0.05, min_range=0.2, max_range=150.0
):
    if not 0 <= dropout < 1 or range_std < 0 or min_range < 0 or max_range <= min_range:
        raise ValueError("invalid lidar settings")
    rng = rng or np.random.default_rng()
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must be Nx3")
    r = np.linalg.norm(p, axis=1)
    keep = (r >= min_range) & (r <= max_range) & (rng.random(len(p)) >= dropout)
    p = p[keep]
    r = r[keep]
    noisy = r + rng.normal(0, range_std, len(r))
    valid = (noisy >= min_range) & (noisy <= max_range)
    return p[valid] * (noisy[valid] / r[valid])[:, None]
