"""Seeded RANSAC followed by total least squares; plane n.p+d=0 in camera frame."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Plane:
    normal: np.ndarray
    offset: float
    inliers: np.ndarray
    rms: float


def extract_deck_plane(points, threshold=0.08, iterations=150, min_inliers=30, seed=0):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("invalid point cloud")
    if len(p) < max(min_inliers, 3):
        return None
    rng = np.random.default_rng(seed)
    best = np.zeros(len(p), dtype=bool)
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        inside = np.abs(p @ n - a @ n) < threshold
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < min_inliers:
        return None
    for _ in range(2):
        q = p[best]
        center = q.mean(0)
        _, _, V = np.linalg.svd(q - center, full_matrices=False)
        n = V[-1]
        d = -float(n @ center)
        if d < 0:
            n, d = -n, -d
        best = np.abs(p @ n + d) < threshold
        if best.sum() < min_inliers:
            return None
    return Plane(n, d, best, float(np.sqrt(np.mean((p[best] @ n + d) ** 2))))
