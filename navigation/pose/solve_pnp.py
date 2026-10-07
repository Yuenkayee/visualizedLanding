"""Planar IPPE solutions are camera-in-deck poses, not OpenCV extrinsics."""

import cv2
import numpy as np
from navigation.common.frames import invert, transform, transform_points


def project_points(points_deck, T_deck_camera, calibration):
    T = invert(T_deck_camera)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    pixels, _ = cv2.projectPoints(
        np.asarray(points_deck, dtype=float),
        rvec,
        T[:3, 3],
        calibration.K,
        calibration.distortion,
    )
    return pixels.reshape(-1, 2)


def solve_pnp(points_deck, pixels, calibration):
    X = np.asarray(points_deck, dtype=np.float64)
    u = np.asarray(pixels, dtype=np.float64)
    if (
        X.ndim != 2
        or X.shape[1] != 3
        or u.shape != (len(X), 2)
        or len(X) < 4
        or not np.isfinite(u).all()
    ):
        raise ValueError("at least four finite 3D/2D correspondences required")
    if np.linalg.matrix_rank(X - X.mean(0)) < 2:
        return []
    planar = np.linalg.matrix_rank(X - X.mean(0), tol=1e-7) == 2
    result = cv2.solvePnPGeneric(
        X,
        u,
        calibration.K,
        calibration.distortion,
        flags=cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP,
    )
    branches = list(zip(result[1], result[2])) if result[0] else []
    if planar:
        # OpenCV IPPE can return unstable Rodrigues vectors at exactly pi.
        # Its homography-initialized iterative solver supplies an independent
        # candidate; keep both planar branches for the existing prior resolver.
        ok, rv, tv = cv2.solvePnP(
            X, u, calibration.K, calibration.distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if ok:
            branches.append((rv, tv))
    candidates = []
    for rv, tv in branches:
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():
            continue
        # Refine each IPPE branch separately. Near fronto-parallel geometry can
        # leave a noticeable numerical reprojection error even for exact pixels.
        rv, tv = cv2.solvePnPRefineLM(
            X, u, calibration.K, calibration.distortion, rv.copy(), tv.copy()
        )
        R, _ = cv2.Rodrigues(rv)
        if not np.isfinite(R).all():
            continue
        T = invert(transform(R, tv.ravel()))
        if (transform_points(invert(T), X)[:, 2] <= 0).any() or T[2, 3] <= 0:
            continue
        error = float(
            np.sqrt(
                np.mean(np.sum((project_points(X, T, calibration) - u) ** 2, axis=1))
            )
        )
        if np.isfinite(error):
            candidates.append((T, error))
    return candidates
