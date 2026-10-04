import numpy as np
from scipy.spatial.transform import Rotation


def navigation_metrics(rows):
    if not rows:
        raise ValueError("no rows")
    errors = []
    angles = []
    health = []
    for row in rows:
        truth = np.asarray(row["truth"]["T_deck_camera"])
        estimate = np.asarray(row["estimate"]["T_deck_camera"])
        errors.append(estimate[:3, 3] - truth[:3, 3])
        angles.append(
            np.linalg.norm(
                Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).as_rotvec()
            )
        )
        health.append(row["estimate"]["healthy"])
    e = np.asarray(errors)
    distance = np.linalg.norm(e, axis=1)
    return dict(
        position_rmse=float(np.sqrt(np.mean(distance**2))),
        axis_rmse=np.sqrt(np.mean(e**2, axis=0)).tolist(),
        position_p95=float(np.percentile(distance, 95)),
        attitude_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(angles))))),
        healthy_fraction=float(np.mean(health)),
        samples=len(rows),
    )
