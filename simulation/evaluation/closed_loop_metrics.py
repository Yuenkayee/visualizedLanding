import numpy as np


def closed_loop_metrics(rows, target_height=0.8, tolerance=0.5):
    if not rows:
        raise ValueError("no rows")
    p = np.array([r["truth"]["T_deck_camera"] for r in rows])[:, :3, 3]
    v = np.array([r["truth"]["velocity_deck"] for r in rows])
    target = np.array([0, 0, target_height])
    error = np.linalg.norm(p - target, axis=1)
    success = (error < tolerance) & (np.linalg.norm(v, axis=1) < 0.3)
    first = int(np.flatnonzero(success)[0]) if success.any() else None
    return dict(
        final_position=p[-1].tolist(),
        final_target_error=float(error[-1]),
        final_speed=float(np.linalg.norm(v[-1])),
        reached_target=bool(success.any()),
        time_to_target=rows[first]["timestamp"] if first is not None else None,
        min_height=float(p[:, 2].min()),
        max_speed=float(np.linalg.norm(v, axis=1).max()),
    )
