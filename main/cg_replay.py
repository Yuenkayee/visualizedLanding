"""CG evaluation for recorded navigation; aircraft truth is used only for scoring."""
import numpy as np
from scipy.spatial.transform import Rotation
from navigation.common.cg_signal import CgSignal, relative_cg
from navigation.common.frames import invert


class CgReplay:
    def __init__(self, geometry, config=None):
        self.geometry = geometry
        self.signal = CgSignal(config)
        self.previous = None
        self.rate_fallback_used = False

    def update(self, row, estimate):
        t = row["timestamp"]
        A = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        R = A @ np.asarray(row["truth"]["T_world_deck"])[:3, :3]
        omega = row.get("ship_angular_velocity_deck", row.get("feedback", {}).get("ship_angular_velocity_deck"))
        if omega is None:
            # Legacy logs retain communicated ship orientation, but no rate.
            # Backward differences use only past/current orientations, never
            # future poses or the aircraft truth trajectory.
            self.rate_fallback_used = True
            omega = np.zeros(3)
            if self.previous is not None:
                previous_t, previous_R = self.previous
                if t > previous_t:
                    omega = Rotation.from_matrix(previous_R.T @ R).as_rotvec() / (t - previous_t)
        self.previous = (t, R)
        values = relative_cg(estimate, self.geometry, R, omega)
        values.update(self.signal.step(t, values["cg_relative_estimate_ned"],
            values["navigation_velocity_cg_ned_mps"], estimate.healthy, estimate.timestamp))
        # Separate scoring branch: fixed optical centre, actual simulated mount.
        truth = row["truth"]
        body = np.asarray(truth["T_deck_camera"]) @ invert(truth["T_body_camera"])
        offset = np.diag([1., -1., -1.]) @ self.geometry["H_ship_body"]
        values["cg_relative_truth_ned"] = (R @ (body[:3, 3] + offset)).tolist()
        values["ship_angular_velocity_deck"] = np.asarray(omega).tolist()
        return values
