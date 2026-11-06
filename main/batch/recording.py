"""Small, lossless (float64 JSON + gzip) evaluation records at 50 Hz."""
from collections import deque
import numpy as np

NED_FROM_ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
KNOT = 1852 / 3600


def state(truth, native, geometry):
    wd = np.asarray(truth["T_world_deck"])
    dc = np.asarray(truth["T_deck_camera"])
    bc = np.asarray(truth["T_body_camera"])
    rd = NED_FROM_ENU @ wd[:3, :3]
    db = dc[:3, :3] @ bc[:3, :3].T
    ship = np.asarray(native["ship_state"]).reshape(12)
    heli = np.asarray(native["helicopter_state"]).reshape(12)
    rs = rd @ np.diag([1., -1., -1.])
    vh = ship[3:6]*KNOT + rs @ np.cross(ship[9:12], geometry["H_ship_body"])
    return dict(p=truth["gear_position_deck"], v=truth["gear_velocity_deck"],
                rd=rd.tolist(), db=db.tolist(), H_up_m=float(wd[2, 3]),
                H_normal_mps=float(rd[:, 2] @ vh),
                ship=ship.tolist(), heli=heli.tolist(),
                velocity_ned_mps=(rd @ db @ heli[3:6]*KNOT).tolist())


def compact_row(row, geometry):
    f, e = row["feedback"], row["estimate"]
    diagnostics = e.get("diagnostics", {})
    # Preserve observation evidence, excluding large rotations/covariances and
    # the duplicated vision_attempts array. Latest diagnostics may be held
    # between sensor ticks, so sampled_camera/lidar explicitly mark fresh data.
    diagnostic = {k: v for k, v in diagnostics.items()
                  if k not in ("vision_attempts", "gimbal_tracking")}
    tracking = diagnostics.get("gimbal_tracking", {})
    diagnostic["tracking"] = {k: v for k, v in tracking.items()
                              if k in ("status", "source", "target_limited", "reference_tilt_rad")}
    return dict(t=row["timestamp"], t1=row["next_timestamp"], wall_s=row["wall_s"],
                s=state(row["truth"], row["native"], geometry),
                s1=state(row["next_truth"], row["next_native"], geometry),
                stage=int(np.asarray(row["batch_telemetry"]["stage"]).item()),
                u=row["next_batch_telemetry"],
                cg_true=f["cg_relative_truth_ned"], cg_raw=f["cg_relative_estimate_ned"],
                cg_shadow=f["cg_shadow_position_ned"],
                nav_valid=bool(f["valid"]), shaper_valid=bool(f["cg_shadow_valid"]),
                nav_healthy=bool(e["healthy"]), cg_age_s=f["cg_signal_age_s"],
                sampled_camera=row["sensors"]["camera"], sampled_lidar=row["sensors"]["lidar"],
                diagnostics=diagnostic, stopped=row["plant_stopped"], stop_reason=row["stop_reason"])


class RunawayMonitor:
    """Predeclared sustained outward motion; brief excursions do not qualify."""
    def __init__(self, config):
        self.c, self.samples, self.initial, self.evidence = config, deque(), None, None

    def update(self, t, position):
        distance = float(np.linalg.norm(position))
        if self.initial is None:
            self.initial = distance
        self.samples.append((float(t), distance))
        while len(self.samples) > 2 and self.samples[1][0] <= t-self.c["window_s"]:
            self.samples.popleft()
        a = np.asarray(self.samples)
        if (t < self.c["minimum_time_s"] or len(a) < 2 or
                a[-1, 0]-a[0, 0] < self.c["window_s"]-1e-8):
            return False
        lower = max(self.c["minimum_range_m"], self.initial+self.c["increase_over_initial_m"])
        net = a[-1, 1]-a[0, 1]
        outward = float(np.mean(np.diff(a[:, 1]) > 0))
        max_return = float(np.max(np.maximum.accumulate(a[:, 1])-a[:, 1]))
        if (float(a[:, 1].min()) > lower and net >= self.c["minimum_net_outward_m"]
                and outward >= self.c["outward_fraction"] and max_return <= self.c["return_tolerance_m"]):
            self.evidence = dict(timestamp=t, initial_range_m=self.initial, range_m=distance,
                                 window_start_s=float(a[0, 0]), net_outward_m=float(net),
                                 outward_fraction=outward, max_return_m=max_return, threshold_m=lower)
            return True
        return False
