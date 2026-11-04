"""Benchmark adapters; the production navigation algorithms remain unchanged.

N0 uses the classical HDetector with the full backend; N2 omits LiDAR; N3 is
the full pipeline. A1 removes ROI and *all* mask geometry, A2 removes reference
feedforward, and A3 bypasses the recovery-pending output-health gate. A3 retains
the configured confirmation count and native recovery bookkeeping. With the
default count of two this re-enables health after the first accepted reacquired
pose, as a count-one ablation would, but changes no numeric algorithm settings.
All other health/gimbal checks remain active. Other configuration values
(including noise, gates, refinement and model weights) are preserved.

N1 uses ordered learned corners, the existing PnP solver and pixel covariance.
Planar ambiguity uses the supplied nominal camera orientation, then the last
accepted camera orientation, without extrapolation. Only the resolver's position
term is disabled; its reprojection/orientation weights are unchanged. There is
no translation prior, ESKF, mask recovery, ROI, LiDAR or reference feedforward.
Its estimate is the raw optical-centre pose/covariance, with zero placeholder
velocities and diagnostics marking them unavailable. Do not shape N1 outputs.
Advance its clock with process_gimbal on every tick: estimate() is None unless
a camera frame succeeded at that timestamp. Its state is only a nominal state
and clock, not a filtered output; filter_point_body is None.

Call reset_telemetry() before each benchmark tick. Timing accumulates seconds:
inference_s measures complete detector calls (pre/postprocessing and ROI proposal
included), synchronized around calls on CUDA; geometry_s covers mask recovery,
PnP/covariance and LiDAR geometry; filter_s covers prediction, fusion, reference
inputs and filtered output conversion. These three stages are exclusive.
camera_s/lidar_s are inclusive packet wall times, not additional stages.
last_detection is always the full-image detection. raw_pose is the last geometric
PoseMeasurement before fusion, even if the innovation gate rejects it; last_pose
is the current camera packet's accepted raw measurement. last_plane and
lidar_associated_points expose the *same* fit/input used by fusion. Diagnostics
retain native camera attempts and add LiDAR point counts, plane_inlier_ratio
(also lidar_inlier_ratio, relative to associated points), and lidar_residual_m
(the fitted plane's RMS). vision_mahalanobis_threshold/lidar_mahalanobis_threshold
are the native squared-distance chi-square cutoffs, or None for disabled gates.

An optional detector=HDetector(...) can share stateless inference weights among
learned methods. Wrappers, telemetry, config and ESKF states remain independent;
N0 always constructs its own classical detector. No detector is monkeypatched.
"""

from contextlib import contextmanager
from copy import deepcopy
from inspect import signature
from time import perf_counter

import numpy as np
from scipy.stats import chi2

from navigation.common.contracts import NavigationEstimate, PoseMeasurement
from navigation.common.frames import skew, transform, transform_points
from navigation.common.timestamps import require_monotonic
from navigation.fusion.eskf_update_lidar import update_lidar
from navigation.fusion.measurement_gating import update as _measurement_update
from navigation.lidar.associate_deck import associate_deck, plane_matches_deck
from navigation.lidar.extract_deck_plane import extract_deck_plane
from navigation.pipeline import NavigationPipeline
from navigation.pose.estimate_covariance import estimate_covariance
from navigation.pose.resolve_planar_ambiguity import resolve_planar_ambiguity
from navigation.pose.solve_pnp import solve_pnp
from navigation.vision.geometric_constraints import valid_keypoints


TIMING_KEYS = ("inference_s", "geometry_s", "filter_s", "camera_s", "lidar_s")
METHODS = ("N0", "N1", "N2", "N3", "A1", "A2", "A3")
# Derive telemetry from the update's actual default; do not change its gate.
_GATE_PROBABILITY = signature(_measurement_update).parameters["probability"].default
_VISION_THRESHOLD = float(chi2.ppf(_GATE_PROBABILITY, 6))
_LIDAR_THRESHOLD = float(chi2.ppf(_GATE_PROBABILITY, 4))


class _TimedDetector:
    def __init__(self, detector, owner):
        self.backend = detector
        self.owner = owner

    def _synchronize(self):
        device = getattr(self.backend, "device", "cpu")
        if (getattr(self.backend, "model", None) is not None
                and str(device).split(":")[0] == "cuda"):
            import torch

            torch.cuda.synchronize(device)

    def _call(self, function, *args):
        self._synchronize()
        with self.owner._stage("inference_s"):
            try:
                return function(*args)
            finally:
                self._synchronize()

    def __call__(self, image):
        detection = self._call(self.backend, image)
        self.owner.last_detection = detection
        return detection

    def __getattr__(self, name):
        attribute = getattr(self.backend, name)
        # Preserve hasattr(detector, 'detect_roi') for backends without ROI.
        if name == "detect_roi":
            return lambda *args: self._call(attribute, *args)
        return attribute


class NavigationMethod(NavigationPipeline):
    """Native pipeline with local timing and measurement telemetry."""

    def __init__(self, name, calibration, points, config, state, detector=None):
        self.name = name
        self.lidar_enabled = name not in ("N1", "N2")
        self.timing = {}
        self._stages = []
        # An injected model must not trigger another checkpoint load. The base
        # constructor still initializes every other field with its native code.
        init_config = config if detector is None else {**config, "checkpoint": None}
        super().__init__(calibration, np.array(points, dtype=float, copy=True),
                         init_config, deepcopy(state))
        self.config = config
        self.detector = _TimedDetector(
            self.detector if detector is None else detector, self
        )
        self.reset_telemetry()

    def reset_telemetry(self):
        """Clear per-tick telemetry, preserving all estimator/health state."""
        self.timing.update(dict.fromkeys(TIMING_KEYS, 0.0))
        self.diagnostics.clear()
        self.diagnostics.update(
            vision_mahalanobis_threshold=None if self.name == "N1" else _VISION_THRESHOLD,
            lidar_mahalanobis_threshold=_LIDAR_THRESHOLD if self.lidar_enabled else None,
        )
        if self.name == "A3":
            self.diagnostics["vision_recovery_gate_enabled"] = False
        if self.name == "N1":
            self.diagnostics.update(single_frame=True, velocity_available=False,
                                    angular_velocity_available=False)
        self.last_detection = None
        self.raw_pose = None
        self.last_pose = None
        self.last_plane = None
        self.lidar_associated_points = None

    @contextmanager
    def _stage(self, key):
        entry = [perf_counter(), 0.0]
        self._stages.append(entry)
        try:
            yield
        finally:
            elapsed = perf_counter() - entry[0]
            self._stages.pop()
            self.timing[key] += max(0.0, elapsed - entry[1])
            if self._stages:
                self._stages[-1][1] += elapsed

    @contextmanager
    def _packet(self, key):
        start = perf_counter()
        try:
            yield
        finally:
            self.timing[key] += perf_counter() - start

    def advance(self, timestamp):
        with self._stage("filter_s"):
            return super().advance(timestamp)

    def process_gimbal(self, sample, timestamp):
        with self._stage("filter_s"):
            return super().process_gimbal(sample, timestamp)

    def process_gimbal_reference(self, command):
        with self._stage("filter_s"):
            return super().process_gimbal_reference(command)

    def process_pose(self, measurement):
        self.raw_pose = deepcopy(measurement)
        with self._stage("filter_s"):
            result = super().process_pose(measurement)
        if self.diagnostics.get("vision_accepted", False):
            self.last_pose = deepcopy(self.raw_pose)
        return result

    def _try_camera(self, frame, detection, path, mode):
        with self._stage("geometry_s"):
            return super()._try_camera(frame, detection, path, mode)

    def process_camera(self, frame):
        self.last_detection = self.raw_pose = self.last_pose = None
        with self._packet("camera_s"):
            return super().process_camera(frame)

    def estimate(self):
        with self._stage("filter_s"):
            estimate = super().estimate()
            if self.name == "A3":
                # Categorical ablation only: retain the native recovery count,
                # but compose output health without `and not self.recovering`.
                healthy, _ = self.health.check(self.state)
                if self.config.get("require_gimbal", False) or self.filter_point_body is not None:
                    healthy = healthy and self.gimbal_synchronized(self.state.timestamp)
                estimate.healthy = healthy
            return estimate

    def process_lidar(self, frame):
        with self._packet("lidar_s"):
            return self._process_lidar(frame)

    def _process_lidar(self, frame):
        # Mirrored from NavigationPipeline.process_lidar: same transformations,
        # call order, gates and default arguments. Keep this in sync with core.
        # The one fitted plane supplies both telemetry and the native update.
        synchronized = self.process_gimbal(frame.gimbal, frame.timestamp)
        self.diagnostics["lidar_accepted"] = False
        for key in ("lidar_rejection", "lidar_mahalanobis", "plane_rms"):
            self.diagnostics.pop(key, None)
        self.last_plane = self.lidar_associated_points = None
        self.diagnostics.update(lidar_associated_points=0, lidar_inlier_count=0,
                                lidar_inlier_ratio=None, plane_inlier_ratio=None,
                                lidar_residual_m=None)
        if not self.lidar_enabled:
            self.diagnostics["lidar_rejection"] = "disabled"
            return self.estimate()
        if self.calibration.lidar_mount == "body" and not synchronized:
            self.diagnostics["lidar_rejection"] = "missing_or_stale_gimbal"
            return self.estimate()
        with self._stage("geometry_s"):
            points = transform_points(
                self.calibration.camera_lidar_at(frame.gimbal, frame.timestamp), frame.points
            )
            if self.filter_point_body is not None:
                if not synchronized:
                    return self.estimate()
                points += self.camera_lever(self.gimbal)
            mask = associate_deck(
                points,
                self.state.pose,
                self.config.get("deck_half_extent", (12.0, 8.0)),
                self.config.get("lidar_height_tolerance", 3.0),
            )
            associated = points[mask]
            self.lidar_associated_points = associated
            self.diagnostics["lidar_associated_points"] = len(associated)
            self.diagnostics["lidar_association_rejection_fraction"] = 1-len(associated)/len(points) if len(points) else None
            plane = extract_deck_plane(
                associated, threshold=self.config.get("plane_threshold", 0.08)
            )
            self.last_plane = plane
            if plane is not None:
                count = int(np.count_nonzero(plane.inliers))
                self.diagnostics.update(lidar_inlier_count=count,
                                        lidar_inlier_ratio=count / len(associated),
                                        plane_inlier_ratio=count / len(associated),
                                        lidar_residual_m=plane.rms)
            if plane is not None and plane_matches_deck(plane, self.state.pose):
                mount_jacobian = None
                mount_covariance = (
                    self.gimbal.covariance if self.calibration.lidar_mount == "body" else None
                )
                if self.calibration.lidar_mount == "camera" and self.filter_point_body is not None:
                    mount_covariance = self.gimbal.covariance
                    mount_jacobian = np.zeros((4, 3))
                    mount_jacobian[3] = -plane.normal @ skew(self.camera_lever(self.gimbal))
                with self._stage("filter_s"):
                    before = self.state.position.copy()
                    ok, d2 = update_lidar(
                        self.state, plane, mount_covariance=mount_covariance,
                        mount_jacobian=mount_jacobian,
                    )
                self.diagnostics.update(
                    lidar_accepted=ok, lidar_mahalanobis=d2, plane_rms=plane.rms,
                    lidar_position_correction_m=float(np.linalg.norm(self.state.position-before))
                )
                if not ok:
                    self.diagnostics["lidar_rejection"] = "innovation_gate"
            else:
                self.diagnostics["lidar_rejection"] = (
                    "no_deck_plane" if plane is None else "plane_inconsistent"
                )
        return self.estimate()


class SingleFrameMethod(NavigationMethod):
    """N1: causal orientation disambiguation, no temporal pose estimation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orientation_prior = self.state.rotation.copy()
        self._accepted_pose = None

    def advance(self, timestamp):
        timestamp = require_monotonic(self.state.timestamp, timestamp, allow_equal=True)
        if timestamp > self.state.timestamp:
            self._accepted_pose = None
            self.diagnostics["vision_accepted"] = False
        self.state.timestamp = timestamp

    def process_gimbal(self, sample, timestamp):
        # Reuse encoder validity checks, but bypass all filter timing/prediction.
        return NavigationPipeline.process_gimbal(self, sample, timestamp)

    def process_gimbal_reference(self, command):
        return None

    def process_pose(self, measurement):
        self.advance(measurement.timestamp)
        self.raw_pose = deepcopy(measurement)
        self.last_pose = deepcopy(measurement)
        self._accepted_pose = deepcopy(measurement)
        self._orientation_prior = measurement.T_deck_camera[:3, :3].copy()
        self.diagnostics["vision_accepted"] = True
        self.diagnostics.pop("vision_rejection", None)
        return self.estimate()

    def process_camera(self, frame):
        with self._packet("camera_s"):
            if self.last_camera_timestamp is not None and frame.timestamp <= self.last_camera_timestamp:
                raise ValueError("Camera frames must have strictly increasing timestamps")
            self.last_camera_timestamp = frame.timestamp
            self.process_gimbal(frame.gimbal, frame.timestamp)
            self._accepted_pose = None
            self.last_detection = self.raw_pose = self.last_pose = None
            self._clear_vision_candidate()
            self.diagnostics.update(vision_attempts=[], vision_roi_attempted=False,
                                    vision_roi={"status": "disabled"}, vision_primary_rejection=None)
            if self.config.get("require_gimbal", False) and not self.gimbal_synchronized(frame.timestamp):
                self.diagnostics["vision_rejection"] = "missing_or_stale_gimbal"
                return None
            detection = self.detector(frame.image)
            with self._stage("geometry_s"):
                self._single_frame_pose(frame, detection)
            self.diagnostics["vision_attempts"].append({
                k: self.diagnostics[k] for k in self._vision_candidate_keys if k in self.diagnostics
            })
            self.diagnostics["vision_primary_rejection"] = self.diagnostics.get("vision_rejection")
            return self.estimate()

    def _single_frame_pose(self, frame, detection):
        self.diagnostics["vision_path"] = "full"
        if detection is None:
            self.diagnostics["vision_rejection"] = "no_target"
            return
        if not detection.ordered:
            self.diagnostics["vision_rejection"] = "unordered_keypoints"
            return
        if (detection.visibility is not None and np.min(detection.visibility)
                < self.config.get("minimum_keypoint_visibility", .5)):
            self.diagnostics["vision_rejection"] = "invisible_keypoint"
            return
        if not valid_keypoints(detection.keypoints, frame.image.shape):
            self.diagnostics["vision_rejection"] = "invalid_quad"
            return
        candidates = solve_pnp(self.marker_points, detection.keypoints, self.calibration,
                               refine=self.config.get("refine_pnp", True))
        match = resolve_planar_ambiguity(
            candidates, transform(self._orientation_prior), position_scale=np.inf
        )
        if match is None:
            self.diagnostics["vision_rejection"] = "pnp_failed"
            return
        pose, error = match
        self.diagnostics["reprojection_error"] = error
        if error > self.config.get("max_reprojection_error", 5.0):
            self.diagnostics["vision_rejection"] = "reprojection"
            return
        sigma = self.config.get("pixel_sigma", 1.5)
        if self.config.get("adaptive_pixel_noise", True):
            sigma = max(sigma, error / np.sqrt(2))
        self.diagnostics["vision_pixel_sigma"] = float(sigma)
        covariance = estimate_covariance(self.marker_points, pose, self.calibration, sigma)
        self.process_pose(PoseMeasurement(frame.timestamp, pose, covariance, error))

    def estimate(self):
        pose = self._accepted_pose
        if pose is None or pose.timestamp != self.state.timestamp:
            return None
        flags = dict(single_frame=True, velocity_available=False,
                     angular_velocity_available=False)
        self.diagnostics.update(flags)
        return NavigationEstimate(
            pose.timestamp, pose.T_deck_camera.copy(), np.zeros(3), pose.covariance.copy(),
            True, deepcopy(self.diagnostics), np.zeros(3), self.gimbal,
        )


def make_method(name, calibration, points, nav_config, state, *, detector=None):
    """Build an isolated benchmark method; optionally reuse a learned HDetector."""
    if name not in METHODS:
        raise ValueError(f"Unknown navigation method {name!r}; expected one of {METHODS}")
    config = deepcopy(nav_config) if nav_config is not None else {}
    if name == "N0":
        config["checkpoint"] = None
        detector = None
    elif name == "N1":
        config.update(mask_geometry=False, roi_recovery=False, filter_point_body=None,
                      gimbal_reference_feedforward=False)
    elif name == "A1":
        config.update(roi_recovery=False, mask_geometry=False)
    elif name == "A2":
        config["gimbal_reference_feedforward"] = False
    cls = SingleFrameMethod if name == "N1" else NavigationMethod
    return cls(name, calibration, points, config, state, detector)
