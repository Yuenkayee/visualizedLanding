"""Chronological fusion; delayed packets must be replayed, never applied at a newer time."""

import numpy as np
from navigation.common.contracts import NavigationEstimate, PoseMeasurement, finite_array
from navigation.common.frames import transform_points, skew, transform
from navigation.fusion.eskf_state import ESKFState
from navigation.fusion.eskf_predict import predict
from navigation.fusion.eskf_update_vision import update_vision
from navigation.fusion.eskf_update_lidar import update_lidar
from navigation.vision.infer_h import HDetector
from navigation.vision.associate_keypoints import associate_keypoints
from navigation.vision.geometric_constraints import valid_keypoints
from navigation.vision.mask_geometry import recover_from_mask
from navigation.lidar.associate_deck import associate_deck, plane_matches_deck
from navigation.lidar.extract_deck_plane import extract_deck_plane
from navigation.pose.estimate_covariance import estimate_covariance
from navigation.health_monitor import HealthMonitor


class NavigationPipeline:
    def __init__(self, calibration, marker_points, config=None, state=None):
        self.calibration = calibration
        self.marker_points = np.asarray(marker_points, float)
        self.config = config or {}
        self.state = state or ESKFState()
        self.detector = HDetector(self.config.get("checkpoint"), self.config.get("device", "cpu"))
        self.health = HealthMonitor(
            self.config.get("max_measurement_age", 1.0),
            self.config.get("max_position_std", 5.0),
        )
        self.diagnostics = {}
        self.gimbal = None
        self.last_gimbal = None
        self.reference_rate_deck = np.zeros(3)
        self.recovering = False
        self.recovery_count = 0
        self.last_camera_timestamp = None
        self.recovery_required = int(self.config.get("reacquisition_confirmations", 2))
        if self.recovery_required < 1:
            raise ValueError("reacquisition_confirmations must be positive")
        point = self.config.get("filter_point_body")
        self.filter_point_body = (
            None if point is None else finite_array(point, (3,), "filter_point_body")
        )

    def camera_lever(self, mount):
        """Reference point -> optical centre, in the current camera axes."""
        T = mount.T_body_camera
        return T[:3, :3].T @ (T[:3, 3] - self.filter_point_body)

    def process_gimbal_reference(self, command):
        """Known motion input for the NEXT prediction, never a pose update.

        Keep the estimated residual angular rate while accounting for changes
        in the slow reference rate. Otherwise a constant-rate predictor keeps
        rotating after the gimbal brakes and can lose the target association.
        Finite servo error remains for vision to estimate; health/covariance and
        actual encoder extrinsics are not replaced with the command.
        """
        if command is None or not self.config.get("gimbal_reference_feedforward", True):
            return
        if abs(command.timestamp - self.state.timestamp) > 1e-8:
            raise ValueError("reference input must follow navigation at the same timestamp")
        delta = command.angular_velocity_deck - self.reference_rate_deck
        self.state.angular_velocity += self.state.rotation.T @ delta
        self.reference_rate_deck = command.angular_velocity_deck.copy()

    def gimbal_synchronized(self, timestamp):
        return (
            self.gimbal is not None and self.gimbal.valid
            and abs(self.gimbal.timestamp - timestamp) <= 1e-8
        )

    def process_gimbal(self, sample, timestamp):
        self.advance(timestamp)
        # A missing/stale encoder never silently reuses an old rigid extrinsic.
        self.gimbal = sample
        synchronized = self.gimbal_synchronized(timestamp)
        self.diagnostics["gimbal_synchronized"] = bool(synchronized)
        self.diagnostics["gimbal_saturated"] = (
            bool(sample.saturated) if sample is not None else False
        )
        if synchronized:
            self.last_gimbal = sample
        return synchronized

    def advance(self, t):
        predict(
            self.state,
            t,
            acceleration_noise=self.config.get("acceleration_noise", 0.5),
            angular_acceleration_noise=self.config.get(
                "angular_acceleration_noise", 0.05
            ),
        )

    def process_pose(self, measurement):
        self.advance(measurement.timestamp)
        if self.filter_point_body is not None:
            if not self.gimbal_synchronized(measurement.timestamp):
                self.diagnostics["vision_accepted"] = False
                return self.estimate()
            # Filter translation/velocity at a fixed BODY point, while attitude
            # remains the stabilized camera attitude. This removes lever-arm
            # oscillations without claiming access to aircraft pose truth.
            lever = self.camera_lever(self.gimbal)
            R = measurement.T_deck_camera[:3, :3]
            J = np.eye(6)
            J[:3, 3:] = R @ skew(lever)
            Jg = np.zeros((6, 3))
            Jg[:3] = -R @ skew(lever)
            P = J @ measurement.covariance @ J.T + Jg @ self.gimbal.covariance @ Jg.T
            measurement = PoseMeasurement(
                measurement.timestamp,
                transform(R, measurement.T_deck_camera[:3, 3] - R @ lever), P,
                measurement.reprojection_error, measurement.source,
            )
        before = self.state.position.copy()
        ok, d2 = update_vision(self.state, measurement)
        self.diagnostics.update(vision_accepted=ok, vision_mahalanobis=d2,
                                vision_position_correction_m=float(np.linalg.norm(self.state.position - before)))
        if not ok:
            self.diagnostics["vision_rejection"] = "innovation_gate"
        if ok:
            self.diagnostics.pop("vision_rejection", None)
            previous = self.health.last_measurement
            age = None if previous is None else measurement.timestamp - previous
            reacquired = age is not None and age > self.health.max_age
            if reacquired:
                self.recovering, self.recovery_count = True, 0
            if self.recovering:
                if age is not None and age > self.config.get("reacquisition_max_gap_s", .3):
                    self.recovery_count = 0
                self.recovery_count += 1
                self.recovering = self.recovery_count < self.recovery_required
            self.diagnostics.update(vision_previous_measurement_age_s=age,
                                    vision_reacquired=bool(reacquired))
            self.health.record(measurement.timestamp)
        return self.estimate()

    def process_camera(self, frame):
        if self.last_camera_timestamp is not None and frame.timestamp <= self.last_camera_timestamp:
            raise ValueError("Camera frames must have strictly increasing timestamps")
        self.last_camera_timestamp = frame.timestamp
        self.process_gimbal(frame.gimbal, frame.timestamp)
        self.diagnostics.update(vision_attempts=[], vision_roi_attempted=False,
                                vision_roi={"status": "not_needed"}, vision_primary_rejection=None)
        self._clear_vision_candidate()
        if (self.config.get("require_gimbal", False) or self.filter_point_body is not None) and not self.gimbal_synchronized(frame.timestamp):
            self.diagnostics["vision_rejection"] = "missing_or_stale_gimbal"
            if self.recovering:
                self.recovery_count = 0
            return self.estimate()
        detection = self.detector(frame.image)
        if self._try_camera(frame, detection, "full", "primary"):
            return self.estimate()
        primary = self.diagnostics["vision_attempts"][0].copy()
        self.diagnostics["vision_primary_rejection"] = primary["vision_rejection"]
        # Rejecting an innovation does not mutate the ESKF. Retry the same
        # image at the same timestamp; return immediately after ONE accepted update.
        if self.config.get("roi_recovery", True) and hasattr(self.detector, "detect_roi"):
            cropped, detail = self.detector.detect_roi(frame.image, detection, self.config)
            self.diagnostics["vision_roi"] = detail
            self.diagnostics["vision_roi_attempted"] = "box" in detail
            if cropped is not None:
                if self._try_camera(frame, cropped, "roi", "learned"):
                    return self.estimate()
                if (self.config.get("mask_geometry", True) and self.config.get("mask_fallback_minimum_side_px", 16) > 0
                        and self._try_camera(frame, cropped, "roi_mask", "mask")):
                    return self.estimate()
        if self.config.get("mask_geometry", True) and self.config.get("mask_fallback_minimum_side_px", 16) > 0:
            if self._try_camera(frame, detection, "full_small_mask", "mask"):
                return self.estimate()
        # Keep the primary rejection as the top-level reason, with every retry
        # recorded separately; do not report a stale residual from another path.
        self._clear_vision_candidate()
        self.diagnostics.update(primary)
        if self.recovering:
            self.recovery_count = 0
        return self.estimate()

    _vision_candidate_keys = ("vision_accepted", "vision_rejection", "vision_source", "vision_path",
        "reprojection_error", "vision_mahalanobis", "mask_iou", "mask_keypoint_difference_px",
        "vision_position_correction_m", "vision_pixel_sigma", "vision_mask",
        "vision_previous_measurement_age_s", "vision_reacquired")

    def _clear_vision_candidate(self):
        for key in self._vision_candidate_keys:
            self.diagnostics.pop(key, None)
        self.diagnostics["vision_accepted"] = False
        self.diagnostics["vision_source"] = "learned"
        self.diagnostics["vision_position_correction_m"] = 0.0
        self.diagnostics["vision_reacquired"] = False

    def _try_camera(self, frame, detection, path, mode):
        self._clear_vision_candidate()
        self.diagnostics["vision_path"] = path
        def finish(reason=None):
            if reason:
                self.diagnostics["vision_rejection"] = reason
            self.diagnostics["vision_attempts"].append({k: self.diagnostics[k]
                for k in self._vision_candidate_keys if k in self.diagnostics})
            return self.diagnostics["vision_accepted"]
        pixel_sigma = self.config.get("pixel_sigma", 1.5)
        if detection is None:
            return finish("no_target")
        pixels, visibility = detection.keypoints, detection.visibility
        if path.startswith("roi"):
            scale, _ = detection.mask_coordinates(frame.image.shape)
            pixel_sigma = max(pixel_sigma, float(scale.max()) * self.config.get("roi_pixel_sigma", 1.))
        if mode != "learned" and self.config.get("mask_geometry", True) and detection.ordered:
            cfg = self.config
            if mode == "mask":
                cfg = {**cfg, "mask_minimum_side_px": cfg.get("mask_fallback_minimum_side_px", 16),
                       "mask_adaptive_edges": cfg.get("mask_fallback_adaptive_edges", True)}
            mask_diagnostics = {}
            recovery = recover_from_mask(detection, frame.image.shape, self.marker_points,
                                         self.calibration, self.estimate().T_deck_camera, cfg, mask_diagnostics)
            self.diagnostics["vision_mask"] = mask_diagnostics
            if recovery is not None:
                recovered, iou, disagreement, scale = recovery
                self.diagnostics.update(mask_iou=iou, mask_keypoint_difference_px=disagreement)
                if mode == "mask" or disagreement > self.config.get("mask_recovery_difference_px", 3):
                    pixels = recovered
                    visibility = None  # Independent, complete silhouette evidence.
                    self.diagnostics["vision_source"] = "mask_recovery"
                    pixel_sigma = max(pixel_sigma, float(scale.max()) * self.config.get("mask_pixel_sigma", 1.)
                                      * mask_diagnostics["noise_multiplier"])
            elif mode == "mask":
                return finish("mask_geometry")
        elif mode == "mask":
            return finish("mask_unavailable")
        self.diagnostics["vision_pixel_sigma"] = float(pixel_sigma)
        if visibility is not None and np.min(visibility) < self.config.get("minimum_keypoint_visibility", .5):
            return finish("invisible_keypoint")
        if valid_keypoints(pixels, frame.image.shape):
            match = associate_keypoints(
                self.marker_points,
                pixels,
                self.calibration,
                self.estimate().T_deck_camera,
                detection.ordered,
                refine=self.config.get("refine_pnp", True),
            )
            if match is not None:
                T, error, _ = match
                self.diagnostics["reprojection_error"] = error
                if error <= self.config.get("max_reprojection_error", 5.0):
                    if self.config.get("adaptive_pixel_noise", True):
                        pixel_sigma = max(pixel_sigma, error / np.sqrt(2))
                    self.diagnostics["vision_pixel_sigma"] = float(pixel_sigma)
                    P = estimate_covariance(
                        self.marker_points,
                        T,
                        self.calibration,
                        pixel_sigma,
                    )
                    self.process_pose(PoseMeasurement(frame.timestamp, T, P, error))
                    return finish()
                return finish("reprojection")
            else:
                return finish("pnp_failed")
        else:
            return finish("invalid_quad")

    def process_lidar(self, frame):
        synchronized = self.process_gimbal(frame.gimbal, frame.timestamp)
        self.diagnostics["lidar_accepted"] = False
        for key in ("lidar_rejection", "lidar_mahalanobis", "plane_rms"):
            self.diagnostics.pop(key, None)
        if self.calibration.lidar_mount == "body" and not synchronized:
            self.diagnostics["lidar_rejection"] = "missing_or_stale_gimbal"
            return self.estimate()
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
        plane = extract_deck_plane(
            points[mask], threshold=self.config.get("plane_threshold", 0.08)
        )
        if plane is not None and plane_matches_deck(plane, self.state.pose):
            mount_jacobian = None
            mount_covariance = (
                self.gimbal.covariance if self.calibration.lidar_mount == "body" else None
            )
            if self.calibration.lidar_mount == "camera" and self.filter_point_body is not None:
                # A co-moving LiDAR has no encoder normal error; only shifting
                # its plane to the BODY reference adds encoder distance error.
                mount_covariance = self.gimbal.covariance
                mount_jacobian = np.zeros((4, 3))
                mount_jacobian[3] = -plane.normal @ skew(self.camera_lever(self.gimbal))
            ok, d2 = update_lidar(
                self.state, plane, mount_covariance=mount_covariance,
                mount_jacobian=mount_jacobian,
            )
            self.diagnostics.update(
                lidar_accepted=ok, lidar_mahalanobis=d2, plane_rms=plane.rms
            )
            if not ok:
                self.diagnostics["lidar_rejection"] = "innovation_gate"
            # A plane does not observe x/y/yaw: only full pose refreshes global health.
        else:
            self.diagnostics["lidar_rejection"] = "no_deck_plane" if plane is None else "plane_inconsistent"
        return self.estimate()

    def estimate(self):
        healthy, health = self.health.check(self.state)
        healthy = healthy and not self.recovering
        health.update(vision_recovery_pending=self.recovering,
                      vision_recovery_count=self.recovery_count,
                      vision_recovery_required=self.recovery_required)
        if self.config.get("require_gimbal", False) or self.filter_point_body is not None:
            healthy = healthy and self.gimbal_synchronized(self.state.timestamp)
        pose = self.state.pose
        velocity = self.state.velocity.copy()
        covariance = self.state.pose_covariance
        if self.filter_point_body is not None and self.last_gimbal is not None:
            mount = self.last_gimbal
            lever = self.camera_lever(mount)
            R = self.state.rotation
            pose[:3, 3] += R @ lever
            velocity += R @ np.cross(
                self.state.angular_velocity - mount.angular_velocity_camera, lever
            )
            J = np.eye(6)
            J[:3, 3:] = -R @ skew(lever)
            Jg = np.zeros((6, 3))
            Jg[:3] = R @ skew(lever)
            covariance = J @ covariance @ J.T + Jg @ mount.covariance @ Jg.T
        return NavigationEstimate(
            self.state.timestamp,
            pose,
            velocity,
            covariance,
            healthy,
            {**self.diagnostics, **health},
            self.state.angular_velocity.copy(),
            self.gimbal,
        )
