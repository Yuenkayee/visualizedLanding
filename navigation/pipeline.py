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
        point = self.config.get("filter_point_body")
        self.filter_point_body = (
            None if point is None else finite_array(point, (3,), "filter_point_body")
        )

    def camera_lever(self, mount):
        """Reference point -> optical centre, in the current camera axes."""
        T = mount.T_body_camera
        return T[:3, :3].T @ (T[:3, 3] - self.filter_point_body)

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
            self.health.record(measurement.timestamp)
        return self.estimate()

    def process_camera(self, frame):
        self.process_gimbal(frame.gimbal, frame.timestamp)
        self.diagnostics["vision_accepted"] = False
        for key in ("vision_rejection", "reprojection_error", "vision_mahalanobis", "mask_iou", "mask_keypoint_difference_px"):
            self.diagnostics.pop(key, None)
        self.diagnostics["vision_source"] = "learned"
        self.diagnostics["vision_position_correction_m"] = 0.0
        detection = self.detector(frame.image)
        pixel_sigma = self.config.get("pixel_sigma", 1.5)
        if detection is None:
            self.diagnostics["vision_rejection"] = "no_target"
            return self.estimate()
        if self.config.get("mask_geometry", True) and detection.ordered:
            recovery = recover_from_mask(detection, frame.image.shape, self.marker_points,
                                         self.calibration, self.estimate().T_deck_camera, self.config)
            if recovery is not None:
                pixels, iou, disagreement, scale = recovery
                self.diagnostics.update(mask_iou=iou, mask_keypoint_difference_px=disagreement)
                if disagreement > self.config.get("mask_recovery_difference_px", 3):
                    detection.keypoints = pixels
                    detection.visibility = None  # Independent, complete silhouette evidence.
                    self.diagnostics["vision_source"] = "mask_recovery"
                    pixel_sigma = max(pixel_sigma, float(scale.max()) * self.config.get("mask_pixel_sigma", 1.))
        if detection.visibility is not None and np.min(detection.visibility) < self.config.get("minimum_keypoint_visibility", .5):
            self.diagnostics["vision_rejection"] = "invisible_keypoint"
            return self.estimate()
        if detection is not None and valid_keypoints(
            detection.keypoints, frame.image.shape
        ):
            match = associate_keypoints(
                self.marker_points,
                detection.keypoints,
                self.calibration,
                self.estimate().T_deck_camera,
                detection.ordered,
                refine=self.config.get("refine_pnp", True),
            )
            if match is not None:
                T, error, _ = match
                self.diagnostics["reprojection_error"] = error
                if error <= self.config.get("max_reprojection_error", 5.0):
                    P = estimate_covariance(
                        self.marker_points,
                        T,
                        self.calibration,
                        max(pixel_sigma, error / np.sqrt(2)) if self.config.get("adaptive_pixel_noise", True) else pixel_sigma,
                    )
                    return self.process_pose(
                        PoseMeasurement(frame.timestamp, T, P, error)
                    )
                self.diagnostics["vision_rejection"] = "reprojection"
            else:
                self.diagnostics["vision_rejection"] = "pnp_failed"
        else:
            self.diagnostics["vision_rejection"] = "invalid_quad"
        return self.estimate()

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
