"""Chronological fusion; delayed packets must be replayed, never applied at a newer time."""

import numpy as np
from navigation.common.contracts import NavigationEstimate, PoseMeasurement
from navigation.common.frames import transform_points
from navigation.fusion.eskf_state import ESKFState
from navigation.fusion.eskf_predict import predict
from navigation.fusion.eskf_update_vision import update_vision
from navigation.fusion.eskf_update_lidar import update_lidar
from navigation.vision.infer_h import HDetector
from navigation.vision.associate_keypoints import associate_keypoints
from navigation.vision.geometric_constraints import valid_keypoints
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
        self.detector = HDetector(self.config.get("checkpoint"))
        self.health = HealthMonitor(
            self.config.get("max_measurement_age", 1.0),
            self.config.get("max_position_std", 5.0),
        )
        self.diagnostics = {}

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
        ok, d2 = update_vision(self.state, measurement)
        self.diagnostics.update(vision_accepted=ok, vision_mahalanobis=d2)
        if ok:
            self.health.record(measurement.timestamp)
        return self.estimate()

    def process_camera(self, frame):
        self.advance(frame.timestamp)
        self.diagnostics["vision_accepted"] = False
        detection = self.detector(frame.image)
        if detection is not None and valid_keypoints(
            detection.keypoints, frame.image.shape
        ):
            match = associate_keypoints(
                self.marker_points,
                detection.keypoints,
                self.calibration,
                self.state.pose,
                detection.ordered,
            )
            if match is not None:
                T, error, _ = match
                self.diagnostics["reprojection_error"] = error
                if error <= self.config.get("max_reprojection_error", 5.0):
                    P = estimate_covariance(
                        self.marker_points,
                        T,
                        self.calibration,
                        self.config.get("pixel_sigma", 1.5),
                    )
                    return self.process_pose(
                        PoseMeasurement(frame.timestamp, T, P, error)
                    )
        return self.estimate()

    def process_lidar(self, frame):
        self.advance(frame.timestamp)
        self.diagnostics["lidar_accepted"] = False
        points = transform_points(self.calibration.T_camera_lidar, frame.points)
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
            ok, d2 = update_lidar(self.state, plane)
            self.diagnostics.update(
                lidar_accepted=ok, lidar_mahalanobis=d2, plane_rms=plane.rms
            )
            # A plane does not observe x/y/yaw: only full pose refreshes global health.
        return self.estimate()

    def estimate(self):
        healthy, health = self.health.check(self.state)
        return NavigationEstimate(
            self.state.timestamp,
            self.state.pose,
            self.state.velocity.copy(),
            self.state.pose_covariance,
            healthy,
            {**self.diagnostics, **health},
        )
