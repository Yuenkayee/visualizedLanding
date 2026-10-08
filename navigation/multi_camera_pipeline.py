"""Synchronized cameras -> gated best observation -> one fixed-reference ESKF."""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from navigation.pipeline import NavigationPipeline
from navigation.common.contracts import PoseMeasurement
from navigation.vision.geometric_constraints import valid_keypoints
from navigation.vision.associate_keypoints import associate_keypoints
from navigation.pose.estimate_covariance import estimate_covariance
from navigation.fusion.measurement_gating import mahalanobis_gate


class MultiCameraPipeline(NavigationPipeline):
    def __init__(self, rig, marker_points, config=None, state=None):
        self.rig = rig
        super().__init__(rig.reference.calibration, marker_points, config, state)
        self.active_camera_id = None
        self.last_bundle_timestamp = None
        if self.config.get("switch_score_ratio", 1.2) < 1:
            raise ValueError("switch_score_ratio must be at least one")

    def process_camera_bundle(self, frames):
        frames = dict(frames)
        if not frames or not set(frames) <= set(self.rig.cameras):
            raise ValueError("empty bundle or unknown camera id")
        times = [f.timestamp for f in frames.values()]
        if max(times) - min(times) > 1e-8:
            raise ValueError(
                "camera bundle must share one exposure timestamp; align asynchronous frames first"
            )
        t = times[0]
        if self.last_bundle_timestamp is not None and t <= self.last_bundle_timestamp:
            raise ValueError("duplicate or stale camera bundle")
        for key, frame in frames.items():
            cal = self.rig.cameras[key].calibration
            if frame.image.shape[:2] != (cal.height, cal.width):
                raise ValueError(f"image size differs from calibration: {key}")
        self.advance(t)
        self.last_bundle_timestamp = t
        candidates, diagnostics = [], {}
        for key, frame in frames.items():
            camera = self.rig.cameras[key]
            detection = self.detector(frame.image)
            d = diagnostics[key] = dict(usable=False)
            if detection is None:
                d["reason"] = "no_detection"
                continue
            visibility = detection.keypoint_visibility
            if visibility is not None and not np.asarray(visibility).all():
                d["reason"] = "incomplete_keypoints"
                continue
            if not valid_keypoints(detection.keypoints, frame.image.shape):
                d["reason"] = "invalid_geometry"
                continue
            prior = self.state.pose @ self.rig.reference_to_camera(key)
            match = associate_keypoints(
                self.marker_points,
                detection.keypoints,
                camera.calibration,
                prior,
                detection.ordered,
            )
            if match is None:
                d["reason"] = "pnp_failed"
                continue
            pose, error, _ = match
            d["reprojection_error"] = float(error)
            if error > self.config.get("max_reprojection_error", 5):
                d["reason"] = "reprojection_gate"
                continue
            P = estimate_covariance(
                self.marker_points,
                pose,
                camera.calibration,
                self.config.get("pixel_sigma", 1.5),
            )
            m = self.rig.measurement_to_reference(
                key, PoseMeasurement(t, pose, P, error)
            )
            # Gate every candidate before selection. A low-error outlier must not
            # suppress a different camera's valid update.
            residual = np.r_[
                m.T_deck_camera[:3, 3] - self.state.position,
                Rotation.from_matrix(
                    self.state.rotation.T @ m.T_deck_camera[:3, :3]
                ).as_rotvec(),
            ]
            ok, d2 = mahalanobis_gate(
                residual, self.state.pose_covariance + m.covariance
            )
            d["mahalanobis"] = d2
            if not ok:
                d["reason"] = "innovation_gate"
                continue
            extent = float(min(np.ptp(detection.keypoints, axis=0)))
            gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            uncertainty = np.sqrt(max(np.trace(m.covariance[:3, :3]), 1e-8))
            score = (
                max(detection.confidence, 1e-4)
                * min(extent / 80, 2)
                * (0.5 + 0.5 * min(sharpness / 50, 1))
                / ((1 + error) * (0.1 + uncertainty))
            )
            d.update(usable=True, score=float(score), target_extent_px=extent)
            candidates.append((score, key, m))
        selected = None
        if candidates:
            selected = max(candidates, key=lambda c: c[0])
            current = next(
                (c for c in candidates if c[1] == self.active_camera_id), None
            )
            if current is not None and selected[0] <= current[0] * self.config.get(
                "switch_score_ratio", 1.2
            ):
                selected = current
        previous = self.active_camera_id
        self.diagnostics.update(
            vision_accepted=False,
            cameras=diagnostics,
            selected_camera_id=selected[1] if selected else None,
            camera_switched=False,
            reference_camera_id=self.rig.reference_camera_id,
        )
        if selected:
            result = self.process_pose(selected[2])
            if result.diagnostics.get("vision_accepted"):
                self.active_camera_id = selected[1]
                self.diagnostics["camera_switched"] = (
                    previous is not None and previous != selected[1]
                )
        return self.estimate()
