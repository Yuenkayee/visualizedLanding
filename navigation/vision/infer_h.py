"""Classical white H baseline or learned segmentation + ordered keypoints."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class HDetection:
    keypoints: np.ndarray
    mask: np.ndarray
    confidence: float
    ordered: bool = False


class HDetector:
    def __init__(self, checkpoint=None, device="cpu", threshold=0.5):
        self.model = None
        self.session = None
        self.device = device
        self.threshold = threshold
        self.image_size = 256
        if checkpoint:
            if str(checkpoint).endswith(".onnx"):
                import onnxruntime as ort

                self.session = ort.InferenceSession(
                    str(checkpoint), providers=["CPUExecutionProvider"]
                )
                self.image_size = int(self.session.get_inputs()[0].shape[-1])
            else:
                import torch
                from visual_training.models.h_segmentation_keypoints import (
                    HSegmentationKeypoints,
                )

                payload = torch.load(checkpoint, map_location=device, weights_only=True)
                self.image_size = payload.get("config", {}).get("image_size", 256)
                self.model = HSegmentationKeypoints(
                    **payload.get("model_config", {})
                ).to(device)
                self.model.load_state_dict(payload["model"])
                self.model.eval()

    def __call__(self, image):
        h, w = image.shape[:2]
        if self.model is not None or self.session is not None:
            a = (
                cv2.resize(image, (self.image_size, self.image_size))
                .astype("float32")
                .transpose(2, 0, 1)[None]
                / 255.0
            )
            if self.session is not None:
                logits, keypoints = self.session.run(None, {"image": a})
            else:
                import torch

                with torch.no_grad():
                    out = self.model(torch.from_numpy(a).to(self.device))
                    logits = out["mask_logits"].cpu().numpy()
                    keypoints = out["keypoints"].cpu().numpy()
            prob = 1 / (1 + np.exp(-np.clip(logits[0, 0], -50, 50)))
            if (prob > self.threshold).sum() < 25:
                return None
            mask = cv2.resize(
                (prob > self.threshold).astype("uint8"),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            kp = keypoints[0] * [w, h]
            confidence = float(prob[prob > self.threshold].mean())
            return HDetection(kp, mask, confidence, True)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 75, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 25:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if min(rw, rh) < 4:
                continue
            fill = area / (rw * rh)
            # H is a connected concave shape; reject solid rectangles/deck lines.
            if not 0.25 < fill < 0.86 or max(rw, rh) / min(rw, rh) > 3:
                continue
            if area > score:
                best = contour
                score = area
        if best is None:
            return None
        # Prefer perspective quadrilateral from convex hull; rectangle is fallback.
        hull = cv2.convexHull(best)
        quad = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
        points = (
            quad.reshape(-1, 2).astype(float)
            if len(quad) == 4
            else cv2.boxPoints(cv2.minAreaRect(best)).astype(float)
        )
        isolated = np.zeros((h, w), np.uint8)
        cv2.drawContours(isolated, [best], -1, 1, -1)
        return HDetection(points, isolated, 0.7, False)
