"""Classical white H baseline or learned segmentation + ordered keypoints."""

from dataclasses import dataclass, replace
import cv2
import numpy as np


@dataclass
class HDetection:
    keypoints: np.ndarray
    mask: np.ndarray
    confidence: float
    ordered: bool = False
    visibility: np.ndarray | None = None
    network_mask: np.ndarray | None = None
    # Native mask pixels -> original camera pixels, including a possible ROI.
    network_scale: np.ndarray | None = None
    network_origin: np.ndarray | None = None

    def mask_coordinates(self, image_shape):
        if self.network_mask is None:
            raise ValueError("No network mask")
        scale = (np.array(image_shape[1::-1], float) / self.network_mask.shape[::-1]
                 if self.network_scale is None else np.asarray(self.network_scale, float))
        origin = np.zeros(2) if self.network_origin is None else np.asarray(self.network_origin, float)
        if scale.shape != (2,) or origin.shape != (2,) or not np.isfinite([scale, origin]).all() or (scale <= 0).any():
            raise ValueError("Invalid network-to-image coordinates")
        return scale, origin


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

    def detect_roi(self, image, detection, config):
        """Retry a learned target at higher scale, using only its current mask.

        Keep the native crop mask and its affine pixel mapping. Camera intrinsics
        stay unchanged: ALL returned landmarks are in original image pixels.
        """
        if self.model is None and self.session is None:
            return None, dict(status="not_learned")
        if detection is None or detection.network_mask is None:
            return None, dict(status="no_target_mask")
        mask = np.asarray(detection.network_mask, np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, dict(status="empty_mask")
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < config.get("roi_minimum_mask_area_px", 9):
            return None, dict(status="small_mask")
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x+w == mask.shape[1] or y+h == mask.shape[0]:
            return None, dict(status="truncated_mask")
        scale, origin = detection.mask_coordinates(image.shape)
        center = np.array([x+w/2, y+h/2])*scale + origin
        height, width = image.shape[:2]
        aspect = width / height
        context = float(config.get("roi_context_scale", 2.5))
        minimum = float(config.get("roi_minimum_width_px", 256))
        if not np.isfinite([context, minimum]).all() or context <= 1 or minimum <= 0:
            raise ValueError("ROI context must exceed 1 and minimum width must be positive")
        cw = min(width, max(minimum, w*scale[0]*context, h*scale[1]*context*aspect))
        ch = cw/aspect
        if cw*ch >= width*height*config.get("roi_maximum_area_fraction", .9):
            return None, dict(status="no_magnification")
        left = int(np.clip(center[0]-cw/2, 0, width-cw))
        top = int(np.clip(center[1]-ch/2, 0, height-ch))
        right, bottom = min(width, left+round(cw)), min(height, top+round(ch))
        box = [left, top, right, bottom]
        cropped = self(image[top:bottom, left:right])
        if cropped is None:
            return None, dict(status="no_target", box=box)
        full_mask = np.zeros((height, width), np.uint8)
        full_mask[top:bottom, left:right] = cropped.mask
        return replace(cropped, keypoints=cropped.keypoints+[left, top], mask=full_mask,
                       network_scale=np.array([right-left, bottom-top])/cropped.network_mask.shape[::-1],
                       network_origin=np.array([left, top])), dict(status="evaluated", box=box)

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
                values = dict(zip([o.name for o in self.session.get_outputs()], self.session.run(None, {"image": a})))
                logits, keypoints = values["mask_logits"], values["keypoints"]
                visibility = values.get("visibility")
            else:
                import torch

                with torch.no_grad():
                    out = self.model(torch.from_numpy(a).to(self.device))
                    logits = out["mask_logits"].cpu().numpy()
                    keypoints = out["keypoints"].cpu().numpy()
                    visibility = out["visibility_logits"].sigmoid().cpu().numpy() if "visibility_logits" in out else None
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
            return HDetection(kp, mask, confidence, True,
                              None if visibility is None else visibility[0],
                              (prob > self.threshold).astype(np.uint8))
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
