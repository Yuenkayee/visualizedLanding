"""Approximate lens/sensor effects; spatial warps are applied to labels too."""

import cv2
import numpy as np
from navigation.pose.solve_pnp import project_points


def _pixel_motion(points, pose, following_pose, calibration):
    if following_pose is None:
        return np.zeros((len(points), 2))
    return project_points(points, following_pose, calibration) - project_points(
        points, pose, calibration
    )


def apply_landing_effects(
    image,
    mask,
    annotation,
    weather,
    camera,
    rng,
    pose,
    following_pose=None,
    dt=1.0,
    calibration=None,
):
    image = image.astype("float32")
    mask = mask.copy()
    kp = np.asarray(annotation["keypoints"], float)
    visible = np.array(annotation["visibility"], bool)
    h, w = mask.shape
    points = np.asarray(annotation["points_deck"])
    shift = (
        _pixel_motion(points, pose, following_pose, calibration)
        if calibration is not None
        else np.zeros_like(kp)
    )
    dt = max(dt, 1e-6)
    # A planar homography over the H/deck models first-order rolling readout.
    readout = camera.get("readout_s", 0.0)
    if readout > 0 and following_pose is not None and np.isfinite(shift).all():
        H, _ = cv2.findHomography(
            kp.astype("float32"), (kp + shift).astype("float32"), 0
        )
        if H is not None:
            yy, xx = np.indices((h, w), dtype="float32")
            v = np.stack([xx, yy, np.ones_like(xx)], -1) @ H.T
            denom = np.where(np.abs(v[:, :, 2]) < 1e-6, 1e-6, v[:, :, 2])
            delta = v[:, :, :2] / denom[:, :, None] - np.stack([xx, yy], -1)
            delta = np.clip(
                delta * readout / dt * ((yy / h - 0.5)[:, :, None]), -30, 30
            )
            mx = (xx - delta[:, :, 0]).astype("float32")
            my = (yy - delta[:, :, 1]).astype("float32")
            image = cv2.remap(
                image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            mask = cv2.remap(
                mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
            )
            kp += np.clip(shift * readout / dt * (kp[:, 1, None] / h - 0.5), -30, 30)
    motion = np.median(shift, axis=0) * camera.get("exposure_s", 0.0) / dt
    motion = np.clip(motion, -15, 15)
    if np.linalg.norm(motion) > 0.3:
        acc = np.zeros_like(image)
        for fraction in np.linspace(-0.5, 0.5, 5):
            matrix = np.float32(
                [[1, 0, motion[0] * fraction], [0, 1, motion[1] * fraction]]
            )
            acc += (
                cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
                / 5
            )
        image = acc
    glare = float(weather.get("glare", 0))
    if glare > 0:
        # Saturated water reflections/daylight and deck lights/night bloom.
        highlights = np.maximum(image.max(2) - 180, 0) / 75
        bloom = cv2.GaussianBlur(highlights, (0, 0), max(w * 0.018, 1))
        image += bloom[:, :, None] * glare * 220
        # Lens ghost from strongest luminous area, not arbitrary label-following.
        y, x = np.unravel_index(
            np.argmax(cv2.GaussianBlur(image.max(2), (0, 0), 4)), (h, w)
        )
        yy, xx = np.indices((h, w))
        strength = float(image[y, x].max() / 255)
        for f, r in [(0.35, 0.05), (0.75, 0.035), (1.2, 0.02)]:
            cx = x + f * (w / 2 - x)
            cy = y + f * (h / 2 - y)
            ring = np.exp(
                -(((np.hypot(xx - cx, yy - cy) - w * r) / (w * 0.008 + 1)) ** 2)
            )
            image += (
                ring[:, :, None] * np.array([35, 25, 12]) * glare * min(strength, 1)
            )
    image *= weather.get("exposure_gain", 1.0)
    noise = weather.get("sensor_noise", 1.5)
    # Read noise plus signal-dependent shot noise (phenomenological).
    sigma = noise + np.sqrt(np.maximum(image, 0)) * 0.18
    image += rng.normal(0, 1, image.shape) * sigma
    visible &= (kp[:, 0] >= 0) & (kp[:, 0] < w) & (kp[:, 1] >= 0) & (kp[:, 1] < h)
    annotation["keypoints"] = kp.tolist()
    annotation["visibility"] = visible.tolist()
    annotation["sensor_effects"] = dict(
        exposure_s=camera.get("exposure_s", 0),
        readout_s=readout,
        motion_blur_vector_px=motion.tolist(),
        glare=glare,
        noise_std=noise,
        label_time="mid_exposure; planar rolling approximation",
    )
    return np.clip(image, 0, 255).astype("uint8"), mask
