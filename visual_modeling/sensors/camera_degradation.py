import cv2
import numpy as np


def degrade_camera(
    image, rng=None, noise_std=2.0, blur_kernel=1, brightness=1.0, fog=0.0
):
    rng = rng or np.random.default_rng()
    if blur_kernel < 1 or blur_kernel % 2 != 1 or not 0 <= fog <= 1 or noise_std < 0:
        raise ValueError("invalid degradation parameters")
    a = image.astype("float32") * brightness
    if blur_kernel > 1:
        a = cv2.GaussianBlur(a, (blur_kernel, blur_kernel), 0)
    a = a * (1 - fog) + np.array([180, 190, 200]) * fog
    return np.clip(a + rng.normal(0, noise_std, a.shape), 0, 255).astype("uint8")
