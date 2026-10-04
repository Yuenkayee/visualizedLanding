import numpy as np
import cv2


def valid_keypoints(pixels, image_shape, min_area=25.0):
    p = np.asarray(pixels, dtype=float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return False
    h, w = image_shape[:2]
    return bool(
        (p[:, 0] >= 0).all()
        and (p[:, 0] < w).all()
        and (p[:, 1] >= 0).all()
        and (p[:, 1] < h).all()
        and abs(cv2.contourArea(p.astype("float32"))) > min_area
        and cv2.isContourConvex(p.astype("float32"))
    )
