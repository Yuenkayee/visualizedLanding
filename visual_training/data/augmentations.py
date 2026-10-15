"""Input-resolution augmentation with joint geometry and observability labels."""
import cv2
import numpy as np


class PhotometricAugmentation:
    def __init__(self, seed=None, difficult=False):
        self.rng = np.random.default_rng(seed)
        self.difficult = difficult

    def __call__(self, image, mask, keypoints, visibility):
        rng = self.rng
        h, w = image.shape[:2]
        keypoints, visibility = keypoints.copy(), visibility.copy()
        if self.difficult and rng.random() < .30:
            scale = rng.uniform(.75, 1.3)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), scale)
            M[:, 2] += rng.uniform(-.1, .1, 2) * [w, h]
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
            # Reflected padding is not labelled background: replace it explicitly.
            support = cv2.warpAffine(np.ones((h, w), np.uint8), M, (w, h))
            image[support == 0] = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
            keypoints = keypoints @ M[:, :2].T + M[:, 2]
            visibility &= ((keypoints >= 0) & (keypoints < [w, h])).all(-1)
        if self.difficult and rng.random() < .12:
            center = keypoints[int(rng.integers(4))]
            half = rng.uniform(.025, .10, 2) * [w, h]
            lo = np.clip(np.floor(center - half), [0, 0], [w, h]).astype(int)
            hi = np.clip(np.ceil(center + half), [0, 0], [w, h]).astype(int)
            image[lo[1]:hi[1], lo[0]:hi[0]] = rng.integers(20, 140, 3, dtype=np.uint8)
            mask[lo[1]:hi[1], lo[0]:hi[0]] = 0
            visibility &= ~((keypoints >= lo) & (keypoints < hi)).all(-1)
        if self.difficult and rng.random() < .04:
            # Explicit target-absent examples, including one distracting deck line.
            image = np.full_like(image, int(rng.integers(15, 100)))
            cv2.line(image, (0, int(rng.integers(h))), (w-1, int(rng.integers(h))),
                     (180, 180, 180), max(1, w // 100))
            mask = np.zeros_like(mask)
            visibility[:] = False
        x = image.astype(np.float32) * np.float32(rng.uniform(.7, 1.2))
        fog = np.float32(rng.uniform(0, .1))
        x = x * (1 - fog) + np.float32(200) * fog
        x += rng.standard_normal(x.shape, dtype=np.float32) * np.float32(rng.uniform(0, 5))
        if rng.random() < 1 / 3:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        return np.clip(x, 0, 255).astype(np.uint8), mask, keypoints, visibility
