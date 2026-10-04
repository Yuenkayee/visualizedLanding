"""Photometric augmentation preserves ordered landmark geometry and visibility."""

import numpy as np
from visual_modeling.sensors.camera_degradation import degrade_camera


class PhotometricAugmentation:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, image, mask, keypoints):
        return (
            degrade_camera(
                image,
                self.rng,
                noise_std=self.rng.uniform(0, 5),
                blur_kernel=int(self.rng.choice([1, 1, 3])),
                brightness=self.rng.uniform(0.7, 1.2),
                fog=self.rng.uniform(0, 0.1),
            ),
            mask,
            keypoints,
        )
