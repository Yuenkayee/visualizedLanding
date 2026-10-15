import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from visual_training.data.manifest import load_manifest


class HDataset(Dataset):
    def __init__(self, split, root="data", image_size=256, augmentation=None, weathers=None):
        records = load_manifest(split, root, weathers)
        self.paths, self.rows = map(list, zip(*records))
        self.root = Path(root)
        self.size = image_size
        self.augmentation = augmentation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.root / row["image"]))
        mask = cv2.imread(str(self.root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"missing image/mask for {self.paths[index]}")
        h, w = image.shape[:2]
        kp = np.asarray(row["keypoints"], dtype="float32") / [w, h] * self.size
        visible = np.asarray(row.get("keypoint_train_visibility", row["visibility"]), bool)
        # Resize first: augmentation works in input pixels, avoiding float64
        # intermediates over every 1280x960 source image in each worker.
        image = cv2.cvtColor(cv2.resize(image, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augmentation:
            image, mask, kp, visible = self.augmentation(image, mask, kp, visible)
        return dict(
            image=torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255,
            mask=torch.from_numpy((mask > 127).astype("float32")[None]),
            keypoints=torch.tensor(kp / self.size, dtype=torch.float32),
            visibility=torch.tensor(visible, dtype=torch.bool),
            index=torch.tensor(index),
        )
