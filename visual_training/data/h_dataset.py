import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HDataset(Dataset):
    def __init__(self, split, root="data", image_size=256, augmentation=None):
        self.paths = [Path(p) for p in json.loads(Path(split).read_text())]
        self.root = Path(root)
        self.size = image_size
        self.augmentation = augmentation
        if not self.paths:
            raise ValueError("empty dataset split")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        row = json.loads(self.paths[index].read_text())
        image = cv2.imread(str(self.root / row["image"]))
        mask = cv2.imread(str(self.root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"missing image/mask for {self.paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kp = np.asarray(row["keypoints"], dtype="float32")
        if self.augmentation:
            image, mask, kp = self.augmentation(image, mask, kp)
        h, w = image.shape[:2]
        kp = kp / [w, h]
        image = cv2.resize(image, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return dict(
            image=torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255,
            mask=torch.from_numpy((mask > 127).astype("float32")[None]),
            keypoints=torch.tensor(kp, dtype=torch.float32),
            visibility=torch.tensor(row["visibility"], dtype=torch.bool),
        )
