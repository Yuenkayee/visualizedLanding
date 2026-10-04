import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from visual_training.data.h_dataset import HDataset
from visual_training.data.augmentations import PhotometricAugmentation
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss


def train(config):
    c = config
    seed = c.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(c.get("threads", 4))
    device = torch.device(c.get("device", "cpu"))
    root = Path(c.get("data_root", "data"))
    out = Path(c.get("output", "visual_training/checkpoints"))
    out.mkdir(parents=True, exist_ok=True)
    size = c.get("image_size", 256)
    batch = c.get("batch_size", 8)
    train_set = HDataset(
        root / "splits/train.json", root, size, PhotometricAugmentation(seed)
    )
    val_set = HDataset(root / "splits/val.json", root, size)
    loaders = [
        DataLoader(d, batch_size=batch, shuffle=i == 0, num_workers=0)
        for i, d in enumerate((train_set, val_set))
    ]
    model = HSegmentationKeypoints(**c.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=c.get("learning_rate", 0.001), weight_decay=1e-4
    )
    best = float("inf")
    history = []
    for epoch in range(c.get("epochs", 20)):
        losses = []
        for phase, loader in enumerate(loaders):
            model.train(phase == 0)
            total = 0.0
            count = 0
            with torch.set_grad_enabled(phase == 0):
                for sample in loader:
                    sample = {k: v.to(device) for k, v in sample.items()}
                    prediction = model(sample["image"])
                    loss, _ = segmentation_keypoint_loss(prediction, sample)
                    if phase == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    n = sample["image"].shape[0]
                    total += float(loss.detach()) * n
                    count += n
            losses.append(total / count)
        record = dict(epoch=epoch + 1, train_loss=losses[0], val_loss=losses[1])
        history.append(record)
        print(json.dumps(record), flush=True)
        payload = dict(
            model=model.state_dict(),
            model_config=model.model_config,
            epoch=epoch + 1,
            optimizer=optimizer.state_dict(),
            config=c,
        )
        torch.save(payload, out / "last.pt")
        if losses[1] < best:
            best = losses[1]
            torch.save(payload, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return out / "best.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--data-root")
    p.add_argument("--output")
    p.add_argument("--image-size", type=int)
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    for name in ("epochs", "data_root", "output", "image_size"):
        if getattr(a, name) is not None:
            c[name] = getattr(a, name)
    train(c)


if __name__ == "__main__":
    main()
