"""Validate generated sensor contracts and exercise the actual training loader."""

import json
from pathlib import Path


def validate_dataset(root, all_images=False, training_device="cpu"):
    import cv2
    import numpy as np
    import torch
    from visual_training.data.h_dataset import HDataset
    from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
    from visual_training.models.losses import segmentation_keypoint_loss
    from visual_training.data.generate_landing_dataset import write_json

    root = Path(root).resolve()
    config = json.loads((root / "plan.json").read_text())["config"]
    report = json.loads((root / "dataset_report.json").read_text())
    seen_paths, seen_sequences, checked_groups, tiles = set(), {}, set(), {}
    counts, checked_images = {}, 0
    for split in ("train", "val", "test"):
        paths = json.loads((root / "splits" / f"{split}.json").read_text())
        if not paths:
            raise ValueError(f"empty {split}")
        counts[split] = len(paths)
        for path_string in paths:
            path = Path(path_string)
            if path_string in seen_paths:
                raise ValueError(f"duplicate label: {path}")
            seen_paths.add(path_string)
            row = json.loads(path.read_text())
            sequence = row["sequence_id"]
            if (
                row["split"] != split
                or seen_sequences.setdefault(sequence, split) != split
            ):
                raise ValueError(f"sortie leaks across splits: {sequence}")
            mount = np.asarray(row["T_body_camera"])
            if not np.allclose(mount[:3, 3], config["camera"]["position_body_m"]):
                raise ValueError(f"camera installation point changed: {path}")
            if not np.allclose(
                np.asarray(row["T_world_deck"]) @ row["T_deck_camera"],
                np.asarray(row["T_world_body"]) @ mount,
                atol=1e-7,
            ):
                raise ValueError(f"inconsistent dynamic camera extrinsics: {path}")
            if row["gimbal"]["timestamp"] != row["timestamp"]:
                raise ValueError(f"unsynchronised gimbal sample: {path}")
            if not np.isfinite(row["keypoints"]).all():
                raise ValueError(f"nonfinite keypoints: {path}")
            image_path, mask_path = root / row["image"], root / row["mask"]
            lidar_path = path.with_name(path.stem + "_lidar.npz")
            for artifact in (image_path, mask_path, lidar_path):
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError(f"missing/empty artifact: {artifact}")
            group = (split, row["weather"], row["stage"])
            if all_images or group not in checked_groups:
                image = cv2.imread(str(image_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                shape = (row["height"], row["width"])
                if (
                    image is None
                    or mask is None
                    or image.shape[:2] != shape
                    or mask.shape != shape
                ):
                    raise ValueError(f"invalid RGB/mask dimensions: {path}")
                if not np.isin(mask, [0, 255]).all():
                    raise ValueError(f"nonbinary segmentation mask: {path}")
                with np.load(lidar_path) as cloud:
                    if (
                        cloud["timestamp"].item() != row["timestamp"]
                        or cloud["points"].ndim != 2
                        or cloud["points"].shape[1] != 3
                        or not np.isfinite(cloud["points"]).all()
                    ):
                        raise ValueError(f"invalid lidar sample: {lidar_path}")
                checked_images += 1
                checked_groups.add(group)
                key = (row["weather"], row["stage"])
                if key not in tiles:
                    tile = cv2.resize(image, (320, 240))
                    cv2.rectangle(tile, (0, 0), (320, 24), (10, 10, 10), -1)
                    cv2.putText(
                        tile,
                        " / ".join(key),
                        (5, 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )
                    tiles[key] = tile
    if sum(counts.values()) != report["frames"]:
        raise ValueError("frame count differs from dataset_report")
    expected_groups = {
        (split, weather, stage)
        for split in counts
        for weather in config["weather"]
        for stage in ("approach", "window_hold", "touchdown")
    }
    if checked_groups != expected_groups:
        raise ValueError("missing weather/stage/split group")
    torch.set_num_threads(1)
    model = HSegmentationKeypoints(channels=4, keypoints=4).to(training_device)
    losses = {}
    for split in counts:
        dataset = HDataset(root / "splits" / f"{split}.json", root, 64)
        sample = dataset[0]
        batch = {
            key: value.unsqueeze(0).to(training_device) for key, value in sample.items()
        }
        model.zero_grad(set_to_none=True)
        loss, _ = segmentation_keypoint_loss(model(batch["image"]), batch)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite training loss: {split}")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError(f"nonfinite training gradients: {split}")
        losses[split] = float(loss.detach())
    sheet = np.vstack(
        [
            np.hstack([tiles[(w, s)] for s in ("approach", "window_hold", "touchdown")])
            for w in config["weather"]
        ]
    )
    if not cv2.imwrite(str(root / "preview.png"), sheet):
        raise OSError("preview write failed")
    validation = dict(
        frames=sum(counts.values()),
        splits=counts,
        image_lidar_samples_decoded=checked_images,
        all_images_decoded=all_images,
        training_device=training_device,
        torch=torch.__version__,
        training_smoke_losses=losses,
    )
    write_json(root / "validation_report.json", validation)
    return validation
