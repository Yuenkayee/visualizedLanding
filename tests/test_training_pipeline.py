import json
import pytest
from visual_training.data.split_by_sequence import split_by_sequence
from visual_training.data.build_dataset import build_dataset
from visual_training.data.h_dataset import HDataset
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss


def test_generated_dataset_disjoint_and_trainable(tmp_path):
    import torch

    torch.set_num_threads(2)
    build_dataset(tmp_path, sequences=3, frames=2)
    splits = [
        json.loads((tmp_path / "splits" / f"{name}.json").read_text())
        for name in ["train", "val", "test"]
    ]
    groups = [
        {json.loads(open(p).read())["sequence_id"] for p in split} for split in splits
    ]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    sample = HDataset(tmp_path / "splits/train.json", tmp_path, 64)[0]
    target = {k: v.unsqueeze(0) for k, v in sample.items()}
    model = HSegmentationKeypoints(channels=8)
    out = model(target["image"])
    loss, _ = segmentation_keypoint_loss(out, target)
    loss.backward()
    assert torch.isfinite(loss) and out["keypoints"].shape == (1, 4, 2)
    assert model.seg.weight.grad is not None


def test_split_requires_independent_sequences(tmp_path):
    with pytest.raises(ValueError):
        split_by_sequence([], tmp_path)
