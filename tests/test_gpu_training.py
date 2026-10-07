import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml
from torch.utils.data import Dataset

from visual_training.check_gpu import check_gpu
from visual_training.data.augmentations import PhotometricAugmentation
from visual_training.data.build_dataset import build_dataset
from visual_training.data.h_dataset import HDataset
from visual_training.evaluate import evaluate
from visual_training.export_model import export_model
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss
from visual_training.train import make_loader, train, training_device


@pytest.mark.parametrize("visible", [True, False])
def test_bf16_coordinates_loss_and_gradients_remain_finite(visible):
    torch.set_num_threads(2)
    model = HSegmentationKeypoints(channels=4, predict_visibility=True)
    target = dict(
        image=torch.rand(2, 3, 64, 64),
        mask=torch.zeros(2, 1, 64, 64),
        keypoints=torch.full((2, 4, 2), 0.45),
        visibility=torch.full((2, 4), visible, dtype=torch.bool),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(target["image"])
    assert output["mask_logits"].dtype == torch.bfloat16
    assert output["keypoints"].dtype == torch.float32
    loss, _ = segmentation_keypoint_loss(output, target, return_metrics=False)
    reference, metrics = segmentation_keypoint_loss(output, target)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    torch.testing.assert_close(loss, reference)
    if not visible:
        assert metrics["heatmap"] == metrics["keypoint"] == 0
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_fp16_large_mask_loss_does_not_overflow():
    output = dict(
        mask_logits=torch.zeros(
            1, 1, 512, 512, dtype=torch.float16, requires_grad=True
        ),
        heatmap_logits=torch.zeros(
            1, 4, 128, 128, dtype=torch.float16, requires_grad=True
        ),
        keypoints=torch.full((1, 4, 2), 0.5, requires_grad=True),
    )
    target = dict(
        mask=torch.ones(1, 1, 512, 512),
        keypoints=torch.full((1, 4, 2), 0.45),
        visibility=torch.ones(1, 4, dtype=torch.bool),
    )
    loss, _ = segmentation_keypoint_loss(output, target, return_metrics=False)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(torch.isfinite(value.grad).all() for value in output.values())


@pytest.mark.parametrize("relative", [False, True])
def test_dataset_transfer_preserves_split_membership(tmp_path, relative):
    original = tmp_path / "mac_data"
    copied = tmp_path / "server_data"
    build_dataset(original, sequences=3, frames=2)
    if relative:
        for split in (original / "splits").glob("*.json"):
            split.write_text(
                json.dumps(
                    [
                        str(Path(p).relative_to(original))
                        for p in json.loads(split.read_text())
                    ]
                )
            )
    shutil.copytree(original, copied)
    for split in (copied / "splits").glob("*.json"):
        before = split.read_text()
        dataset = HDataset(split, copied, 32)
        assert all(p.is_relative_to(copied / "annotations") for p in dataset.paths)
        assert dataset[0]["image"].shape == (3, 32, 32)
        assert split.read_text() == before
    original.rename(tmp_path / "archived_data")
    assert HDataset(copied / "splits/train.json", copied, 32)[0]["mask"].shape == (
        1,
        32,
        32,
    )


class IdenticalImages(Dataset):
    def __init__(self):
        self.augmentation = PhotometricAugmentation(42)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        image, _, _ = self.augmentation(
            np.full((32, 32, 3), 128, dtype=np.uint8), None, None
        )
        return torch.from_numpy(image.copy())


def test_workers_have_independent_reproducible_augmentation():
    config = dict(
        batch_size=1, num_workers=2, persistent_workers=False, prefetch_factor=1
    )
    first = list(make_loader(IdenticalImages(), config, False, 42))
    second = list(make_loader(IdenticalImages(), config, False, 42))
    assert not torch.equal(first[0], first[1])
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_bf16_training_checkpoint_evaluation_and_onnx(tmp_path):
    root = tmp_path / "data"
    build_dataset(root, sequences=3, frames=2)
    checkpoint = train(
        dict(
            data_root=str(root),
            output=str(tmp_path / "weights"),
            image_size=32,
            batch_size=2,
            epochs=1,
            threads=2,
            device="cpu",
            precision="bf16",
            num_workers=2,
            multiprocessing_context="spawn",
            persistent_workers=True,
            prefetch_factor=1,
            pin_memory=False,
            channels_last=True,
            model=dict(channels=4, predict_visibility=True),
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 1 and payload["config"]["precision"] == "bf16"
    result = evaluate(checkpoint, root / "splits/test.json", root, image_size=32)
    assert 0 <= result["mask_iou"] <= 1
    model = HSegmentationKeypoints(**payload["model_config"])
    model.load_state_dict(payload["model"])
    model.eval()
    session = ort.InferenceSession(
        str(export_model(checkpoint, tmp_path / "model.onnx", 32)),
        providers=["CPUExecutionProvider"],
    )
    image = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        output = model(image)
    exported = session.run(None, {"image": image.numpy()})
    for name, actual in zip(
        ("mask_logits", "keypoints", "visibility_logits"), exported
    ):
        np.testing.assert_allclose(actual, output[name].numpy(), rtol=1e-4, atol=1e-5)


def test_cuda_missing_and_unsupported_precision_fail_early(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requirements-rtx5090.txt"):
        training_device(dict(device="cuda:0", precision="bf16"))
    with pytest.raises(ValueError, match="FP16 requires CUDA"):
        training_device(dict(device="cpu", precision="fp16"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Requires an RTX 5090 CUDA host",
)
def test_rtx5090_real_training_step():
    config = yaml.safe_load(
        Path("configs/multi_camera_training_rtx5090.yaml").read_text()
    )
    assert check_gpu(config)["status"] == "ok"
