import importlib
import json

import pytest
import torch

from visual_training.data.build_dataset import build_dataset
from visual_training.metrics import EpochMetrics
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss

trainer = importlib.import_module("visual_training.train")


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "data"
    build_dataset(root, sequences=3, frames=2)
    torch.manual_seed(9)
    model = HSegmentationKeypoints(channels=4, predict_visibility=True)
    source = tmp_path / "legacy_best.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": model.model_config, "epoch": 17},
        source,
    )
    return {
        "data_root": str(root),
        "output": str(tmp_path / "finetune"),
        "init_checkpoint": str(source),
        "model": model.model_config,
        "image_size": 32,
        "batch_size": 2,
        "epochs": 4,
        "threads": 2,
        "num_workers": 0,
        "device": "cpu",
        "learning_rate": 0.0001,
        "log_every_batches": 0,
        "lr_scheduler": {
            "name": "plateau",
            "patience": 0,
            "factor": 0.5,
            "min_delta": 0.001,
        },
        "early_stopping": {"patience": 8, "min_delta": 0.001},
    }


def test_metrics_use_visible_points_and_global_counts():
    target = {
        "image": torch.zeros(1, 3, 2, 2),
        "mask": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        "keypoints": torch.zeros(1, 3, 2),
        "visibility": torch.tensor([[True, True, False]]),
    }
    prediction = {
        "mask_logits": torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]]),
        "keypoints": torch.tensor([[[0.02, 0.0], [0.06, 0.0], [9.0, 9.0]]]),
        "visibility_logits": torch.tensor([[1.0, -1.0, 1.0]]),
    }
    metrics = EpochMetrics("cpu", 100)
    metrics.update(prediction, target, torch.tensor(2.0), {"bce": torch.tensor(0.5)})
    result = metrics.result()
    assert result["loss"] == 2 and result["bce"] == 0.5
    assert result["mask_iou"] == 0.5
    assert result["keypoint_error_px"] == pytest.approx(4.0)
    assert result["keypoint_pck_2px"] == result["keypoint_pck_5px"] == 0.5
    assert result["visibility_precision"] == result["visibility_recall"] == 0.5
    target["visibility"].zero_()
    empty = EpochMetrics("cpu", 100)
    empty.update(prediction, target, torch.tensor(0.0), {})
    assert empty.result()["keypoint_error_px"] is None


def test_component_logging_preserves_loss_and_gradients():
    torch.set_num_threads(2)
    model = HSegmentationKeypoints(channels=4, predict_visibility=True)
    target = {
        "image": torch.rand(2, 3, 16, 16),
        "mask": torch.zeros(2, 1, 16, 16),
        "keypoints": torch.rand(2, 4, 2),
        "visibility": torch.ones(2, 4, dtype=torch.bool),
    }
    output = model(target["image"])
    loss, tensors = segmentation_keypoint_loss(output, target, metrics_as_tensors=True)
    reference, scalars = segmentation_keypoint_loss(output, target)
    torch.testing.assert_close(loss, reference)
    assert all(not value.requires_grad for value in tensors.values())
    assert {k: float(v) for k, v in tensors.items()} == scalars
    expected = (
        sum(tensors[k] for k in ("bce", "dice", "heatmap", "visibility"))
        + 10 * tensors["keypoint"]
    )
    torch.testing.assert_close(loss.detach(), expected)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_plateau_keeps_legacy_baseline_and_stops(experiment, monkeypatch):
    source = experiment["init_checkpoint"]
    before = torch.load(source, weights_only=True)
    real_epoch = trainer.run_epoch
    validation = iter([0.620, 0.68, 0.69, 0.70])

    def fixed_validation(*args):
        result = real_epoch(*args)
        if args[-1] == "val":
            result["loss"] = next(validation)
        return result

    monkeypatch.setattr(trainer, "run_epoch", fixed_validation)
    experiment["early_stopping"]["patience"] = 3
    best_path = trainer.train(experiment)
    best = torch.load(best_path, weights_only=True)
    last = torch.load(best_path.parent / "last.pt", weights_only=True)
    assert best["epoch"] == 0 and best["best_val_loss"] == 0.620
    assert last["epoch"] == last["bad_epochs"] == 3
    assert last["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0001 / 8)
    assert last["scheduler"]["best"] == 0.620
    for name, value in before["model"].items():
        torch.testing.assert_close(best["model"][name], value)
        torch.testing.assert_close(
            torch.load(source, weights_only=True)["model"][name], value
        )
    assert any(not torch.equal(last["model"][k], v) for k, v in best["model"].items())
    assert (
        json.loads((best_path.parent / "history.json").read_text()) == last["history"]
    )
    with pytest.raises(FileExistsError):
        trainer.train(experiment)
    with pytest.raises(ValueError, match="already early-stopped"):
        trainer.train({"resume": str(best_path.parent / "last.pt"), "epochs": 5})


def test_resume_matches_uninterrupted_training_without_workers(experiment, tmp_path):
    full_path = trainer.train({**experiment, "output": str(tmp_path / "full")})
    part_path = trainer.train(
        {**experiment, "output": str(tmp_path / "part"), "epochs": 2}
    )
    trainer.train({"resume": str(part_path.parent / "last.pt"), "epochs": 4})
    full = torch.load(full_path.parent / "last.pt", weights_only=True)
    resumed = torch.load(part_path.parent / "last.pt", weights_only=True)
    assert resumed["epoch"] == 4
    assert resumed["scheduler"] == full["scheduler"]
    assert resumed["optimizer"]["param_groups"] == full["optimizer"]["param_groups"]
    assert resumed["best_val_loss"] == full["best_val_loss"]
    for name in full["model"]:
        torch.testing.assert_close(
            resumed["model"][name], full["model"][name], rtol=0, atol=0
        )
    for a, b in zip(resumed["history"], full["history"]):
        assert a["val_loss"] == b["val_loss"]
    assert (
        json.loads((part_path.parent / "history.json").read_text())
        == resumed["history"]
    )


def test_finetune_must_be_explicit_and_legacy_resume_is_rejected(experiment):
    with pytest.raises(ValueError, match="requires --init-checkpoint"):
        trainer.train({"require_init_checkpoint": True})
    with pytest.raises(ValueError, match="legacy checkpoint"):
        trainer.train({"resume": experiment["init_checkpoint"]})


def test_resume_recovers_interruption_between_checkpoint_writes(
    experiment, monkeypatch
):
    real_epoch = trainer.run_epoch
    real_save = trainer.atomic_save
    validation = iter([0.62, 0.60, 0.58, 0.56])

    def improving_validation(*args):
        result = real_epoch(*args)
        if args[-1] == "val":
            result["loss"] = next(validation)
        return result

    def interrupt_last(payload, path):
        if path.name == "last.pt" and payload["epoch"] == 2:
            raise OSError("simulated interruption")
        return real_save(payload, path)

    monkeypatch.setattr(trainer, "run_epoch", improving_validation)
    monkeypatch.setattr(trainer, "atomic_save", interrupt_last)
    with pytest.raises(OSError, match="simulated interruption"):
        trainer.train(experiment)
    monkeypatch.setattr(trainer, "atomic_save", real_save)
    last_path = f"{experiment['output']}/last.pt"
    assert torch.load(last_path, weights_only=True)["epoch"] == 1
    best_path = trainer.train({"resume": last_path, "epochs": 3})
    state = torch.load(best_path, weights_only=True)
    assert [row["epoch"] for row in state["history"]] == [0, 1, 2, 3]
    assert state["best_val_loss"] == 0.56
    assert torch.load(last_path, weights_only=True)["epoch"] == 3


def test_nonfinite_epoch_is_rejected():
    model = HSegmentationKeypoints(channels=4)
    with torch.no_grad():
        model.seg.weight.fill_(float("nan"))
    sample = {
        "image": torch.zeros(1, 3, 16, 16),
        "mask": torch.zeros(1, 1, 16, 16),
        "keypoints": torch.zeros(1, 4, 2),
        "visibility": torch.ones(1, 4, dtype=torch.bool),
    }
    with pytest.raises(FloatingPointError, match="non-finite val"):
        trainer.run_epoch(
            model,
            [sample],
            None,
            None,
            torch.device("cpu"),
            "fp32",
            {"image_size": 16},
            0,
            "val",
        )
