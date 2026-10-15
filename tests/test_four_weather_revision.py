import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from visual_training.data.manifest import WEATHERS, load_manifest
from visual_training.data.prepare_four_weather import prepare
from visual_training.data.h_dataset import HDataset
from visual_training.data.augmentations import PhotometricAugmentation
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss
from navigation.vision.infer_h import HDetection
from navigation.vision.mask_geometry import recover_from_mask, outer_quad
from navigation.common.contracts import CameraCalibration
from navigation.common.frames import transform
from navigation.pose.solve_pnp import project_points, solve_pnp
from visual_modeling.blender.build_h_marker import marker_points, marker_polygons


@pytest.fixture
def weather_data(tmp_path):
    root = tmp_path / "old"
    (root / "splits").mkdir(parents=True)
    (root / "trajectories").mkdir()
    for split in ("train", "val", "test"):
        paths = []
        for weather in (*WEATHERS, "glare"):
            seq = f"{weather}_{split}"
            path = root / "annotations" / seq / "000000.json"
            path.parent.mkdir(parents=True)
            image = np.full((32, 32, 3), 80, np.uint8)
            mask = np.zeros((32, 32), np.uint8)
            mask[8:24, 8:24] = 255
            cv2.imwrite(str(root / f"{seq}.png"), image)
            cv2.imwrite(str(root / f"{seq}_mask.png"), mask)
            row = dict(sequence_id=seq, weather=weather, split=split, stage="touchdown",
                       image=f"{seq}.png", mask=f"{seq}_mask.png", keypoints=[[8,8],[24,8],[24,24],[8,24]], visibility=[True]*4)
            path.write_text(json.dumps(row))
            np.savez(path.with_name("000000_lidar.npz"), points=np.ones((10,3)), timestamp=0.)
            (root / "trajectories" / f"{seq}.json").write_text("{}")
            # Intentionally stale absolute server paths, like the downloaded data.
            paths.append(f"/server/old/annotations/{seq}/000000.json")
        (root / "splits" / f"{split}.json").write_text(json.dumps(paths))
    return root


def test_portable_four_weather_preparation_is_nondestructive(weather_data, tmp_path):
    target = tmp_path / "four"
    original = (weather_data / "splits/train.json").read_bytes()
    report = prepare(weather_data, target, "test")
    assert report["counts"] == dict(train=4, val=4, test=4)
    assert not target.exists()
    prepare(weather_data, target, "full")
    assert (target / "SUCCESS.json").exists()
    assert (weather_data / "splits/train.json").read_bytes() == original
    records = load_manifest(target / "splits/train.json", target)
    assert {r["weather"] for _, r in records} == set(WEATHERS)
    assert all(not Path(p).is_absolute() for p in json.loads((target / "splits/train.json").read_text()))
    assert len(HDataset(weather_data / "splits/train.json", weather_data, weathers=WEATHERS)) == 4
    with pytest.raises(FileExistsError):
        prepare(weather_data, target, "full")
    (target / "SUCCESS.json").unlink()
    from visual_training.server_train import validate_data
    with pytest.raises(ValueError, match="incomplete"):
        validate_data(target)


def test_new_heads_soft_labels_and_missing_targets_backpropagate():
    torch.set_num_threads(1)
    model = HSegmentationKeypoints(channels=4, context=True, visibility_head=True)
    image = torch.rand(2, 3, 32, 32)
    prediction = model(image)
    truth = dict(mask=torch.zeros(2, 1, 32, 32), keypoints=torch.tensor([[[.2,.2],[.8,.2],[.8,.8],[.2,.8]], [[-1.,2.]]*4]),
                 visibility=torch.tensor([[True]*4, [False]*4]))
    loss, terms = segmentation_keypoint_loss(prediction, truth, heatmap_sigma=1., coordinate_beta=.01)
    loss.backward()
    assert np.isfinite(list(terms.values())).all()
    assert model.visibility[-1].weight.grad.abs().sum() > 0
    assert model.kp.weight.grad.abs().sum() > 0
    # Invisible coordinates do not alter heatmap/coordinate supervision.
    truth["keypoints"][1] = 123.
    other, _ = segmentation_keypoint_loss(prediction, truth, heatmap_sigma=1., coordinate_beta=.01)
    assert torch.allclose(loss, other)


def test_occlusion_and_negative_labels_stay_aligned():
    aug = PhotometricAugmentation(42, difficult=True)
    image = np.full((64,64,3), 100, np.uint8)
    mask = np.zeros((64,64), np.uint8)
    mask[8:56,8:56] = 255
    points = np.array([[8,8],[56,8],[56,56],[8,56]], float)
    negatives = partial = 0
    for _ in range(150):
        _, m, p, v = aug(image.copy(), mask.copy(), points, np.ones(4, bool))
        assert np.all(((p >= 0) & (p < 64)).all(-1)[v])
        if not m.any():
            assert not v.any()
            negatives += 1
        partial += not v.all()
    assert negatives > 0 and partial > 0


def test_mask_recovers_outer_corners_but_rejects_clipped_or_solid_shapes():
    calibration = CameraCalibration(np.array([[260.,0,256],[0,260,256],[0,0,1]]), 512,512)
    pose = transform(np.diag([1.,-1,-1]), [0.,0.,6.])
    points = marker_points()
    pixels = project_points(points, pose, calibration)
    mask = np.zeros((512,512), np.uint8)
    for polygon in marker_polygons():
        cv2.fillConvexPoly(mask, np.rint(project_points(polygon, pose, calibration)-.5).astype(np.int32), 1)
    corrupted = pixels.copy()
    corrupted[2] = .6*pixels[2] + .4*pixels[0]
    det = HDetection(corrupted, mask, .999, True, network_mask=mask)
    recovered = recover_from_mask(det, (512,512,3), points, calibration, pose, {})
    assert recovered is not None
    assert np.max(np.linalg.norm(recovered[0]-pixels, axis=1)) < 2
    assert recovered[2] > 50 and recovered[1] > .9
    assert outer_quad(mask[:,150:]) is None  # Clipped rail must not invent a corner.
    solid = np.zeros_like(mask)
    cv2.fillConvexPoly(solid, np.rint(pixels).astype(np.int32), 1)
    det.network_mask = solid
    assert recover_from_mask(det, (512,512,3), points, calibration, pose, {}) is None


def test_ippe_refinement_keeps_cheirality_and_reduces_residual():
    c = CameraCalibration(np.array([[250.,0,320],[0,250,240],[0,0,1]]), 640,480)
    pose = transform(np.diag([1.,-1,-1]), [.5,-.2,5.])
    x = marker_points()
    pixels = project_points(x,pose,c) + np.random.default_rng(5).normal(0,.8,(4,2))
    raw = solve_pnp(x,pixels,c)
    refined = solve_pnp(x,pixels,c,refine=True)
    assert len(refined) == len(raw) == 2
    assert min(e for _,e in refined) <= min(e for _,e in raw)
    assert all(T[2,3]>0 for T,_ in refined)


def test_new_training_outputs_reports_and_legacy_checkpoint_still_loads(weather_data, tmp_path):
    from visual_training.train import train
    from navigation.vision.infer_h import HDetector
    c = dict(data_root=str(weather_data), output=str(tmp_path / "weights"), weathers=list(WEATHERS),
             log_dir=str(tmp_path / "logs"), epochs=2, batch_size=2, image_size=32, num_workers=0, threads=1,
             model=dict(channels=4, context=True, visibility_head=True), difficult_augmentation=True,
             touchdown_sampling_weight=1.5, loss=dict(heatmap_sigma=1.), max_train_batches=1, max_val_batches=2,
             scheduler=dict(mode="min", factor=.5, patience=0, min_lr=1e-5), early_stopping_patience=1)
    checkpoint = train(c)
    model = HDetector(checkpoint)
    assert model.model.visibility is not None and model.image_size == 32
    for name in ("best_loss.pt", "best_keypoint.pt", "metrics/analysis.md", "metrics/training_curves.png"):
        assert (checkpoint.parent / name).exists()
    history = json.loads((checkpoint.parent / "history.json").read_text())
    assert "weather/glare" not in history[-1]["keypoint_metrics"]["by_condition"]
    legacy = HSegmentationKeypoints(channels=4)
    old = tmp_path / "old.pt"
    torch.save(dict(model=legacy.state_dict(), model_config=dict(channels=4,keypoints=4)),old)
    assert HDetector(old).model.visibility is None


def test_export_visibility_contract(tmp_path):
    from visual_training.export_model import export_model
    from navigation.vision.infer_h import HDetector
    model = HSegmentationKeypoints(channels=4, visibility_head=True, context=True)
    checkpoint = tmp_path / "new.pt"
    torch.save(dict(model=model.state_dict(),model_config=model.model_config,config=dict(image_size=32)),checkpoint)
    output = export_model(checkpoint,tmp_path / "model.onnx")
    detector = HDetector(output)
    assert detector.image_size == 32
    # Compare a textured input: a constant zero image makes tiny-variance
    # GroupNorm reductions unusually sensitive to runtime rounding.
    image = np.random.default_rng(5).random((1,3,32,32), dtype=np.float32)
    values = detector.session.run(None, {"image": image})
    assert [v.shape for v in values] == [(1,1,32,32),(1,4,2),(1,4)]
    model.eval()
    with torch.no_grad():
        reference = model(torch.from_numpy(image))
    for actual, expected in zip(values, (reference["mask_logits"], reference["keypoints"], reference["visibility_logits"].sigmoid())):
        assert np.allclose(actual, expected.numpy(), atol=1e-5)


def test_gimbal_variation_is_reproducible_and_keeps_fixed_mount(tmp_path):
    from visual_training.data.generate_landing_dataset import _plan_sortie
    config = yaml.safe_load(Path("configs/landing_dataset_robust.yaml").read_text())
    config["trajectory_hz"] = 4
    for key in ("approach_duration_s", "hover_duration_s", "descent_duration_s"):
        config["mission"][key] = [1,2]
    before = copy.deepcopy(config)
    paths = [tmp_path / f"trajectory{i}.json" for i in range(3)]
    for i, seed in enumerate((123,123,456)):
        _plan_sortie((config, seed, "fixture", "clear", None, "train", paths[i]))
    sorties = [json.loads(p.read_text()) for p in paths]
    assert config == before
    assert sorties[0] == sorties[1]
    assert sorties[0]["gimbal_config"] != sorties[2]["gimbal_config"]
    mount = config["camera"]["position_body_m"]
    for sortie in sorties:
        for row in sortie["rows"]:
            assert np.allclose(np.array(row["T_body_camera"])[:3,3], mount)
            assert row["gimbal"]["timestamp"] == row["timestamp"]


def test_offline_replay_keeps_truth_out_of_measurements(tmp_path):
    from simulation.offline.replay_dataset import replay_sequence, plot_sequence, write_json
    from visual_training.data.generate_landing_dataset import plan_dataset, render_dataset
    from navigation.vision.infer_h import HDetector
    from types import SimpleNamespace
    c = yaml.safe_load(Path("configs/landing_dataset.yaml").read_text())
    c.update(sorties_per_weather=3, frames_per_stage=2, trajectory_hz=4)
    c["weather"] = {"clear": c["weather"]["clear"]}
    c["camera"].update(width=80, height=60)
    for key in ("approach_duration_s", "hover_duration_s", "descent_duration_s"):
        c["mission"][key] = [1,2]
    root = tmp_path / "data"
    plan = plan_dataset(c, root)
    render_dataset(c, plan, root, "cpu")
    records = load_manifest(root / "splits/val.json", root, WEATHERS)
    records.sort(key=lambda item: item[1]["timestamp"])
    args = SimpleNamespace(initialization="truth", initial_position_error=[0,0,0],
                           initial_rotation_error=[0,0,0], compare_baseline=True, split="val")
    config = yaml.safe_load(Path("configs/navigation.yaml").read_text())
    detector = HDetector()
    original = replay_sequence(records, root, detector, config, args)
    # Change all later pose/scoring truth consistently, retaining measurements.
    changed = copy.deepcopy(records)
    for _, row in changed[1:]:
        T = np.asarray(row["T_world_body"])
        T[:3,3] += np.asarray(row["T_world_deck"])[:3,:3] @ [2.,0.,0.]
        row["T_world_body"] = T.tolist()
        row["gear_relative_deck_m"][0] += 2.
        row["T_deck_camera"][0][3] += 2.
    replayed = replay_sequence(changed, root, detector, config, args)
    for variant in original["variants"]:
        x = np.array([r["estimate_deck_m"] for r in original["variants"][variant]])
        y = np.array([r["estimate_deck_m"] for r in replayed["variants"][variant]])
        assert np.array_equal(x,y)
    out = tmp_path / "plots"
    out.mkdir()
    plot_sequence(original, out)
    write_json(out / "result.json", original)
    assert len(list(out.glob("*.png"))) == len(list(out.glob("*.pdf"))) == 2
