import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from visual_training.data.h_dataset import HDataset
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints


def evaluate(checkpoint, split, root="data", image_size=256, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = HSegmentationKeypoints(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = HDataset(split, root, image_size)
    loader = DataLoader(dataset, batch_size=4)
    intersection = union = 0
    error = count = 0.0
    groups = {}
    offset = 0
    with torch.no_grad():
        for sample in loader:
            sample = {k: v.to(device) for k, v in sample.items()}
            out = model(sample["image"])
            p = out["mask_logits"].sigmoid() > 0.5
            y = sample["mask"] > 0.5
            intersection += int((p & y).sum())
            union += int((p | y).sum())
            e = torch.linalg.vector_norm(out["keypoints"] - sample["keypoints"], dim=-1)
            v = sample["visibility"]
            error += float(e[v].sum())
            count += int(v.sum())
            for i in range(sample["image"].shape[0]):
                row = json.loads(dataset.paths[offset + i].read_text())
                keys = [
                    f"stage/{row.get('stage', 'unspecified')}",
                    f"weather/{row.get('weather', 'unspecified')}",
                    f"stage_weather/{row.get('stage', 'unspecified')}/{row.get('weather', 'unspecified')}",
                ]
                for key in keys:
                    g = groups.setdefault(
                        key,
                        dict(intersection=0, union=0, error=0.0, visible=0, samples=0),
                    )
                    g["intersection"] += int((p[i] & y[i]).sum())
                    g["union"] += int((p[i] | y[i]).sum())
                    g["error"] += float(e[i][v[i]].sum())
                    g["visible"] += int(v[i].sum())
                    g["samples"] += 1
            offset += sample["image"].shape[0]
    return dict(
        mask_iou=intersection / max(union, 1),
        normalized_keypoint_error=error / count if count else None,
        visible_keypoints=int(count),
        by_condition={
            key: dict(
                mask_iou=g["intersection"] / max(g["union"], 1),
                normalized_keypoint_error=g["error"] / g["visible"]
                if g["visible"]
                else None,
                visible_keypoints=g["visible"],
                samples=g["samples"],
            )
            for key, g in groups.items()
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="visual_training/checkpoints/best.pt")
    p.add_argument("--root", default="data")
    p.add_argument("--split")
    p.add_argument("--output", default="outputs/metrics/vision.json")
    p.add_argument("--image-size", type=int, default=256)
    a = p.parse_args()
    result = evaluate(
        a.checkpoint,
        a.split or str(Path(a.root) / "splits/test.json"),
        a.root,
        a.image_size,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
