import argparse
from pathlib import Path
import torch
from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model(image)
        if "visibility_logits" in out:
            return out["mask_logits"], out["keypoints"], out["visibility_logits"]
        return out["mask_logits"], out["keypoints"]


def export_model(checkpoint, output, image_size=256):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = HSegmentationKeypoints(**payload["model_config"])
    model.load_state_dict(payload["model"])
    model.eval()
    names = ["mask_logits", "keypoints"] + (
        ["visibility_logits"] if model.visibility is not None else []
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        ExportWrapper(model),
        torch.zeros(1, 3, image_size, image_size),
        str(path),
        input_names=["image"],
        output_names=names,
        opset_version=17,
        dynamic_axes={
            "image": {0: "batch"},
            "mask_logits": {0: "batch"},
            "keypoints": {0: "batch"},
            **(
                {"visibility_logits": {0: "batch"}}
                if model.visibility is not None
                else {}
            ),
        },
    )
    import onnx

    onnx.checker.check_model(onnx.load(path))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="visual_training/checkpoints/best.pt")
    p.add_argument("--output", default="visual_training/checkpoints/h_detector.onnx")
    p.add_argument("--image-size", type=int, default=256)
    a = p.parse_args()
    print(export_model(a.checkpoint, a.output, a.image_size))


if __name__ == "__main__":
    main()
