"""On-device epoch aggregation; transfer scalars only when logging."""

import torch


class EpochMetrics:
    def __init__(self, device, image_size):
        self.size = image_size
        self.samples = 0
        self.losses = {}
        dtype = torch.float32 if torch.device(device).type == "mps" else torch.float64
        self.counts = torch.zeros(9, device=device, dtype=dtype)
        self.has_visibility = False

    @torch.no_grad()
    def update(self, output, target, loss, components):
        n = target["image"].shape[0]
        self.samples += n
        for key, value in dict(loss=loss.detach(), **components).items():
            if key not in self.losses:
                self.losses[key] = torch.zeros_like(value)
            self.losses[key] += value * n
        mask = output["mask_logits"] > 0
        truth = target["mask"] > 0.5
        visible = target["visibility"]
        error = (
            torch.linalg.vector_norm(
                output["keypoints"].float() - target["keypoints"], dim=-1
            )
            * self.size
        )
        self.counts[:6] += torch.stack(
            [
                (mask & truth).sum(),
                (mask | truth).sum(),
                (error * visible).sum(),
                visible.sum(),
                ((error <= 2) & visible).sum(),
                ((error <= 5) & visible).sum(),
            ]
        )
        if "visibility_logits" in output:
            self.has_visibility = True
            predicted = output["visibility_logits"] > 0
            self.counts[6:] += torch.stack(
                [
                    (predicted & visible).sum(),
                    predicted.sum(),
                    visible.sum(),
                ]
            )

    def result(self):
        losses = {k: float(v / self.samples) for k, v in self.losses.items()}
        intersection, union, error, visible, pck2, pck5, tp, pred, actual = (
            self.counts.tolist()
        )
        result = dict(
            **losses,
            mask_iou=intersection / max(union, 1),
            keypoint_error_px=error / visible if visible else None,
            keypoint_pck_2px=pck2 / visible if visible else None,
            keypoint_pck_5px=pck5 / visible if visible else None,
            visible_keypoints=int(visible),
        )
        if self.has_visibility:
            result.update(
                visibility_precision=tp / pred if pred else None,
                visibility_recall=tp / actual if actual else None,
            )
        return result
