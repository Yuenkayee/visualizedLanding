import torch
from torch.nn import functional as F


def segmentation_keypoint_loss(
    output, target, keypoint_weight=10.0, heatmap_weight=1.0, return_metrics=True
):
    # Compute reductions and landmark supervision in FP32 under mixed precision.
    logits = output["mask_logits"].float()
    truth = target["mask"]
    bce = F.binary_cross_entropy_with_logits(
        logits, truth, pos_weight=torch.tensor(4.0, device=logits.device)
    )
    p = logits.sigmoid()
    dims = (1, 2, 3)
    dice = 1 - ((2 * (p * truth).sum(dims) + 1) / ((p + truth).sum(dims) + 1)).mean()
    visible = target["visibility"].float()
    error = F.smooth_l1_loss(
        output["keypoints"].float(), target["keypoints"], reduction="none"
    ).sum(-1)
    kp = (error * visible).sum() / visible.sum().clamp_min(1)
    heat = output["heatmap_logits"].float()
    n, k, h, w = heat.shape
    xy = target["keypoints"]
    x = (xy[:, :, 0] * w).long().clamp(0, w - 1)
    y = (xy[:, :, 1] * h).long().clamp(0, h - 1)
    ce = F.cross_entropy(
        heat.reshape(n * k, -1), (y * w + x).reshape(-1), reduction="none"
    ).reshape(n, k)
    heat_loss = (ce * visible).sum() / visible.sum().clamp_min(1)
    total = bce + dice + keypoint_weight * kp + heatmap_weight * heat_loss
    visibility_loss = logits.new_tensor(0)
    if "visibility_logits" in output:
        visibility_loss = F.binary_cross_entropy_with_logits(
            output["visibility_logits"].float(), visible
        )
        total = total + visibility_loss
    if not return_metrics:
        return total, {}
    return total, dict(
        bce=float(bce.detach()),
        dice=float(dice.detach()),
        keypoint=float(kp.detach()),
        heatmap=float(heat_loss.detach()),
        visibility=float(visibility_loss.detach()),
    )
