import torch
from torch.nn import functional as F


def segmentation_keypoint_loss(
    output, target, keypoint_weight=10.0, heatmap_weight=1.0, return_metrics=True,
    heatmap_sigma=0.0, coordinate_beta=1.0, visibility_weight=0.2
):
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
        output["keypoints"].float(), target["keypoints"], reduction="none", beta=coordinate_beta
    ).sum(-1)
    kp = (error * visible).sum() / visible.sum().clamp_min(1)
    heat = output["heatmap_logits"].float()
    n, k, h, w = heat.shape
    xy = target["keypoints"]
    flat = heat.reshape(n, k, -1)
    if heatmap_sigma > 0:
        # Sparse continuous-centre Gaussian targets; no dense NxKxHxW target.
        radius = max(1, int(3 * heatmap_sigma + .5))
        offsets = torch.arange(-radius, radius + 1, device=heat.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        centers = xy * xy.new_tensor([w, h]) - .5
        cells = centers.floor().long()[..., None, :] + torch.stack([dx.flatten(), dy.flatten()], -1)
        valid = ((cells >= 0) & (cells < cells.new_tensor([w, h]))).all(-1)
        weights = torch.exp(-((cells - centers[..., None, :]) ** 2).sum(-1) / (2 * heatmap_sigma ** 2)) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        indices = cells[..., 1].clamp(0, h-1) * w + cells[..., 0].clamp(0, w-1)
        ce = flat.logsumexp(-1) - (flat.gather(-1, indices) * weights).sum(-1)
    else:
        x = (xy[:, :, 0] * w).long().clamp(0, w - 1)
        y = (xy[:, :, 1] * h).long().clamp(0, h - 1)
        ce = F.cross_entropy(flat.reshape(n*k, -1), (y*w+x).reshape(-1), reduction="none").reshape(n, k)
    heat_loss = (ce * visible).sum() / visible.sum().clamp_min(1)
    visibility_loss = F.binary_cross_entropy_with_logits(
        output["visibility_logits"].float(), visible
    ) if "visibility_logits" in output else logits.new_zeros(())
    total = bce + dice + keypoint_weight * kp + heatmap_weight * heat_loss + visibility_weight * visibility_loss
    components = dict(bce=bce.detach(), dice=dice.detach(), keypoint=kp.detach(),
                      heatmap=heat_loss.detach(), visibility=visibility_loss.detach())
    if return_metrics == "tensor":
        return total, components
    return total, {k: float(v) for k, v in components.items()} if return_metrics else {}
