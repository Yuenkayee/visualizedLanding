"""Small encoder/decoder with segmentation and spatial-softargmax landmark heads."""

import torch
from torch import nn
from torch.nn import functional as F


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1),
        nn.GroupNorm(4, o),
        nn.SiLU(),
        nn.Conv2d(o, o, 3, padding=1),
        nn.GroupNorm(4, o),
        nn.SiLU(),
    )


class HSegmentationKeypoints(nn.Module):
    def __init__(self, channels=16, keypoints=4, context=False, visibility_head=False):
        super().__init__()
        if channels % 4:
            raise ValueError("channels must be divisible by four")
        self.model_config = dict(channels=channels, keypoints=keypoints,
                                 context=context, visibility_head=visibility_head)
        self.e1 = block(3, channels)
        self.e2 = block(channels, channels * 2)
        self.e3 = block(channels * 2, channels * 4)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels * 4, channels * 4, 3, padding=1), nn.SiLU(),
        ) if context else None
        self.visibility = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(channels * 4 * 16, keypoints),
        ) if visibility_head else None
        self.d2 = block(channels * 6, channels * 2)
        self.d1 = block(channels * 3, channels)
        self.seg = nn.Conv2d(channels, 1, 1)
        self.kp = nn.Conv2d(channels, keypoints, 1)

    def forward(self, image):
        a = self.e1(image)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        # Pool visibility before context resize so ONNX retains static spatial
        # dimensions with dynamic batch size (legacy exporter compatibility).
        visibility_logits = self.visibility(c) if self.visibility is not None else None
        if self.context is not None:
            c = c + F.interpolate(self.context(c), size=c.shape[-2:], mode="bilinear", align_corners=False)
        d = self.d2(
            torch.cat(
                [
                    F.interpolate(
                        c, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ],
                1,
            )
        )
        d = self.d1(
            torch.cat(
                [
                    F.interpolate(
                        d, size=a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                ],
                1,
            )
        )
        heatmaps = self.kp(d)
        n, k, h, w = heatmaps.shape
        # Wide 512x512 heatmaps need FP32 probabilities and reductions under
        # BF16 autocast; otherwise small per-pixel probabilities lose precision.
        prob = heatmaps.float().flatten(2).softmax(-1).reshape(n, k, h, w)
        xs = (torch.arange(w, device=image.device, dtype=image.dtype) + 0.5) / w
        ys = (torch.arange(h, device=image.device, dtype=image.dtype) + 0.5) / h
        x = (prob.sum(2) * xs).sum(2)
        y = (prob.sum(3) * ys).sum(2)
        output = dict(
            mask_logits=self.seg(d),
            keypoints=torch.stack([x, y], -1),
            heatmap_logits=heatmaps,
        )
        if visibility_logits is not None:
            output["visibility_logits"] = visibility_logits
        return output
