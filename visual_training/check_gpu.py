"""Verify the RTX 5090 CUDA build with a real BF16 training step."""

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from packaging.version import Version

from visual_training.models.h_segmentation_keypoints import HSegmentationKeypoints
from visual_training.models.losses import segmentation_keypoint_loss
from visual_training.train import training_device


def check_profile_versions(python_version, torch_version, cuda_runtime):
    """Validate the server profile independently of driver/device discovery."""
    if tuple(python_version[:2]) != (3, 12):
        raise RuntimeError("RTX 5090 training profile requires Python 3.12")
    if Version(str(torch_version)).public != "2.8.0":
        raise RuntimeError("RTX 5090 training profile requires PyTorch 2.8.0 (cu128)")
    if cuda_runtime != "12.8":
        raise RuntimeError(
            "RTX 5090 training profile requires PyTorch CUDA runtime 12.8; "
            "install requirements-rtx5090.txt (system CUDA Toolkit is separate)"
        )


def check_gpu(config):
    check_profile_versions(sys.version_info, torch.__version__, torch.version.cuda)
    if torch.device(config.get("device", "cuda:0")).type != "cuda":
        raise ValueError("RTX 5090 preflight requires a CUDA device")
    device, precision = training_device(config)
    capability = torch.cuda.get_device_capability(device)
    architectures = torch.cuda.get_arch_list()
    if capability != (12, 0):
        raise RuntimeError(
            f"Expected RTX 5090 / SM 120; selected {torch.cuda.get_device_name(device)} / SM {capability}"
        )
    if "sm_120" not in architectures:
        raise RuntimeError(f"This PyTorch build lacks SM 120 kernels: {architectures}")
    if precision != "bf16":
        raise ValueError("RTX 5090 preflight expects the BF16 training profile")
    torch.set_num_threads(config.get("threads", 8))
    torch.manual_seed(config.get("seed", 42))
    model = HSegmentationKeypoints(**config.get("model", {})).to(device)
    image = torch.rand(2, 3, 64, 64, device=device)
    if config.get("channels_last", False):
        model.to(memory_format=torch.channels_last)
        image = image.contiguous(memory_format=torch.channels_last)
    target = dict(
        image=image,
        mask=torch.zeros(2, 1, 64, 64, device=device),
        keypoints=torch.full((2, 4, 2), 0.4, device=device),
        visibility=torch.tensor([[True, False, True, True]] * 2, device=device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        fused=bool(config.get("fused_optimizer", False)),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(image)
    loss, _ = segmentation_keypoint_loss(output, target, return_metrics=False)
    if output["keypoints"].dtype != torch.float32 or loss.dtype != torch.float32:
        raise RuntimeError("Keypoint coordinates and loss must remain FP32")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite CUDA loss")
    loss.backward()
    if not all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    ):
        raise RuntimeError("Missing or non-finite CUDA gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    return dict(
        status="ok",
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        device=str(device),
        gpu=torch.cuda.get_device_name(device),
        capability=list(capability),
        architectures=architectures,
        total_memory_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
        precision=precision,
        smoke_image_size=64,
        loss=float(loss.detach()),
        keypoint_dtype=str(output["keypoints"].dtype),
        loss_dtype=str(loss.dtype),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/multi_camera_training_rtx5090.yaml"
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.device is not None:
        config["device"] = args.device
    try:
        print(json.dumps(check_gpu(config)), flush=True)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"GPU preflight failed: {error}") from error


if __name__ == "__main__":
    main()
