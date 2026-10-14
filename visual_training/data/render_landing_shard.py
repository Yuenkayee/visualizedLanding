"""Private fresh-process renderer entry; no torch/CUDA fork or shared manifests."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--device", choices=["OPTIX", "CUDA", "CPU"], default="OPTIX")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--renderer", choices=["blender", "cpu"], default="blender")
    args = parser.parse_args()
    if not 0 <= args.worker_id < args.workers or args.threads < 1:
        parser.error("invalid worker/thread configuration")
    import cv2
    from visual_training.data.generate_landing_dataset import render_dataset, write_json

    cv2.setNumThreads(1)
    start = time.monotonic()
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    sorties = plan["sorties"][args.worker_id :: args.workers]
    runtime = dict(backend="cpu_synthetic", threads=1)
    if args.renderer == "blender":
        from visual_modeling.blender.cycles_device import configure_cycles

        runtime = configure_cycles(args.device, args.threads, args.gpu_index)
    print(json.dumps(runtime), flush=True)
    result = render_dataset(
        plan["config"],
        sorties,
        root,
        args.renderer,
        finalize=False,
        progress_path=root / "workers" / f"{args.worker_id:02d}_progress.json",
    )
    result.update(
        worker_id=args.worker_id,
        runtime=runtime,
        sequences=[s["sequence_id"] for s in sorties],
        elapsed_s=time.monotonic() - start,
    )
    write_json(root / "workers" / f"{args.worker_id:02d}.json", result)


if __name__ == "__main__":
    main()
