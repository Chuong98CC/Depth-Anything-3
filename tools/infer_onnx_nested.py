#!/usr/bin/env python3
"""
Run nested DepthAnything3 inference via split ONNX models on the Astribot dataset.

Combines an any-view ONNX model with a metric ONNX model (``DA3NestedONNX``),
aligns their outputs, and saves results matching ``infer_pytorch.py``.

Usage:
    # Single frame
    python tools/infer_onnx_nested.py --camera-set set1 --frame 0

    # All frames
    python tools/infer_onnx_nested.py --camera-set set1 --all-frames

    # Custom ONNX paths
    python tools/infer_onnx_nested.py --camera-set set1 --frame 0 \\
        --onnx-anyview weights/da3_anyview_n3_644x490.onnx \\
        --onnx-metric  weights/da3_metric_644x490.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure tools/ is on sys.path for astribot_dataloader and model.* imports
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))

from astribot_dataloader import CAMERA_SETS, count_frames, load_images_cam_params  # noqa: E402
from model.da3nested_onnx import DA3NestedONNX  # noqa: E402

DEFAULT_ANYVIEW_ONNX = "weights/da3_anyview_n3_644x490_giant-large-1.1.onnx"
DEFAULT_METRIC_ONNX = "weights/da3_metric_644x490_giant-large-1.1.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested DepthAnything3 ONNX inference on Astribot stereo dataset",
    )
    parser.add_argument(
        "--camera-set",
        choices=["set1", "set2"],
        default="set1",
        help="set1 = head_rgbd+stereo_left+stereo_right (3 views), "
        "set2 = set1+torso_rgbd (4 views).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Single frame index (0-based).  Use --all-frames to process all.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process all frames common to the selected cameras.",
    )
    parser.add_argument(
        "--onnx-anyview",
        type=str,
        default=DEFAULT_ANYVIEW_ONNX,
        help="Path to any-view ONNX model.",
    )
    parser.add_argument(
        "--onnx-metric",
        type=str,
        default=DEFAULT_METRIC_ONNX,
        help="Path to metric ONNX model.",
    )
    parser.add_argument(
        "--export-dir",
        default="output_onnx",
        help="Directory to save results (NPZ per frame).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX Runtime device.",
    )
    parser.add_argument(
        "--no-align-input-ext-scale",
        dest="align_input_ext_scale",
        action="store_false",
        help="Disable Umeyama alignment of the prediction to the input camera "
        "poses (on by default, matching DepthAnything3.inference).",
    )
    parser.set_defaults(align_input_ext_scale=True)
    args = parser.parse_args()

    # --- Determine frame range ---
    camera_set = CAMERA_SETS[args.camera_set]
    if args.all_frames:
        frame_counts = {key: count_frames(key) for key in camera_set}
        num_frames = min(frame_counts.values())
        if num_frames == 0:
            raise RuntimeError("No frames found.")
        print(f"Frames per camera: {frame_counts}")
        print(f"Processing all {num_frames} common frames.")
        frame_indices = list(range(num_frames))
    elif args.frame is not None:
        frame_indices = [args.frame]
    else:
        parser.error("Specify --frame N or --all-frames.")

    # --- Load nested ONNX pipeline ---
    pipeline = DA3NestedONNX(args.onnx_anyview, args.onnx_metric, device=args.device)

    # --- Per-frame inference ---
    for frame_idx in frame_indices:
        print(f"\n--- Frame {frame_idx} ---")
        images, exts, ixts = load_images_cam_params(args.camera_set, frame_idx)
        print(f"  Views: {len(images)}")
        for p in images:
            print(f"    {p}")

        result = pipeline.infer(
            images,
            exts,
            ixts,
            align_input_ext_scale=args.align_input_ext_scale,
        )

        print(f"  depth shape:      {result['depth'].shape}")
        print(f"  depth_conf shape: {result['depth_conf'].shape}")
        print(f"  extrinsics out:   {result['extrinsics'].shape}")
        print(f"  intrinsics out:   {result['intrinsics'].shape}")

        # --- Save ---
        out_dir = Path(args.export_dir)
        if len(frame_indices) > 1:
            out_dir = out_dir / f"frame_{frame_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_dir / "result.npz",
            depth=result["depth"].astype(np.float32),
            depth_conf=result["depth_conf"].astype(np.float32),
            extrinsics=result["extrinsics"].astype(np.float32),
            intrinsics=result["intrinsics"].astype(np.float32),
        )
        print(f"  Saved: {out_dir / 'result.npz'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
