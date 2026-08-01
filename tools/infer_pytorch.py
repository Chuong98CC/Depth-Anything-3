"""
Run DepthAnything3 inference on the Astribot stereo dataset.

Loads camera extrinsics and intrinsics (scaled to 640x480) from the calibration
file, selects a camera set, and feeds synchronized multi-view frames to the model.

Usage:
    # Set1: head_rgbd + head_stereo_left + head_stereo_right (3 views per frame)
    python tools/infer_pytorch.py --camera-set set1 --frame 0

    # Set2: Set1 + torso_rgbd (4 views per frame)
    python tools/infer_pytorch.py --camera-set set2 --frame 0

    # Process all common frames across selected cameras
    python tools/infer_pytorch.py --camera-set set1 --all-frames

    # Export GLB and NPZ
    python tools/infer_pytorch.py --camera-set set2 --frame 0 \\
        --export-dir output/stereo --export-format mini_npz-glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3
from tools.utils.astribot_dataloader import load_images_cam_params



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DepthAnything3 inference on Astribot stereo dataset",
    )
    parser.add_argument(
        "--camera-set", choices=["set0", "set1", "set2"], default="set1",
        help="Camera set: set0 = stereo, "
             "set1 = head_rgbd+stereo_left+stereo_right, "
             "set2 = set1+torso_rgbd (default: set1)",
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Single frame index to process (0-based). "
             "Use --all-frames to process all common frames.",
    )
    parser.add_argument(
        "--model-name", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Model name or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--export-dir", default='output',
        help="Directory to export results",
    )
    parser.add_argument(
        "--export-format", default="mini_npz-glb-depth_vis",
        help="Export format(s), hyphen-separated (e.g. mini_npz-glb)",
    )
    parser.add_argument(
        "--process-res", type=int, default=644,
        help="Base processing resolution",
    )
    parser.add_argument(
        "--infer-gs", action="store_true",
        help="Enable Gaussian Splatting branch",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--use-extrinsics", action="store_true",
        help="Use extrinsics for inference",
    )
    parser.add_argument(
        "--use-ray-pose", action="store_true",
        help="Use ray pose for inference",
    )
    parser.add_argument(
        "--show-cameras", action="store_true",
        help="Show camera frustums in exported GLB",
    )
    args = parser.parse_args()

    # --- Load & scale calibration ---


    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading model: {args.model_name}")
    model = DepthAnything3.from_pretrained(args.model_name)
    model = model.to(device=device)

    # --- Run inference per multi-view group ---

    frame_idx = args.frame
    print(f"\n--- Frame {frame_idx} ---")
    images, exts, ixts = load_images_cam_params(args.camera_set, frame_idx)
    print(f"  Views: {len(images)}")
    for img_path in images:
        print(f"    {img_path}")

    export_dir = None
    if args.export_dir:
        base = Path(args.export_dir)
        export_dir = str(base)

    prediction = model.inference(
        image=images,
        extrinsics=exts if args.use_extrinsics else None,
        intrinsics=ixts,
        align_to_input_ext_scale=True,
        infer_gs=args.infer_gs,
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
        export_dir=export_dir,
        export_format=args.export_format,
        use_ray_pose=args.use_ray_pose,
        show_cameras=args.show_cameras,
        # Match the ONNX/TRT export (which pins "first"): the default
        # "saddle_balanced" picks the reference view via a data-dependent argmin
        # that is fp-noise-sensitive and does not survive ONNX export. Using
        # "first" keeps this PyTorch reference consistent with the exported models.
        ref_view_strategy="first",
    )

    print(f"  depth shape:    {prediction.depth.shape}")
    if hasattr(prediction, "conf") and prediction.conf is not None:
        print(f"  conf shape:     {prediction.conf.shape}")
    print(f"  extrinsics out: {prediction.extrinsics.shape}")
    print(f"  intrinsics out: {prediction.intrinsics.shape}")

    camera_matrices = {
        "camera_set": args.camera_set,
        "frame_idx": frame_idx,
        "images": images,
        "input": {
            "extrinsics": exts.tolist(),
            "intrinsics": ixts.tolist(),
        },
        "output": {
            "extrinsics": (
                prediction.extrinsics.tolist()
                if prediction.extrinsics is not None
                else None
            ),
            "intrinsics": (
                prediction.intrinsics.tolist()
                if prediction.intrinsics is not None
                else None
            ),
        },
    }

    camera_json_path = Path(export_dir or "output") / f"frame_{frame_idx:06d}_camera_matrices.json"
    camera_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(camera_json_path, "w") as f:
        json.dump(camera_matrices, f, indent=2)
    print(f"  camera matrices: {camera_json_path}")


if __name__ == "__main__":
    main()
