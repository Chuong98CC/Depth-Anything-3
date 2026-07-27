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
from typing import Optional

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


# ---------------------------------------------------------------------------
# Inlined from tools/scale_intrinsics.py — avoids cross-module import issues
# ---------------------------------------------------------------------------

def _scale_intrinsics_matrix(
    matrix: list[float],
    orig_w: int,
    orig_h: int,
    target_w: int = 640,
    target_h: int = 480,
) -> np.ndarray:
    """Scale a 3x3 intrinsic matrix from original to target resolution."""
    K = np.array(matrix, dtype=np.float64).reshape(3, 3).copy()
    sx = target_w / orig_w
    sy = target_h / orig_h
    K[0, 0] *= sx  # fx
    K[0, 2] *= sx  # cx
    K[1, 1] *= sy  # fy
    K[1, 2] *= sy  # cy
    return K


def _get_resolution(cam_data: dict) -> Optional[str]:
    """Extract resolution string from camera data (camera-level or intrinsics-level)."""
    if "resolution" in cam_data:
        return cam_data["resolution"]
    if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
        return cam_data["intrinsics"]["resolution"]
    return None


def load_calib(
    json_path: str | Path,
    target_res: tuple[int, int] = (640, 480),
) -> dict:
    """Load calibration JSON and scale all intrinsics to target resolution."""
    with open(json_path) as f:
        data = json.load(f)

    target_w, target_h = target_res
    target_res_str = f"{target_w}x{target_h}"

    cameras = data.get("camera", {})
    for cam_name, cam_data in cameras.items():
        res_str = _get_resolution(cam_data)
        if res_str is None:
            print(f"Warning: No resolution found for '{cam_name}', skipping.")
            continue

        orig_w, orig_h = map(int, res_str.split("x"))
        if orig_w == target_w and orig_h == target_h:
            cam_data["resolution"] = target_res_str
            if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
                cam_data["intrinsics"]["resolution"] = target_res_str
            continue

        if "intrinsics" in cam_data:
            for sensor_type, sensor in cam_data["intrinsics"].items():
                if sensor_type == "resolution":
                    continue
                if "matrix" not in sensor:
                    continue
                K_scaled = _scale_intrinsics_matrix(
                    sensor["matrix"], orig_w, orig_h, target_w, target_h
                )
                sensor["matrix"] = K_scaled.flatten().tolist()

        cam_data["resolution"] = target_res_str
        if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
            cam_data["intrinsics"]["resolution"] = target_res_str

    return data


def get_camera_params(
    calib: dict,
    camera_names: list[str],
    sensor_type: str = "color",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract extrinsics (N,4,4) and intrinsics (N,3,3) for given cameras."""
    exts, ixts = [], []
    for name in camera_names:
        cam = calib["camera"][name]
        exts.append(np.array(cam["extrinsics"]["matrix"], dtype=np.float64).reshape(4, 4))
        ixts.append(np.array(cam["intrinsics"][sensor_type]["matrix"], dtype=np.float64).reshape(3, 3))
    return np.stack(exts), np.stack(ixts)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paths relative to repo root
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/chuong/workspace/depth_models/Depth-Anything-3/data")
CALIB_PATH = DATA_ROOT / "astribot_camera_calib_params" / "astribot_calibration_full.json"
IMAGES_ROOT = DATA_ROOT / "astribot_stereo_lrb" / "images"

# ---------------------------------------------------------------------------
# Mapping: image directory suffix → calibration key
# ---------------------------------------------------------------------------
DIR_TO_CALIB: dict[str, str] = {
    "cam_head":                "head_rgbd",
    "cam_head_stereo_left":    "head_stereo_left",
    "cam_head_stereo_right":   "head_stereo_right",
    "cam_torso":               "torso_rgbd",
    "cam_left_wrist":          "left_wrist_rgbd",
    "cam_right_wrist":         "right_wrist_rgbd",
}

# Reverse: calibration key → image directory name
CALIB_TO_DIR: dict[str, str] = {v: k for k, v in DIR_TO_CALIB.items()}

# ---------------------------------------------------------------------------
# Camera sets
# ---------------------------------------------------------------------------
CAMERA_SETS: dict[str, list[str]] = {
    "set1": ["head_rgbd", "head_stereo_left", "head_stereo_right"],
    "set2": ["head_rgbd", "head_stereo_left", "head_stereo_right", "torso_rgbd"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_image_dir(calib_key: str) -> Path:
    """Return the image directory path for a given calibration key."""
    dir_name = CALIB_TO_DIR[calib_key]
    return IMAGES_ROOT / f"observation.images.{dir_name}" / "chunk-000" / "file-000"

def load_frame(calib_key: str, frame_idx: int) -> str:
    """Return the absolute path to a specific frame from a camera.

    Args:
        calib_key: e.g. "head_rgbd".
        frame_idx: 0-based frame number.

    Returns:
        Absolute path to the JPEG file.
    """
    img_dir = resolve_image_dir(calib_key)
    img_path = img_dir / f"frame_{frame_idx:06d}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"Frame not found: {img_path}")
    return str(img_path)

def count_frames(calib_key: str) -> int:
    """Return the number of JPEG frames in a camera's image directory."""
    img_dir = resolve_image_dir(calib_key)
    return len(sorted(img_dir.glob("frame_*.jpg")))

def load_camera_data(
    calib: dict,
    camera_set: list[str],
    frame_idx: int,
    sensor_type: str = "color",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load images, extrinsics, and intrinsics for one frame across cameras.

    Args:
        calib: Scaled calibration dict.
        camera_set: List of calibration keys.
        frame_idx: Frame index to load.
        sensor_type: "color" or "depth".

    Returns:
        (image_paths, extrinsics, intrinsics) tuple:
            image_paths: list of N absolute paths.
            extrinsics: ndarray shape (N, 4, 4).
            intrinsics: ndarray shape (N, 3, 3).
    """
    images = [load_frame(key, frame_idx) for key in camera_set]
    exts, ixts = get_camera_params(calib, camera_set, sensor_type=sensor_type)
    return images, exts, ixts

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DepthAnything3 inference on Astribot stereo dataset",
    )
    parser.add_argument(
        "--camera-set", choices=["set1", "set2"], default="set1",
        help="Camera set: set1 = head_rgbd+stereo_left+stereo_right, "
             "set2 = set1+torso_rgbd (default: set1)",
    )
    parser.add_argument(
        "--frame", type=int, default=None,
        help="Single frame index to process (0-based). "
             "Use --all-frames to process all common frames.",
    )
    parser.add_argument(
        "--all-frames", action="store_true",
        help="Process all frames common to the selected cameras.",
    )
    parser.add_argument(
        "--sensor-type", choices=["color", "depth"], default="color",
        help="Which intrinsics to use (default: color)",
    )
    parser.add_argument(
        "--model-name", default="depth-anything/DA3NESTED-GIANT-LARGE",
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
        "--use-ray-pose", action="store_true",
        help="Use ray pose for inference",
    )
    parser.add_argument(
        "--show-cameras", action="store_true",
        help="Show camera frustums in exported GLB",
    )
    args = parser.parse_args()

    # --- Load & scale calibration ---
    calib = load_calib(str(CALIB_PATH), target_res=(640, 480))
    camera_set = CAMERA_SETS[args.camera_set]

    print(f"Camera set: {args.camera_set} → {camera_set}")

    # --- Determine frame range ---
    if args.all_frames:
        frame_counts = {key: count_frames(key) for key in camera_set}
        print("Frames per camera:")
        for key, n in frame_counts.items():
            print(f"  {key}: {n}")
        num_frames = min(frame_counts.values())
        if num_frames == 0:
            raise RuntimeError("No frames found for one or more cameras.")
        print(f"Processing all {num_frames} common frames.")
        frame_indices = list(range(num_frames))
    elif args.frame is not None:
        frame_indices = [args.frame]
    else:
        parser.error("Specify --frame N or --all-frames.")

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
    for frame_idx in frame_indices:
        print(f"\n--- Frame {frame_idx} ---")
        images, exts, ixts = load_camera_data(
            calib, camera_set, frame_idx, sensor_type=args.sensor_type,
        )
        print(f"  Views: {len(images)}")
        for img_path in images:
            print(f"    {img_path}")

        export_dir = None
        if args.export_dir:
            base = Path(args.export_dir)
            if len(frame_indices) > 1:
                export_dir = str(base / f"frame_{frame_idx:06d}")
            else:
                export_dir = str(base)

        prediction = model.inference(
            image=images,
            extrinsics=exts if args.camera_set else None,
            intrinsics=ixts,
            align_to_input_ext_scale=True,
            infer_gs=args.infer_gs,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=export_dir,
            export_format=args.export_format,
            use_ray_pose=args.use_ray_pose,
            show_cameras=args.show_cameras,
        )

        print(f"  depth shape:    {prediction.depth.shape}")
        if hasattr(prediction, "conf") and prediction.conf is not None:
            print(f"  conf shape:     {prediction.conf.shape}")
        print(f"  extrinsics out: {prediction.extrinsics.shape}")
        print(f"  intrinsics out: {prediction.intrinsics.shape}")


if __name__ == "__main__":
    main()
