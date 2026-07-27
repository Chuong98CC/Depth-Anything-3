"""
Scale camera intrinsics from calibration resolution to a target resolution.

The calibration file (astribot_calibration_full.json) contains intrinsics calibrated
at various resolutions (1280x720, 640x320, 1600x1200), but all videos are recorded at
640x480. This script scales the intrinsics accordingly.

Usage as CLI (outputs a new JSON file):
    python tools/scale_intrinsics.py

Usage as module:
    from tools.scale_intrinsics import load_calib, get_camera_params, list_cameras

    # Load and scale all intrinsics to 640x480
    calib = load_calib("path/to/calib.json", target_res=(640, 480))

    # Get numpy arrays ready for DepthAnything3.inference()
    exts, ixts = get_camera_params(calib, ["head_rgbd", "torso_rgbd"])
    # exts: (N, 4, 4), ixts: (N, 3, 3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def scale_intrinsics_matrix(
    matrix: list[float],
    orig_w: int,
    orig_h: int,
    target_w: int = 640,
    target_h: int = 480,
) -> np.ndarray:
    """Scale a 3x3 intrinsic matrix from original to target resolution.

    Args:
        matrix: Flat list of 9 floats representing a 3x3 intrinsic matrix.
        orig_w: Original image width.
        orig_h: Original image height.
        target_w: Target image width (default 640).
        target_h: Target image height (default 480).

    Returns:
        Scaled 3x3 intrinsic matrix as a numpy array.

    Formula:
        fx *= target_w / orig_w    cx *= target_w / orig_w
        fy *= target_h / orig_h    cy *= target_h / orig_h
    """
    K = np.array(matrix, dtype=np.float64).reshape(3, 3).copy()
    sx = target_w / orig_w
    sy = target_h / orig_h
    K[0, 0] *= sx  # fx
    K[0, 2] *= sx  # cx
    K[1, 1] *= sy  # fy
    K[1, 2] *= sy  # cy
    return K


def _get_resolution(cam_data: dict) -> Optional[str]:
    """Extract resolution string from camera data.

    Checks both camera-level and intrinsics-level (some entries nest resolution
    inside the intrinsics dict).
    """
    if "resolution" in cam_data:
        return cam_data["resolution"]
    if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
        return cam_data["intrinsics"]["resolution"]
    return None


def _normalize_resolution(cam_data: dict, target_res_str: str) -> None:
    """Ensure resolution is set at camera level (not nested inside intrinsics)."""
    cam_data["resolution"] = target_res_str
    if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
        cam_data["intrinsics"]["resolution"] = target_res_str


def load_calib(
    json_path: str | Path,
    target_res: tuple[int, int] = (640, 480),
) -> dict:
    """Load a calibration JSON file and scale all intrinsics to target resolution.

    Modifies the data in-place: fx, fy, cx, cy are scaled, and resolution fields
    are updated. Extrinsics are left untouched (resolution-independent).

    Args:
        json_path: Path to the calibration JSON file.
        target_res: (width, height) to scale to. Default (640, 480).

    Returns:
        The calibration dict with scaled intrinsics.
    """
    with open(json_path) as f:
        data = json.load(f)

    target_w, target_h = target_res
    target_res_str = f"{target_w}x{target_h}"

    cameras = data.get("camera", {})
    if not cameras:
        print("Warning: No 'camera' key found in calibration file.")
        return data

    for cam_name, cam_data in cameras.items():
        res_str = _get_resolution(cam_data)
        if res_str is None:
            print(f"Warning: No resolution found for '{cam_name}', skipping.")
            continue

        orig_w, orig_h = map(int, res_str.split("x"))

        if orig_w == target_w and orig_h == target_h:
            _normalize_resolution(cam_data, target_res_str)
            print(f"  {cam_name}: already at {target_res_str}, no scaling needed.")
            continue

        # Scale all intrinsics (color, depth, etc.)
        if "intrinsics" in cam_data:
            for sensor_type, sensor in cam_data["intrinsics"].items():
                if sensor_type == "resolution":
                    continue  # skip resolution string inside intrinsics
                if "matrix" not in sensor:
                    continue
                K_scaled = scale_intrinsics_matrix(
                    sensor["matrix"], orig_w, orig_h, target_w, target_h
                )
                sensor["matrix"] = K_scaled.flatten().tolist()

        _normalize_resolution(cam_data, target_res_str)
        print(f"  {cam_name}: {res_str} -> {target_res_str}")

    return data


def get_camera_params(
    calib: dict,
    camera_names: list[str],
    sensor_type: str = "color",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract extrinsics and intrinsics for given cameras as stacked numpy arrays.

    Args:
        calib: Calibration dict (from load_calib or raw JSON).
        camera_names: List of camera names to extract (e.g. ["head_rgbd"]).
        sensor_type: Which intrinsics to use: "color" or "depth".

    Returns:
        (extrinsics, intrinsics) tuple:
            extrinsics: shape (N, 4, 4) — world-to-camera transforms.
            intrinsics: shape (N, 3, 3) — camera intrinsic matrices.
    """
    exts, ixts = [], []
    for name in camera_names:
        cam = calib["camera"][name]
        exts.append(np.array(cam["extrinsics"]["matrix"], dtype=np.float64).reshape(4, 4))
        ixts.append(np.array(cam["intrinsics"][sensor_type]["matrix"], dtype=np.float64).reshape(3, 3))
    return np.stack(exts), np.stack(ixts)


def list_cameras(calib: dict) -> list[str]:
    """Return sorted list of camera names in the calibration dict."""
    return sorted(calib.get("camera", {}).keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    input_path = repo_root / "data" / "astribot_camera_calib_params" / "astribot_calibration_full.json"
    output_path = repo_root / "data" / "astribot_camera_calib_params" / "astribot_calibration_full_640x480.json"

    print(f"Loading: {input_path}")
    calib = load_calib(str(input_path), target_res=(640, 480))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\nScaled calibration written to: {output_path}")
    print(f"Cameras: {list_cameras(calib)}")
