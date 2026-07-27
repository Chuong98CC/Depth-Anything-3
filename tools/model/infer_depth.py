"""
Monocular metric depth inference on video via Depth-Anything-v3 TensorRT.

Pipeline (per frame)
--------------------
1. Monocular Depth  → metric depth (DA3 TRT + camera intrinsics)
2. Depth map saved  → per-frame .npz or .npy
3. Visualisation    → side-by-side: original frame + colour-mapped depth

Usage
-----
python -m data_generation.depth_model.infer_depth \\
    --trt_engine /path/to/da3.trt \\
    --calib molmo-motion/data_generation/astribot_camera_calib_params/astribot_camera_intrinsics.json \\
    --video /path/to/input.mp4 \\
    --camera head_stereo_left \\
    --start_frame 0 --end_frame 100 --frame_interval 2 \\
    --output /path/to/output_dir
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from data_generation.depth_model.da3metric import DA3MetricModel
from data_generation.depth_model.data_structure import CameraIntrinsics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_video(video_path: str):
    """Open a video file and return (cap, fps, total_frames, width, height)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, total_frames, width, height


def write_video(frames: list[np.ndarray], output_path: str, fps: float):
    """Write a list of BGR frames to an mp4 video file."""
    if not frames:
        print("No frames to write.")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Output video saved: {output_path}")


def colour_map_depth(depth: np.ndarray, vmin: float = 0.0, vmax: float = 8.0) -> np.ndarray:
    """Map a metric depth map (metres) to a BGR colour image."""
    depth_clipped = np.clip(depth, vmin, vmax)
    depth_norm = ((depth_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    # grey out invalid pixels (depth ≤ 0)
    invalid = (depth <= 0) | (depth > vmax)
    colour[invalid] = (128, 128, 128)
    return colour


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monocular metric depth inference (DA3 TRT) on video frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--trt_engine", required=True, type=str,
        help="Path to the DA3 TensorRT engine (.trt) file.",
    )
    parser.add_argument(
        "--calib", required=True, type=str,
        help="Path to the camera calibration JSON file "
             "(full, multi-camera, or single-camera format).",
    )
    parser.add_argument(
        "--video", required=True, type=str,
        help="Path to the input video file.",
    )

    # Camera selection
    parser.add_argument(
        "--camera", default="head_stereo_left", type=str,
        help="Camera name to read intrinsics/resolution from (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline_cams", default=None, type=str,
        help="Stereo pair for baseline in 'camA->camB' notation. "
             "Auto-detected if omitted (e.g. 'head_stereo_left->head_stereo_right').",
    )

    # Frame range
    parser.add_argument(
        "--start_frame", default=0, type=int,
        help="First frame index to process (inclusive, default: %(default)s).",
    )
    parser.add_argument(
        "--end_frame", default=-1, type=int,
        help="Last frame index to process (inclusive). -1 = end of video (default: %(default)s).",
    )
    parser.add_argument(
        "--frame_interval", default=1, type=int,
        help="Stride / step size between frames (default: %(default)s = every frame).",
    )

    # Depth post-processing
    parser.add_argument(
        "--conf_thresh", default=0.5, type=float,
        help="Sky-mask confidence threshold — pixels below this are marked invalid "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--depth_vmin", default=0.0, type=float,
        help="Min depth (metres) for colour-map clipping (default: %(default)s).",
    )
    parser.add_argument(
        "--depth_vmax", default=1.0, type=float,
        help="Max depth (metres) for colour-map clipping (default: %(default)s).",
    )

    # Output
    parser.add_argument(
        "--output", "-o", default="output_depth", type=str,
        help="Output directory for depth arrays and visualisation video (default: %(default)s).",
    )
    parser.add_argument(
        "--save_depth", action="store_true",
        help="Save per-frame metric-depth .npz files (default).",
    )
    parser.add_argument(
        "--save_viz", action="store_true",
        help="Save a side-by-side visualisation video (default).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    """Run DA3 monocular depth inference on selected video frames."""

    # ---- load models ----
    print(f"Calib file : {args.calib}")
    print(f"Camera     : {args.camera}")
    cam_intrinsics = CameraIntrinsics.from_calib_file(
        args.calib, camera=args.camera, baseline_cams=args.baseline_cams,
    )
    print(f"Intrinsics : fx={cam_intrinsics.fx:.1f} fy={cam_intrinsics.fy:.1f} "
          f"cx={cam_intrinsics.cx:.1f} cy={cam_intrinsics.cy:.1f} "
          f"res={cam_intrinsics.w}x{cam_intrinsics.h}")

    print(f"TRT engine : {args.trt_engine}")
    depth_model = DA3MetricModel(
        args.trt_engine, cam_intrinsics, conf_thresh=args.conf_thresh,
    )
    target_h, target_w = depth_model.input_height, depth_model.input_width
    print(f"Model input: {target_w}x{target_h}")

    # ---- open video ----
    cap, fps, total_frames, src_w, src_h = open_video(args.video)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError("Cannot read video frames (zero-size frames).")
    print(f"Source     : {src_w}x{src_h}  fps={fps:.1f}  total_frames={total_frames}")

    # ---- determine frame indices ----
    end_frame = args.end_frame if args.end_frame >= 0 else total_frames - 1
    end_frame = min(end_frame, total_frames - 1)
    start_frame = max(0, args.start_frame)

    frame_indices = list(range(start_frame, end_frame + 1, args.frame_interval))
    n_frames = len(frame_indices)
    print(f"Processing : frames [{start_frame}..{end_frame}] stride={args.frame_interval}  → {n_frames} frame(s)")

    if n_frames == 0:
        print("No frames to process — check --start_frame / --end_frame / --frame_interval.")
        cap.release()
        return

    # ---- output directories ----
    out_dir = Path(args.output)
    depth_dir = out_dir / "depth"
    viz_dir = out_dir / "viz"
    if args.save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- run inference ----
    times_depth: list[float] = []
    viz_frames: list[np.ndarray] = []

    print("\nStarting inference …")
    print("-" * 60)

    # Seek to the first requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])

    for step, frame_idx in enumerate(frame_indices):
        # If we need to skip ahead (e.g. after a gap), seek explicitly
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx}: read failed — stopping.")
            break

        # ---- monocular depth ----
        t0 = time.time()
        pred_depth, depth_conf, scale_factor = depth_model.infer(frame)
        elapsed_ms = (time.time() - t0) * 1000.0
        times_depth.append(elapsed_ms)

        # Report
        valid_frac = (pred_depth > 0).mean()
        print(
            f"[{step + 1:4d}/{n_frames:4d}]  "
            f"idx={frame_idx:6d}  "
            f"depth={pred_depth[pred_depth > 0].mean():5.2f}±{pred_depth[pred_depth > 0].std():4.2f}m  "
            f"valid={valid_frac:.1%}  "
            f"time={elapsed_ms:6.1f}ms"
        )

        # ---- save depth array ----
        if args.save_depth:
            npz_path = depth_dir / f"depth_{frame_idx:06d}.npz"
            np.savez_compressed(
                npz_path,
                metric_depth=pred_depth.astype(np.float32),
                confidence=depth_conf.astype(np.float32),
                scale_factor=np.float32(scale_factor),
            )

        # ---- visualisation frame ----
        if args.save_viz:
            depth_colour = colour_map_depth(
                pred_depth, vmin=args.depth_vmin, vmax=args.depth_vmax,
            )
            # Resize depth colour map to match source frame height
            if depth_colour.shape[:2] != frame.shape[:2]:
                depth_colour = cv2.resize(
                    depth_colour, (frame.shape[1], frame.shape[0]),
                )
            side_by_side = np.hstack([frame, depth_colour])
            viz_frames.append(side_by_side)

    cap.release()

    # ---- summary ----
    print("-" * 60)
    if times_depth:
        times = np.array(times_depth)
        print(f"Depth inference — mean: {times.mean():.1f}ms  "
              f"median: {np.median(times):.1f}ms  "
              f"min: {times.min():.1f}ms  max: {times.max():.1f}ms")
    print(f"Frames processed: {len(times_depth)} / {n_frames}")

    # ---- write visualisation video ----
    if args.save_viz and viz_frames:
        viz_video_path = viz_dir / "depth_visualisation.mp4"
        write_video(viz_frames, str(viz_video_path), fps / args.frame_interval)

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
