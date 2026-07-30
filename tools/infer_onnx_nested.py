#!/usr/bin/env python3
"""
Run nested DepthAnything3 inference via split ONNX models on the Astribot dataset.

Combines an any-view ONNX model with a metric ONNX model, aligns their outputs
using ``align_anyview_with_metric``, and saves results matching the PyTorch
``infer_pytorch.py`` output format.

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

import cv2
import numpy as np
import onnxruntime as ort
import torch

# Ensure tools/ is on sys.path for astribot_dataloader and model.* imports
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))

from astribot_dataloader import CAMERA_SETS, count_frames, load_images_cam_params  # noqa: E402
from model.alignment import align_anyview_with_metric, align_to_input_ext_scale  # noqa: E402

# ---------------------------------------------------------------------------
# Default ONNX model paths
# ---------------------------------------------------------------------------
DEFAULT_ANYVIEW_ONNX = "weights/da3_anyview_n3_644x490_giant-large-1.1.onnx"
DEFAULT_METRIC_ONNX = "weights/da3_metric_644x490_giant-large-1.1.onnx"

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==========================================================================
#  Preprocessing
# ==========================================================================

def preprocess_views(
    image_paths: list[str],
    intrs: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess *N* images to ``(1, N, 3, H, W)`` with scaled intrinsics."""
    N = len(image_paths)
    proc = np.zeros((N, 3, target_h, target_w), dtype=np.float32)
    intrs_out = np.zeros((N, 3, 3), dtype=np.float32)

    for i, path in enumerate(image_paths):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not load: {path}")
        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_f = rgb_r.astype(np.float32) / 255.0
        proc[i] = ((img_f - _MEAN) / _STD).transpose(2, 0, 1)

        # Scale intrinsics
        K = intrs[i].copy().astype(np.float32)
        sx, sy = target_w / orig_w, target_h / orig_h
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        intrs_out[i] = K

    return proc[None], intrs_out


# ==========================================================================
#  ONNX nested pipeline
# ==========================================================================

class NestedONNXInference:
    """Load two ONNX models and run the nested any-view + metric pipeline."""

    def __init__(
        self,
        onnx_anyview_path: str,
        onnx_metric_path: str,
        device: str = "cuda",
    ) -> None:
        if device == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.sess_av = ort.InferenceSession(onnx_anyview_path, providers=providers)
        self.sess_m = ort.InferenceSession(onnx_metric_path, providers=providers)
        print(f"[ONNX] Any-view provider: {self.sess_av.get_providers()[0]}")
        print(f"[ONNX] Metric   provider: {self.sess_m.get_providers()[0]}")

        # Read expected input sizes
        av_shape = self.sess_av.get_inputs()[0].shape   # (1, N, 3, H, W)
        m_shape = self.sess_m.get_inputs()[0].shape      # (1, 3, H_m, W_m)
        self.av_h = av_shape[3] if isinstance(av_shape[3], int) else 490
        self.av_w = av_shape[4] if isinstance(av_shape[4], int) else 644
        self.m_h = m_shape[2] if isinstance(m_shape[2], int) else self.av_h
        self.m_w = m_shape[3] if isinstance(m_shape[3], int) else self.av_w
        print(f"[ONNX] Any-view input: {self.av_h}x{self.av_w}")
        print(f"[ONNX] Metric   input: {self.m_h}x{self.m_w}")

    def infer(
        self,
        image_paths: list[str],
        extrs: np.ndarray,
        intrs: np.ndarray,
        align_input_ext_scale: bool = True,
    ) -> dict[str, np.ndarray]:
        """Run the full nested ONNX pipeline.

        Parameters
        ----------
        align_input_ext_scale : bool, default ``True``
            Align the prediction to the input camera poses via Umeyama Sim(3)
            (numpy post-processing, mirrors ``align_to_input_ext_scale=True`` in
            ``DepthAnything3.inference``).  When ``True`` the output extrinsics are
            the input poses and depth is rescaled to the input pose scale.

        Returns
        -------
        dict with ``depth``, ``depth_conf``, ``extrinsics``, ``intrinsics``.
        """
        N = len(image_paths)

        # ---- 1. Preprocess for any-view -----------------------------------
        img_batch, intrs_adj = preprocess_views(image_paths, intrs, self.av_h, self.av_w)

        # Normalise extrinsics (numpy replica of DepthAnything3._normalize_extrinsics)
        extrs_norm = _normalize_extrinsics_numpy(extrs)

        # ---- 2. Any-view ONNX ---------------------------------------------
        av_out = self.sess_av.run(
            [o.name for o in self.sess_av.get_outputs()],
            {
                "image": img_batch.astype(np.float32),
                "extrinsics": extrs_norm[None].astype(np.float32),
                "intrinsics": intrs_adj[None].astype(np.float32),
            },
        )
        av = _map_av_keys(dict(zip(
            [o.name for o in self.sess_av.get_outputs()], av_out,
        )))

        # ---- 3. Metric ONNX (one view at a time) --------------------------
        metric_depths = np.zeros((1, N, 1, self.av_h, self.av_w), dtype=np.float32)
        metric_skys = np.zeros((1, N, 1, self.av_h, self.av_w), dtype=np.float32)
        need_metric_resize = (self.m_h, self.m_w) != (self.av_h, self.av_w)

        for i in range(N):
            view_chw = img_batch[0, i]  # (3, H_av, W_av)
            if need_metric_resize:
                view_hwc = cv2.resize(
                    view_chw.transpose(1, 2, 0), (self.m_w, self.m_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                view_chw = view_hwc.transpose(2, 0, 1)

            m_out = self.sess_m.run(
                [o.name for o in self.sess_m.get_outputs()],
                {"image": view_chw[None].astype(np.float32)},
            )
            m_raw = dict(zip([o.name for o in self.sess_m.get_outputs()], m_out))
            m_depth, m_sky = _extract_metric(m_raw)

            if need_metric_resize:
                m_depth = cv2.resize(m_depth, (self.av_w, self.av_h),
                                     interpolation=cv2.INTER_LINEAR)
                m_sky = cv2.resize(m_sky, (self.av_w, self.av_h),
                                   interpolation=cv2.INTER_LINEAR)

            metric_depths[0, i, 0] = m_depth
            metric_skys[0, i, 0] = m_sky

        metric_depths = metric_depths.squeeze(2)  # (1, N, H, W)
        metric_skys = metric_skys.squeeze(2)

        # ---- 4. Align -----------------------------------------------------
        aligned = align_anyview_with_metric(
            anyview_depth=torch.from_numpy(av["depth"]),
            anyview_conf=torch.from_numpy(av["depth_conf"]),
            anyview_extrinsics=torch.from_numpy(av["extrinsics"]),
            anyview_intrinsics=torch.from_numpy(av["intrinsics"]),
            metric_depth=torch.from_numpy(metric_depths),
            metric_sky=torch.from_numpy(metric_skys),
        )
        # Squeeze batch dim: (1, N, ...) → (N, ...)
        result = {}
        for k in ("depth", "depth_conf", "extrinsics", "intrinsics"):
            val = aligned[k].float().cpu().numpy()
            if val.ndim >= 4 and val.shape[0] == 1:
                val = val.squeeze(0)
            result[k] = val

        # ---- 5. Align to input camera poses (numpy post-processing) -------
        if align_input_ext_scale:
            aligned_in = align_to_input_ext_scale(
                pred_depth=result["depth"],
                pred_extrinsics=result["extrinsics"],
                input_extrinsics=extrs,       # raw, un-normalised input poses
                input_intrinsics=intrs_adj,   # scaled to processing resolution
                align_scale=True,
            )
            result["depth"] = aligned_in["depth"]
            result["extrinsics"] = aligned_in["extrinsics"]
            result["intrinsics"] = aligned_in["intrinsics"]

        return result


# ==========================================================================
#  Helpers
# ==========================================================================

def _normalize_extrinsics_numpy(extrs: np.ndarray) -> np.ndarray:
    """Normalise extrinsics (numpy replica)."""
    ex_t = extrs.copy()
    transform = np.linalg.inv(ex_t[0])
    ex_t_norm = ex_t @ transform
    c2ws = np.linalg.inv(ex_t_norm)
    dists = np.linalg.norm(c2ws[..., :3, 3], axis=-1)
    median_dist = max(float(np.median(dists)), 1e-1)
    ex_t_norm[..., :3, 3] /= median_dist
    return ex_t_norm


def _map_av_keys(raw: dict) -> dict:
    """Normalise any-view ONNX output keys."""
    out: dict[str, np.ndarray] = {}
    for name, val in raw.items():
        low = name.lower()
        if "depth_conf" in low or "conf" in low:
            out["depth_conf"] = val
        elif "pred_extrinsics" in low:
            out["extrinsics"] = val
        elif "pred_intrinsics" in low:
            out["intrinsics"] = val
        elif "depth" in low:
            out["depth"] = val
        else:
            out[name] = val
    return out


def _extract_metric(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (depth, sky) from metric ONNX outputs."""
    depth = sky = None
    for name, val in raw.items():
        low = name.lower()
        arr = val.squeeze().astype(np.float32)
        if "sky" in low:
            sky = arr
        elif "depth" in low:
            depth = arr
        elif depth is None:
            depth = arr
    if sky is None:
        sky = np.zeros_like(depth)
    return depth, sky


# ==========================================================================
#  Main
# ==========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested DepthAnything3 ONNX inference on Astribot stereo dataset",
    )
    parser.add_argument(
        "--camera-set", choices=["set1", "set2"], default="set1",
        help="set1 = head_rgbd+stereo_left+stereo_right (3 views), "
             "set2 = set1+torso_rgbd (4 views).",
    )
    parser.add_argument(
        "--frame", type=int, default=None,
        help="Single frame index (0-based).  Use --all-frames to process all.",
    )
    parser.add_argument(
        "--all-frames", action="store_true",
        help="Process all frames common to the selected cameras.",
    )
    parser.add_argument(
        "--onnx-anyview", type=str, default=DEFAULT_ANYVIEW_ONNX,
        help="Path to any-view ONNX model.",
    )
    parser.add_argument(
        "--onnx-metric", type=str, default=DEFAULT_METRIC_ONNX,
        help="Path to metric ONNX model.",
    )
    parser.add_argument(
        "--export-dir", default="output_onnx",
        help="Directory to save results (NPZ per frame).",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="ONNX Runtime device.",
    )
    parser.add_argument(
        "--no-align-input-ext-scale", dest="align_input_ext_scale",
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

    # --- Load ONNX models ---
    pipeline = NestedONNXInference(args.onnx_anyview, args.onnx_metric, device=args.device)

    # --- Per-frame inference ---
    for frame_idx in frame_indices:
        print(f"\n--- Frame {frame_idx} ---")
        images, exts, ixts = load_images_cam_params(args.camera_set, frame_idx)
        print(f"  Views: {len(images)}")
        for p in images:
            print(f"    {p}")

        result = pipeline.infer(
            images, exts, ixts,
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
