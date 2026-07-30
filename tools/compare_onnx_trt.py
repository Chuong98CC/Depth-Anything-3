#!/usr/bin/env python3
"""
Compare any-view ONNX vs TensorRT outputs using real multi-view data.

Loads Astribot set1 / frame 0, preprocesses identically for both backends,
and reports per-output error statistics.

Usage:
    python tools/compare_onnx_trt.py \
        --onnx-path weights/da3_anyview_n3_644x490_small2.onnx \
        --trt-path  weights/da3_anyview_n3_644x490_small2.engine/da3_anyview_n3_644x490_small2_fp16.engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# ImageNet normalisation — must match ONNX export / TRT wrapper
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers: identical preprocessing for ONNX and TRT paths
# ---------------------------------------------------------------------------


def preprocess_views(
    image_paths: list[str],
    intrs: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Preprocess *N* views into a model-ready batch.

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to *N* JPEG / PNG images.
    intrs : np.ndarray
        ``(N, 3, 3)`` camera intrinsics at original resolution.
    target_h, target_w : int
        Target spatial dimensions (must match ONNX / TRT input).

    Returns
    -------
    img_batch : np.ndarray
        ``(1, N, 3, H, W)`` float32, ImageNet-normalised.
    intrs_adj : np.ndarray
        ``(N, 3, 3)`` float32, intrinsics re-scaled to *target* size.
    metas : list[dict]
        Per-view metadata (original size, scale factors).
    """
    N = len(image_paths)
    proc = np.zeros((N, 3, target_h, target_w), dtype=np.float32)
    intrs_out = np.zeros((N, 3, 3), dtype=np.float32)
    metas: list[dict] = []

    for i, path in enumerate(image_paths):
        proc[i], intrs_out[i], meta = _preprocess_one(path, intrs[i], target_h, target_w)
        metas.append(meta)

    return proc[None], intrs_out, metas  # add B=1


def _preprocess_one(
    path: str,
    K: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Preprocess a single view."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load: {path}")

    orig_h, orig_w = img_bgr.shape[:2]

    # RGB + resize to exact target
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Normalise
    img_f = img_resized.astype(np.float32) / 255.0
    img_norm = (img_f - _MEAN) / _STD  # (H, W, 3)
    img_chw = img_norm.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)

    # Scale intrinsics
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    K_adj = K.copy().astype(np.float32)
    K_adj[0, 0] *= scale_x
    K_adj[0, 2] *= scale_x
    K_adj[1, 1] *= scale_y
    K_adj[1, 2] *= scale_y

    meta = {"orig_h": orig_h, "orig_w": orig_w, "scale_x": scale_x, "scale_y": scale_y}
    return img_chw, K_adj, meta


# ---------------------------------------------------------------------------
# Data loading (mirrors astribot_dataloader)
# ---------------------------------------------------------------------------


def load_example_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load Astribot set1 / frame 0."""
    # Late import — the dataloader lives under tools/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from astribot_dataloader import load_images_cam_params  # noqa: PLC0415

    images, exts_np, ixts_np = load_images_cam_params("set1", 0)
    print(f"[DATA] Loaded {len(images)} views:")
    for p in images:
        print(f"  {p}")
    return images, exts_np, ixts_np


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def run_onnx(
    onnx_path: str,
    img_batch: np.ndarray,
    extrs: np.ndarray,
    intrs: np.ndarray,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Run the any-view ONNX model and return named outputs."""
    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[ONNX]  Provider: {sess.get_providers()[0]}")

    # Resolve output names dynamically
    onnx_output_names = [o.name for o in sess.get_outputs()]

    onnx_inputs = {
        "image": img_batch.astype(np.float32),
        "extrinsics": extrs[None].astype(np.float32),  # add B=1
        "intrinsics": intrs[None].astype(np.float32),
    }
    outputs = sess.run(onnx_output_names, onnx_inputs)
    return dict(zip(onnx_output_names, outputs))


# ---------------------------------------------------------------------------
# TRT inference
# ---------------------------------------------------------------------------


def run_trt(
    trt_path: str,
    img_batch: np.ndarray,
    extrs: np.ndarray,
    intrs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the any-view TRT engine and return named outputs.

    Uses :class:`DA3AnyViewModel` for TRT lifecycle management.
    """
    from model.da3anyview import DA3AnyViewModel  # noqa: PLC0415

    # TRT engine target H, W is read from the engine itself
    model = DA3AnyViewModel(trt_path)

    # DA3AnyViewModel now inherits the generic TRTModel._run; feed the
    # already-preprocessed batch and add the batch dim to extrs/intrs.
    raw = model._run(
        {
            "image": img_batch.astype(np.float32),
            "extrinsics": extrs[None].astype(np.float32),
            "intrinsics": intrs[None].astype(np.float32),
        }
    )
    return raw


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_outputs(
    onx: dict[str, np.ndarray],
    trt: dict[str, np.ndarray],
) -> None:
    """Print per-output error statistics between ONNX and TRT."""

    def _find_key(needle: str, d: dict) -> str | None:
        """Case-insensitive key lookup."""
        for k in d:
            if needle in k.lower():
                return k
        return None

    # Normalise key names for consistent reporting
    key_pairs = [
        ("depth", "depth"),
        ("depth_conf", "conf"),
        ("extrinsics", "extrinsics"),
        ("intrinsics", "intrinsics"),
    ]

    print("\n" + "=" * 72)
    print("[COMPARE] ONNX vs TRT — per-output error statistics")
    print("-" * 72)

    for onx_key_hint, trt_key_hint in key_pairs:
        onx_k = _find_key(onx_key_hint, onx)
        trt_k = _find_key(trt_key_hint, trt)
        if onx_k is None or trt_k is None:
            print(f"  SKIP {onx_key_hint}: missing in one backend")
            continue

        onx_val = onx[onx_k]
        trt_val = trt[trt_k]

        if onx_val.shape != trt_val.shape:
            print(
                f"  WARN {onx_key_hint}: shape mismatch "
                f"ONNX={list(onx_val.shape)}  TRT={list(trt_val.shape)}"
            )
            # Try to compare anyway on overlapping region
            min_shape = tuple(min(a, b) for a, b in zip(onx_val.shape, trt_val.shape))
            onx_val = onx_val[tuple(slice(0, s) for s in min_shape)]
            trt_val = trt_val[tuple(slice(0, s) for s in min_shape)]

        abs_diff = np.abs(onx_val.astype(np.float64) - trt_val.astype(np.float64))
        rel_err = abs_diff / (np.abs(onx_val.astype(np.float64)) + 1e-8)
        valid = np.isfinite(onx_val) & np.isfinite(trt_val)

        print(
            f"  {onx_key_hint:15s} "
            f"max_abs={float(abs_diff[valid].max()):.4e}  "
            f"mean_abs={float(abs_diff[valid].mean()):.4e}  "
            f"max_rel={float(rel_err[valid].max()):.4e}  "
            f"mean_rel={float(rel_err[valid].mean()):.4e}  "
            f"shape={list(onx_val.shape)}"
        )

    # Allclose checks
    print("-" * 72)
    for label, hint in [("depth", "depth"), ("depth_conf", "conf")]:
        onx_k = _find_key(hint, onx)
        trt_k = _find_key(hint, trt)
        if onx_k is None or trt_k is None:
            continue
        ok = np.allclose(onx[onx_k], trt[trt_k], atol=5e-3)
        print(f"[COMPARE] {label:15s} allclose(atol=5e-3): {ok}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare any-view ONNX vs TRT on Astribot set1 / frame 0",
    )
    parser.add_argument(
        "--onnx-path", required=True, type=str, help="Path to any-view ONNX model."
    )
    parser.add_argument("--trt-path", required=True, type=str, help="Path to any-view TRT engine.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX Runtime device (TRT always uses CUDA).",
    )
    args = parser.parse_args()

    # 1. Load data ----------------------------------------------------------
    images, exts, ixts = load_example_data()
    orig_intrs = ixts.copy()  # (N, 3, 3) at original resolution

    # 2. Determine target size from ONNX model ------------------------------
    sess_tmp = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    img_input = sess_tmp.get_inputs()[0]
    img_shape = img_input.shape  # e.g. [1, 3, 490, 644] for BNCHW or [1, "N", 3, 490, 644]
    # ONNX symbolic shapes use strings for dynamic dims
    if isinstance(img_shape[3], int) and isinstance(img_shape[4], int):
        target_h, target_w = img_shape[3], img_shape[4]
    elif len(img_shape) == 5 and isinstance(img_shape[3], int) and isinstance(img_shape[4], int):
        target_h, target_w = img_shape[3], img_shape[4]
    else:
        # Fallback: look at the TRT engine input
        raise RuntimeError(
            f"Cannot determine concrete H,W from ONNX input shape {img_shape}. "
            "Pass --height / --width explicitly."
        )
    print(f"[INFO]  Target resolution: {target_h}x{target_w}")
    del sess_tmp

    # 3. Preprocess once, share between backends ----------------------------
    img_batch, ixts_adj, metas = preprocess_views(images, orig_intrs, target_h, target_w)
    print(f"[PREP]  Image batch shape: {list(img_batch.shape)}")

    # 4. ONNX inference -----------------------------------------------------
    print("\n[ONNX]  Running inference ...")
    onx_out = run_onnx(args.onnx_path, img_batch, exts, ixts_adj, device=args.device)

    # 5. TRT inference ------------------------------------------------------
    print("\n[TRT]   Running inference ...")
    trt_out = run_trt(args.trt_path, img_batch, exts, ixts_adj)

    # 6. Compare -----------------------------------------------------------
    compare_outputs(onx_out, trt_out)


if __name__ == "__main__":
    main()
