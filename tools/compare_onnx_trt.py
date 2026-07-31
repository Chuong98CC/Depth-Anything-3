#!/usr/bin/env python3
"""Compare ONNX vs TensorRT across the metric, any-view, and nested modules.

Loads an Astribot camera set / frame and, for each requested module, runs the
ONNX and TensorRT wrappers on identical inputs and reports per-output error
statistics.  All pre/post-processing comes from the shared wrapper classes
(``DA3Metric*``, ``DA3AnyView*``, ``DA3Nested*``) so the two backends differ only
in the inference engine.

Usage:
    python tools/compare_onnx_trt.py \\
        --onnx-anyview weights/da3_anyview_n3_644x490_giant-large-1.1.onnx \\
        --trt-anyview  weights/da3_anyview_n3_644x490_giant-large-1.1.engine \\
        --onnx-metric  weights/da3_metric_644x490_giant-large-1.1.onnx \\
        --trt-metric   weights/da3_metric_644x490_giant-large-1.1.engine \\
        --camera-set set1 --frame-idx 0
"""

from __future__ import annotations

import argparse
import gc

import numpy as np

from astribot_dataloader import load_images_cam_params
from model.da3anyview import DA3AnyViewModel, DA3AnyViewONNX
from model.da3metric import DA3MetricModel, DA3MetricONNX
from model.da3nested import DA3NestedModel, DA3NestedONNX

ALL_MODULES = ["metric", "anyview", "nested"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _compare(onx: dict, trt: dict, label: str, keys: list[str]) -> None:
    """Print per-output ONNX-vs-TRT error statistics for the given keys."""
    print(f"\n{'=' * 72}")
    print(f"[COMPARE] {label}")
    print(f"{'-' * 72}")

    for key in keys:
        if key not in onx or key not in trt:
            print(f"  SKIP {key}: missing from one side")
            continue

        o = np.asarray(onx[key])
        t = np.asarray(trt[key])
        if o.shape != t.shape:
            print(f"  WARN {key}: shape ONNX={list(o.shape)} TRT={list(t.shape)}")
            mins = tuple(min(a, b) for a, b in zip(o.shape, t.shape))
            o = o[tuple(slice(0, s) for s in mins)]
            t = t[tuple(slice(0, s) for s in mins)]

        a = o.astype(np.float64)
        b = t.astype(np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        abs_diff = np.abs(a - b)[valid]
        rel_err = abs_diff / (np.abs(a[valid]) + 1e-8)

        # Median (robust) alongside max: fp16 tail noise inflates max_* but the
        # median reflects the true agreement between the two backends.
        print(
            f"  {key:12s} "
            f"max_abs={float(abs_diff.max()):.3e}  "
            f"med_abs={float(np.median(abs_diff)):.3e}  "
            f"max_rel={float(rel_err.max()):.3e}  "
            f"med_rel={float(np.median(rel_err)):.3e}  "
            f"shape={list(o.shape)}"
        )

    print(f"{'-' * 72}")
    for key in keys:
        if key in onx and key in trt and np.asarray(onx[key]).shape == np.asarray(trt[key]).shape:
            ok = np.allclose(onx[key], trt[key], atol=5e-3)
            print(f"[COMPARE] {key:12s} allclose(atol=5e-3): {ok}")
    print(f"{'=' * 72}")


def _cuda_gc() -> None:
    """Release GPU memory held by just-deleted ONNX sessions / TRT engines."""
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-module comparisons
# ---------------------------------------------------------------------------


def compare_metric(
    onnx_path: str,
    trt_path: str,
    image_path: str,
    intrinsics: np.ndarray,
) -> None:
    """Single-image metric ONNX vs TRT (depth + sky) on one letterboxed view.

    Uses the view's real camera intrinsics ``(3, 3)`` — monocular metric depth
    derives its focal as ``(fx + fy) / 2`` from these.
    """
    onnx = DA3MetricONNX(onnx_path, "cuda")
    trt = DA3MetricModel(trt_path)

    img_batch, _, _ = onnx.preprocess_views([image_path], intrinsics[None])
    chw = img_batch[0, 0]  # identical preprocessed input for both backends

    depth_o, sky_o = onnx.infer_view(chw)
    depth_t, sky_t = trt.infer_view(chw)
    _compare(
        {"depth": depth_o, "sky": sky_o},
        {"depth": depth_t, "sky": sky_t},
        "METRIC  (ONNX vs TRT)",
        ["depth", "sky"],
    )

    del onnx, trt
    _cuda_gc()


def compare_anyview(
    onnx_path: str,
    trt_path: str,
    images: list[str],
    exts: np.ndarray,
    ixts: np.ndarray,
) -> None:
    """Any-view ONNX vs TRT (depth / conf / extrinsics / intrinsics)."""
    onnx = DA3AnyViewONNX(onnx_path, "cuda")
    trt = DA3AnyViewModel(trt_path)

    _check_view_count(onnx.num_views, len(images), "any-view")

    o = onnx.infer(images, exts, ixts)
    t = trt.infer(images, exts, ixts)
    _compare(o, t, "ANYVIEW  (ONNX vs TRT)", ["depth", "depth_conf", "extrinsics", "intrinsics"])

    del onnx, trt
    _cuda_gc()


def compare_nested(
    onnx_anyview: str,
    onnx_metric: str,
    trt_anyview: str,
    trt_metric: str,
    images: list[str],
    exts: np.ndarray,
    ixts: np.ndarray,
) -> None:
    """Full nested pipeline ONNX vs TRT (any-view + metric + alignment)."""
    onnx = DA3NestedONNX(onnx_anyview, onnx_metric, "cuda")
    trt = DA3NestedModel(trt_anyview, trt_metric)

    _check_view_count(onnx.av.num_views, len(images), "nested any-view")

    o = onnx.infer(images, exts, ixts)
    t = trt.infer(images, exts, ixts)
    _compare(o, t, "NESTED  (ONNX vs TRT)", ["depth", "depth_conf", "extrinsics", "intrinsics"])

    del onnx, trt
    _cuda_gc()


def _check_view_count(model_views: int | None, n_images: int, what: str) -> None:
    """Fail early if the camera set's view count doesn't match the fixed export."""
    if model_views is not None and n_images != model_views:
        raise SystemExit(
            f"[ERROR] Camera set provides {n_images} views, but the {what} model "
            f"was exported for {model_views}. Pick a --camera-set with "
            f"{model_views} views (set0=2, set1=3, set2=4) or re-export."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ONNX vs TensorRT on the metric, any-view, and nested modules.",
    )
    parser.add_argument("--onnx-anyview",
                        default="weights/da3_anyview_n3_644x490_giant-large-1.1.onnx",
                        type=str, help="Any-view ONNX path.")
    parser.add_argument("--trt-anyview",
                        default="weights/da3_anyview_n3_644x490_giant-large-1.1.engine",
                        type=str, help="Any-view TRT engine path.")
    parser.add_argument("--onnx-metric",
                        default="weights/da3_metric_644x490_giant-large-1.1.onnx",
                        type=str, help="Metric ONNX path.")
    parser.add_argument("--trt-metric", 
                        default="weights/da3_metric_644x490_giant-large-1.1.engine",
                        type=str, help="Metric TRT engine path.")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=ALL_MODULES,
        default=ALL_MODULES,
        help="Which modules to compare (default: all three).",
    )
    parser.add_argument(
        "--camera-set",
        default="set1",
        choices=["set0", "set1", "set2"],
        help="Astribot camera set (view count must match the any-view export).",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to load (0-based).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Preserve a stable run order (cheapest first) regardless of arg order.
    modules = [m for m in ALL_MODULES if m in args.modules]

    print(f"[DATA] Loading Astribot {args.camera_set} / frame {args.frame_idx} ...")
    images, exts, ixts = load_images_cam_params(args.camera_set, frame_idx=args.frame_idx)
    print(f"[DATA] {len(images)} views loaded")

    if "metric" in modules:
        compare_metric(args.onnx_metric, args.trt_metric, images[0], ixts[0])

    if "anyview" in modules:
        compare_anyview(args.onnx_anyview, args.trt_anyview, images, exts, ixts)

    if "nested" in modules:
        compare_nested(
            args.onnx_anyview,
            args.onnx_metric,
            args.trt_anyview,
            args.trt_metric,
            images,
            exts,
            ixts,
        )


if __name__ == "__main__":
    main()
