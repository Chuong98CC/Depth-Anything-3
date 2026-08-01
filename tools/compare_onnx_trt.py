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

from tools.utils.astribot_dataloader import camera_set_for_views, load_images_cam_params
from model.da3anyview import DA3AnyViewTRT, DA3AnyViewONNX
from model.da3metric import DA3MetricTRT, DA3MetricONNX
from model.da3nested import DA3NestedTRT, DA3NestedONNX

ALL_MODULES = ["metric", "anyview", "nested"]

# Any-view defaults: plain (model predicts poses) vs the "-with-camera-pose"
# export selected by --use-extrinsics (consumes extrinsics/intrinsics as priors).
DEFAULT_ANYVIEW = {
    ("onnx", False): "weights/da3_anyview_n3_644x490_giant-large-1.1.onnx",
    ("onnx", True): "weights/da3_anyview_n3_644x490_giant-large-1.1-with-camera-pose.onnx",
    ("trt", False): "weights/da3_anyview_n3_644x490_giant-large-1.1.engine",
    ("trt", True): "weights/da3_anyview_n3_644x490_giant-large-1.1-with-camera-pose.engine",
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _compare(
    onx: dict,
    trt: dict,
    label: str,
    keys: list[str],
    atol: float = 5e-3,
    rtol: float = 1e-3,
    nonzero_eps: float = 1e-6,
) -> None:
    """Print per-output ONNX-vs-TRT error statistics for the given keys.

    Relative error is reported over **non-zero reference elements only** — the
    extrinsic/intrinsic matrices are structurally sparse, so dividing by a ~0
    reference makes ``max_rel`` meaningless.  Median (robust) sits alongside max
    because fp16 tail noise inflates ``max_*``; ``within_atol`` gives the fraction
    of elements that agree so a lone outlier does not hide a good match.  The pass
    line uses the combined ``|a-b| <= atol + rtol*|b|`` so pixel-scale intrinsics
    are judged relatively.
    """
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
        av = a[valid]
        abs_diff = np.abs(av - b[valid])

        nz = np.abs(av) > nonzero_eps
        if nz.any():
            rel_err = abs_diff[nz] / np.abs(av[nz])
            max_rel, med_rel = float(rel_err.max()), float(np.median(rel_err))
        else:
            max_rel = med_rel = 0.0

        frac_ok = float((abs_diff <= atol).mean())
        print(
            f"  {key:12s} "
            f"max_abs={float(abs_diff.max()):.3e}  "
            f"med_abs={float(np.median(abs_diff)):.3e}  "
            f"p99_abs={float(np.percentile(abs_diff, 99)):.3e}  "
            f"max_rel={max_rel:.3e}  "
            f"med_rel={med_rel:.3e}  "
            f"within_atol={100 * frac_ok:.4f}%  "
            f"nz={int(nz.sum())}/{av.size}"
        )

    print(f"{'-' * 72}")
    for key in keys:
        if key in onx and key in trt and np.asarray(onx[key]).shape == np.asarray(trt[key]).shape:
            a = np.asarray(onx[key]).astype(np.float64)
            b = np.asarray(trt[key]).astype(np.float64)
            diff = np.abs(a - b)
            tol = atol + rtol * np.abs(b)
            bad = diff > tol
            ok = not bool(bad.any())
            tail = (
                ""
                if ok
                else f"  ({int(bad.sum())}/{diff.size} over tol; p99.9={np.percentile(diff, 99.9):.2e})"
            )
            print(f"[COMPARE] {key:12s} allclose(atol={atol:g}, rtol={rtol:g}): {ok}{tail}")
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
    trt = DA3MetricTRT(trt_path)

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
    trt = DA3AnyViewTRT(trt_path)

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
    align_scale: bool = True,
) -> None:
    """Full nested pipeline ONNX vs TRT (any-view + metric + alignment)."""
    onnx = DA3NestedONNX(onnx_anyview, onnx_metric, "cuda")
    trt = DA3NestedTRT(trt_anyview, trt_metric)

    _check_view_count(onnx.av.num_views, len(images), "nested any-view")

    # Align to input poses only with a camera-pose model; otherwise keep predicted
    # poses (matches the no-extrinsics inference path).
    exts_in = exts if onnx.av.uses_extrinsics else None
    o = onnx.infer(images, exts_in, ixts, align_scale=align_scale)
    t = trt.infer(images, exts_in, ixts, align_scale=align_scale)
    _compare(o, t, "NESTED  (ONNX vs TRT)", ["depth", "depth_conf", "extrinsics", "intrinsics"])

    del onnx, trt
    _cuda_gc()


def _num_views_from_onnx(onnx_path: str) -> int | None:
    """Read the fixed view count N from an any-view ONNX input shape (B, N, 3, H, W).

    Loads graph metadata only (no external weights), so it's cheap enough to size
    the camera set before building the heavy engines/sessions.
    """
    import onnx  # noqa: PLC0415

    model = onnx.load(onnx_path, load_external_data=False)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) == 5:  # (B, N, 3, H, W) — any-view
        n = dims[1].dim_value
        return n if n > 0 else None
    return None


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
    parser.add_argument(
        "--use-extrinsics",
        action="store_true",
        help="Compare the '-with-camera-pose' any-view model (consumes camera "
        "extrinsics/intrinsics as priors). Selects those ONNX/engine checkpoints by "
        "default; both wrappers feed poses only if the loaded model declares them.",
    )
    parser.add_argument("--onnx-anyview",
                        default=None,
                        type=str,
                        help="Any-view ONNX path (default: --use-extrinsics variant).")
    parser.add_argument("--trt-anyview",
                        default=None,
                        type=str,
                        help="Any-view TRT engine path (default: --use-extrinsics variant).")
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
        default=None,
        choices=["set0", "set1", "set2"],
        help="Astribot camera set (view count must match the any-view export). "
        "Default: auto-selected from the any-view ONNX model's view count.",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to load (0-based).")
    parser.add_argument(
        "--keep-predicted-pose",
        dest="align_scale",
        action="store_false",
        help="Nested with camera pose: keep predicted poses aligned into the input "
        "frame (align_scale=False) instead of replacing them with the input poses "
        "and rescaling depth. Applied to both the ONNX and TRT sides.",
    )
    parser.set_defaults(align_scale=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_anyview = args.onnx_anyview or DEFAULT_ANYVIEW[("onnx", args.use_extrinsics)]
    trt_anyview = args.trt_anyview or DEFAULT_ANYVIEW[("trt", args.use_extrinsics)]
    # Preserve a stable run order (cheapest first) regardless of arg order.
    modules = [m for m in ALL_MODULES if m in args.modules]

    # Resolve camera set: honour --camera-set, else match the any-view model views.
    if args.camera_set is not None:
        camera_set = args.camera_set
    elif any(m in ("anyview", "nested") for m in modules):
        nv = _num_views_from_onnx(onnx_anyview)
        camera_set = camera_set_for_views(nv) if nv else "set1"
        print(f"[DATA] Auto-selected --camera-set {camera_set} for {nv} views.")
    else:
        camera_set = "set1"  # metric only needs one view

    print(f"[DATA] Loading Astribot {camera_set} / frame {args.frame_idx} ...")
    images, exts, ixts = load_images_cam_params(camera_set, frame_idx=args.frame_idx)
    print(f"[DATA] {len(images)} views loaded")

    if "metric" in modules:
        compare_metric(args.onnx_metric, args.trt_metric, images[0], ixts[0])

    if "anyview" in modules:
        compare_anyview(onnx_anyview, trt_anyview, images, exts, ixts)

    if "nested" in modules:
        compare_nested(
            onnx_anyview,
            args.onnx_metric,
            trt_anyview,
            args.trt_metric,
            images,
            exts,
            ixts,
            align_scale=args.align_scale,
        )


if __name__ == "__main__":
    main()
