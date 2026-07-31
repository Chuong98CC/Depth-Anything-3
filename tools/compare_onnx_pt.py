#!/usr/bin/env python3
"""Compare PyTorch vs ONNX across the metric, any-view, and nested modules.

Loads an Astribot camera set / frame and, for each requested module, runs the
PyTorch reference forward and the corresponding ONNX wrapper on identical
letterbox-preprocessed inputs, then reports per-output error statistics.  The
ONNX side uses the shared wrapper classes (``DA3MetricONNX``, ``DA3AnyViewONNX``,
``DA3NestedONNX``); the PyTorch side runs the matching sub-model of a single
nested checkpoint (``depth-anything/DA3NESTED-GIANT-LARGE-1.1``).

The giant PyTorch model and the ONNX CUDA session never co-reside on the GPU:
each module runs ONNX first (model offloaded to CPU), frees it, then runs the
PyTorch forward, avoiding the any-view attention-Softmax OOM.

Usage:
    python tools/compare_onnx_pt.py \\
        --onnx-anyview weights/da3_anyview_n3_644x490_giant-large-1.1.onnx \\
        --onnx-metric  weights/da3_metric_644x490_giant-large-1.1.onnx \\
        --camera-set set1 --frame-idx 0
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import torch

from astribot_dataloader import load_images_cam_params
from depth_anything_3.api import DepthAnything3
from model.base_da3 import BaseDA3Model
from model.da3anyview import DA3AnyViewONNX
from model.da3metric import DA3MetricONNX
from model.da3nested import DA3NestedONNX

ALL_MODULES = ["metric", "anyview", "nested"]
DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# Bare mixin instance for reusing the shared post-processing (align_to_input) on
# the PyTorch side without an ONNX session.
_BASE = BaseDA3Model()


# ---------------------------------------------------------------------------
# Sub-model accessors (nested checkpoint → any-view / metric branch)
# ---------------------------------------------------------------------------


def _get_da3_submodel(api_model: DepthAnything3) -> torch.nn.Module:
    """Any-view sub-model: ``.model.da3`` (nested) or ``.model`` (plain)."""
    return getattr(api_model.model, "da3", api_model.model)


def _get_metric_submodel(api_model: DepthAnything3) -> torch.nn.Module:
    """Metric sub-model: ``.model.da3_metric`` (nested) or ``.model`` (plain)."""
    return getattr(api_model.model, "da3_metric", api_model.model)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _compare(pt: dict, onx: dict, label: str, keys: list[str]) -> None:
    """Print per-output PyTorch-vs-ONNX error statistics for the given keys."""
    print(f"\n{'=' * 72}")
    print(f"[COMPARE] {label}")
    print(f"{'-' * 72}")

    for key in keys:
        if key not in pt or key not in onx:
            print(f"  SKIP {key}: missing from one side")
            continue

        p = np.asarray(pt[key])
        o = np.asarray(onx[key])
        if p.shape != o.shape:
            print(f"  WARN {key}: shape PT={list(p.shape)} ONNX={list(o.shape)}")
            mins = tuple(min(a, b) for a, b in zip(p.shape, o.shape))
            p = p[tuple(slice(0, s) for s in mins)]
            o = o[tuple(slice(0, s) for s in mins)]

        a = p.astype(np.float64)
        b = o.astype(np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        abs_diff = np.abs(a - b)[valid]
        rel_err = abs_diff / (np.abs(a[valid]) + 1e-8)

        print(
            f"  {key:12s} "
            f"max_abs={float(abs_diff.max()):.3e}  "
            f"med_abs={float(np.median(abs_diff)):.3e}  "
            f"max_rel={float(rel_err.max()):.3e}  "
            f"med_rel={float(np.median(rel_err)):.3e}  "
            f"shape={list(p.shape)}"
        )

    print(f"{'-' * 72}")
    for key in keys:
        if key in pt and key in onx and np.asarray(pt[key]).shape == np.asarray(onx[key]).shape:
            ok = np.allclose(pt[key], onx[key], atol=1e-2)
            print(f"[COMPARE] {key:12s} allclose(atol=1e-2): {ok}")
    print(f"{'=' * 72}")


def _cuda_gc() -> None:
    """Release GPU memory held by a just-deleted ONNX session / offloaded model."""
    gc.collect()
    torch.cuda.empty_cache()


def _check_view_count(model_views: int | None, n_images: int, what: str) -> None:
    """Fail early if the camera set's view count doesn't match the fixed export."""
    if model_views is not None and n_images != model_views:
        raise SystemExit(
            f"[ERROR] Camera set provides {n_images} views, but the {what} model "
            f"was exported for {model_views}. Pick a --camera-set with "
            f"{model_views} views (set0=2, set1=3, set2=4) or re-export."
        )


# ---------------------------------------------------------------------------
# PyTorch reference forwards (fed the wrapper's letterbox-preprocessed inputs)
# ---------------------------------------------------------------------------


def _pt_metric(api_model: DepthAnything3, chw: np.ndarray) -> dict[str, np.ndarray]:
    """Full metric forward on one ``(3, H, W)`` view → ``{depth, sky}`` (H, W).

    Runs the complete ``metric_model.forward`` — including the mono-sky depth
    post-processing — so it validates the ONNX/TRT wrapper's ``infer_view`` with
    ``apply_mono_sky=True`` (which replicates that step outside the graph).
    """
    dev = torch.device("cuda")
    img_t = torch.from_numpy(chw)[None, None].to(dev).float()  # (1, 1, 3, H, W)
    metric_model = _get_metric_submodel(api_model)
    with torch.no_grad():
        out = metric_model(
            img_t, extrinsics=None, intrinsics=None, export_feat_layers=[], infer_gs=False
        )
    return {
        "depth": out["depth"].float().cpu().numpy().squeeze(),
        "sky": out["sky"].float().cpu().numpy().squeeze(),
    }


def _pt_anyview(
    api_model: DepthAnything3,
    img_batch: np.ndarray,
    intrs_adj: np.ndarray,
    exts_norm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Any-view sub-model → ``{depth, depth_conf, extrinsics, intrinsics}`` (1, N, …)."""
    dev = torch.device("cuda")
    imgs_t = torch.from_numpy(img_batch).to(dev).float()  # (1, N, 3, H, W)
    ex_t = torch.from_numpy(exts_norm[None]).to(dev).float()  # (1, N, 4, 4)
    in_t = torch.from_numpy(intrs_adj[None]).to(dev).float()  # (1, N, 3, 3)
    with torch.no_grad():
        out = _get_da3_submodel(api_model)(
            imgs_t, extrinsics=ex_t, intrinsics=in_t, export_feat_layers=[], infer_gs=False
        )
    return {
        k: out[k].float().cpu().numpy()
        for k in ["depth", "depth_conf", "extrinsics", "intrinsics"]
    }


def _pt_nested(
    api_model: DepthAnything3,
    img_batch: np.ndarray,
    intrs_adj: np.ndarray,
    exts: np.ndarray,
    metas: list[dict],
) -> dict[str, np.ndarray]:
    """Full nested model (any-view + metric + alignment), cropped to the tile.

    ``api_model.model`` reproduces the metric-alignment step but returns uncropped
    letterbox outputs without the Umeyama input-pose alignment.  ``DA3NestedONNX``
    exports/infers with ``align_input_ext_scale=True`` by default, so the same
    alignment (via ``BaseDA3Model.align_to_input``, using the raw input extrinsics
    and the letterbox-scaled intrinsics) is applied here before cropping with the
    same per-view metas the wrapper uses.
    """
    dev = torch.device("cuda")
    exts_norm = BaseDA3Model.normalize_extrinsics(exts)
    imgs_t = torch.from_numpy(img_batch).to(dev).float()
    ex_t = torch.from_numpy(exts_norm[None]).to(dev).float()
    in_t = torch.from_numpy(intrs_adj[None]).to(dev).float()
    with torch.no_grad():
        out = api_model.model(
            imgs_t, extrinsics=ex_t, intrinsics=in_t, export_feat_layers=[], infer_gs=False
        )
    result = {
        k: out[k].float().cpu().numpy()
        for k in ["depth", "depth_conf", "extrinsics", "intrinsics"]
    }
    # Squeeze the batch dim (B=1) → per-view arrays.
    result = {k: (v[0] if v.shape[0] == 1 else v) for k, v in result.items()}

    # Umeyama align to the input poses — matches DA3NestedONNX's default
    # align_input_ext_scale=True (raw extrinsics + letterbox-scaled intrinsics).
    result = _BASE.align_to_input(result, exts, intrs_adj)

    # Crop padded depth/conf to the tile and un-pad the intrinsics (mirrors
    # DA3NestedONNX._crop_result).
    depth = np.stack(
        [BaseDA3Model.crop_to_tile(result["depth"][i], metas[i]) for i in range(len(metas))]
    )
    conf = np.stack(
        [BaseDA3Model.crop_to_tile(result["depth_conf"][i], metas[i]) for i in range(len(metas))]
    )
    intr = result["intrinsics"].copy()
    for i, m in enumerate(metas):
        intr[i, 0, 2] -= m["pad_left"]
        intr[i, 1, 2] -= m["pad_top"]
    return {
        "depth": depth,
        "depth_conf": conf,
        "extrinsics": result["extrinsics"],
        "intrinsics": intr,
    }


# ---------------------------------------------------------------------------
# Per-module comparisons (ONNX first, then PyTorch — never GPU co-resident)
# ---------------------------------------------------------------------------


def compare_metric(
    onnx_path: str,
    api_model: DepthAnything3,
    image_path: str,
    intrinsics: np.ndarray,
) -> None:
    """Single-image metric PyTorch vs ONNX (depth + sky) on one letterboxed view."""
    onnx = DA3MetricONNX(onnx_path, "cuda")
    img_batch, _, _ = onnx.preprocess_views([image_path], intrinsics[None])
    chw = img_batch[0, 0]  # identical preprocessed input for both backends
    depth_o, sky_o = onnx.infer_view(chw)
    del onnx
    _cuda_gc()

    api_model.to("cuda")
    pt = _pt_metric(api_model, chw)
    api_model.to("cpu")
    _cuda_gc()

    _compare(pt, {"depth": depth_o, "sky": sky_o}, "METRIC  (PyTorch vs ONNX)", ["depth", "sky"])


def compare_anyview(
    onnx_path: str,
    api_model: DepthAnything3,
    images: list[str],
    exts: np.ndarray,
    ixts: np.ndarray,
) -> None:
    """Any-view PyTorch vs ONNX (depth / conf / extrinsics / intrinsics)."""
    onnx = DA3AnyViewONNX(onnx_path, "cuda")
    _check_view_count(onnx.num_views, len(images), "any-view")
    img_batch, intrs_adj, _ = onnx.preprocess_views(images, ixts)
    onx = onnx.infer(images, exts, ixts, normalize_extrinsics=True)
    del onnx
    _cuda_gc()

    exts_norm = BaseDA3Model.normalize_extrinsics(exts)
    api_model.to("cuda")
    pt = _pt_anyview(api_model, img_batch, intrs_adj, exts_norm)
    api_model.to("cpu")
    _cuda_gc()

    _compare(
        pt, onx, "ANYVIEW  (PyTorch vs ONNX)", ["depth", "depth_conf", "extrinsics", "intrinsics"]
    )


def compare_nested(
    onnx_anyview: str,
    onnx_metric: str,
    api_model: DepthAnything3,
    images: list[str],
    exts: np.ndarray,
    ixts: np.ndarray,
) -> None:
    """Full nested pipeline PyTorch vs ONNX (any-view + metric + alignment)."""
    onnx = DA3NestedONNX(onnx_anyview, onnx_metric, "cuda")
    _check_view_count(onnx.av.num_views, len(images), "nested any-view")
    img_batch, intrs_adj, metas = onnx.av.preprocess_views(images, ixts)
    onx = onnx.infer(images, exts, ixts, align_input_ext_scale=True)
    del onnx
    _cuda_gc()

    api_model.to("cuda")
    pt = _pt_nested(api_model, img_batch, intrs_adj, exts, metas)
    api_model.to("cpu")
    _cuda_gc()

    _compare(
        pt, onx, "NESTED  (PyTorch vs ONNX)", ["depth", "depth_conf", "extrinsics", "intrinsics"]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs ONNX on the metric, any-view, and nested modules.",
    )
    parser.add_argument(
        "--pt-ckpt",
        type=str,
        default=DEFAULT_MODEL,
        help="PyTorch nested checkpoint (HuggingFace id or local dir); used for all modules.",
    )
    parser.add_argument(
        "--onnx-anyview",
        type=str,
        default="weights/da3_anyview_n3_644x490_giant-large-1.1.onnx",
        help="Any-view ONNX path.",
    )
    parser.add_argument(
        "--onnx-metric",
        type=str,
        default="weights/da3_metric_644x490_giant-large-1.1.onnx",
        help="Metric ONNX path.",
    )
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
    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA is required for this comparison.")
    # Preserve a stable run order (cheapest first) regardless of arg order.
    modules = [m for m in ALL_MODULES if m in args.modules]

    print(f"[DATA] Loading Astribot {args.camera_set} / frame {args.frame_idx} ...")
    images, exts, ixts = load_images_cam_params(args.camera_set, frame_idx=args.frame_idx)
    print(f"[DATA] {len(images)} views loaded")

    # Load the PyTorch reference once; kept on CPU except during its own forward.
    print(f"[PYTORCH] Loading nested model from {args.pt_ckpt} ...")
    api_model = DepthAnything3.from_pretrained(args.pt_ckpt)
    api_model.eval()

    if "metric" in modules:
        compare_metric(args.onnx_metric, api_model, images[0], ixts[0])

    if "anyview" in modules:
        compare_anyview(args.onnx_anyview, api_model, images, exts, ixts)

    if "nested" in modules:
        compare_nested(args.onnx_anyview, args.onnx_metric, api_model, images, exts, ixts)


if __name__ == "__main__":
    main()
