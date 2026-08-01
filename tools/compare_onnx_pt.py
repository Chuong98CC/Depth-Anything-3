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

from tools.utils.astribot_dataloader import camera_set_for_views, load_images_cam_params
from tools.export_onnx import EXPORT_REF_VIEW_STRATEGY
from depth_anything_3.api import DepthAnything3
from model.base_da3 import BaseDA3Model
from model.da3anyview import DA3AnyViewONNX
from model.da3metric import DA3MetricONNX
from model.da3nested import DA3NestedONNX

ALL_MODULES = ["metric", "anyview", "nested"]
DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# Any-view ONNX defaults: plain (model predicts poses) vs the "-with-camera-pose"
# export selected by --use-extrinsics (consumes extrinsics/intrinsics as priors).
DEFAULT_ANYVIEW_ONNX = {
    False: "weights/da3_anyview_n3_644x490_giant-large-1.1.onnx",
    True: "weights/da3_anyview_n3_644x490_giant-large-1.1-with-camera-pose.onnx",
}

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


def _compare(
    pt: dict,
    onx: dict,
    label: str,
    keys: list[str],
    atol: float = 1e-2,
    rtol: float = 1e-3,
    nonzero_eps: float = 1e-6,
) -> None:
    """Print per-output PyTorch-vs-ONNX error statistics for the given keys.

    Relative error is reported over **non-zero reference elements only**: the
    extrinsic/intrinsic matrices are structurally sparse (rotation off-diagonals,
    the ``fx/fy/cx/cy`` intrinsic layout, an identity reference pose), and dividing
    by a ~0 reference makes ``max_rel`` meaningless.  Also reports ``p99`` and the
    fraction of elements within ``atol`` so a single boundary-pixel outlier (e.g. a
    sky-mask flip in the nested depth) does not hide an otherwise-exact match.

    The pass line uses ``np.allclose``'s combined tolerance ``|a-b| <= atol +
    rtol*|b|`` so large-magnitude outputs (intrinsics in pixels, ~hundreds) are
    judged relatively — a 0.1 px focal/principal difference is a ~1e-4 match, not
    a failure at an absolute 1e-2.
    """
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
        av = a[valid]
        abs_diff = np.abs(av - b[valid])

        # Relative error only where the reference is non-zero (sparse-matrix safe).
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
        if key in pt and key in onx and np.asarray(pt[key]).shape == np.asarray(onx[key]).shape:
            a = np.asarray(pt[key]).astype(np.float64)
            b = np.asarray(onx[key]).astype(np.float64)
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
    """Release GPU memory held by a just-deleted ONNX session / offloaded model."""
    gc.collect()
    torch.cuda.empty_cache()


def _num_views_from_onnx(onnx_path: str) -> int | None:
    """Read the fixed view count N from an any-view ONNX input shape (B, N, 3, H, W).

    Loads graph metadata only (no external weights), so it's cheap enough to size
    the camera set before building the heavy inference sessions.
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
    use_extrinsics: bool = True,
) -> dict[str, np.ndarray]:
    """Any-view sub-model → ``{depth, depth_conf, extrinsics, intrinsics}`` (1, N, …).

    ``extrinsics``/``intrinsics`` are passed as priors only when ``use_extrinsics``,
    matching whether the ONNX graph consumes them (a "-with-camera-pose" export).
    """
    dev = torch.device("cuda")
    imgs_t = torch.from_numpy(img_batch).to(dev).float()  # (1, N, 3, H, W)
    ex_t = in_t = None
    if use_extrinsics:
        ex_t = torch.from_numpy(exts_norm[None]).to(dev).float()  # (1, N, 4, 4)
        in_t = torch.from_numpy(intrs_adj[None]).to(dev).float()  # (1, N, 3, 3)
    with torch.no_grad():
        out = _get_da3_submodel(api_model)(
            imgs_t, extrinsics=ex_t, intrinsics=in_t, export_feat_layers=[], infer_gs=False,
            ref_view_strategy=EXPORT_REF_VIEW_STRATEGY,
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
    use_extrinsics: bool = True,
    align_scale: bool = True,
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
    imgs_t = torch.from_numpy(img_batch).to(dev).float()
    ex_t = in_t = None
    if use_extrinsics:
        exts_norm = BaseDA3Model.normalize_extrinsics(exts)
        ex_t = torch.from_numpy(exts_norm[None]).to(dev).float()
        in_t = torch.from_numpy(intrs_adj[None]).to(dev).float()
    with torch.no_grad():
        out = api_model.model(
            imgs_t, extrinsics=ex_t, intrinsics=in_t, export_feat_layers=[], infer_gs=False,
            ref_view_strategy=EXPORT_REF_VIEW_STRATEGY,
        )
    result = {
        k: out[k].float().cpu().numpy()
        for k in ["depth", "depth_conf", "extrinsics", "intrinsics"]
    }
    # Squeeze the batch dim (B=1) → per-view arrays.
    result = {k: (v[0] if v.shape[0] == 1 else v) for k, v in result.items()}

    # Umeyama align to the input poses — matches DA3NestedONNX's default
    # align_input_ext_scale=True (raw extrinsics + letterbox-scaled intrinsics).
    # Skipped without camera pose so the predicted poses are compared directly,
    # exactly as DA3NestedONNX does when extrs is None.
    if use_extrinsics:
        result = _BASE.align_to_input(result, exts, intrs_adj, align_scale=align_scale)

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
    # The loaded graph is the single source of truth for whether poses are used;
    # drive the PyTorch reference the same way so the two sides stay comparable.
    use_ext = onnx.uses_extrinsics
    print(f"[COMPARE] any-view uses camera pose: {use_ext}")
    img_batch, intrs_adj, _ = onnx.preprocess_views(images, ixts)
    onx = onnx.infer(images, exts, ixts, normalize_extrinsics=True)
    del onnx
    _cuda_gc()

    exts_norm = BaseDA3Model.normalize_extrinsics(exts)
    api_model.to("cuda")
    pt = _pt_anyview(api_model, img_batch, intrs_adj, exts_norm, use_extrinsics=use_ext)
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
    align_scale: bool = True,
) -> None:
    """Full nested pipeline PyTorch vs ONNX (any-view + metric + alignment)."""
    onnx = DA3NestedONNX(onnx_anyview, onnx_metric, "cuda")
    _check_view_count(onnx.av.num_views, len(images), "nested any-view")
    # Match the PyTorch any-view branch to the loaded graph's pose usage.
    use_ext = onnx.av.uses_extrinsics
    print(f"[COMPARE] nested any-view uses camera pose: {use_ext}  align_scale={align_scale}")
    # Feed/align to input poses only in the with-camera-pose case; otherwise both
    # sides keep predicted poses (so the comparison actually exercises them).
    exts_in = exts if use_ext else None
    img_batch, intrs_adj, metas = onnx.av.preprocess_views(images, ixts)
    onx = onnx.infer(images, exts_in, ixts, align_input_ext_scale=True, align_scale=align_scale)
    del onnx
    _cuda_gc()

    api_model.to("cuda")
    pt = _pt_nested(
        api_model, img_batch, intrs_adj, exts_in, metas,
        use_extrinsics=use_ext, align_scale=align_scale,
    )
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
        "--use-extrinsics",
        action="store_true",
        help="Compare the '-with-camera-pose' any-view model (consumes camera "
        "extrinsics/intrinsics as priors). Selects that ONNX checkpoint by default; "
        "the actual pose usage is read from the loaded graph and applied to the "
        "PyTorch reference too.",
    )
    parser.add_argument(
        "--onnx-anyview",
        type=str,
        default=None,
        help="Any-view ONNX path (default: --use-extrinsics variant under weights/).",
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
        "and rescaling depth. Applied to both the ONNX and PyTorch sides.",
    )
    parser.set_defaults(align_scale=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA is required for this comparison.")
    onnx_anyview = args.onnx_anyview or DEFAULT_ANYVIEW_ONNX[args.use_extrinsics]
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

    # Load the PyTorch reference once; kept on CPU except during its own forward.
    print(f"[PYTORCH] Loading nested model from {args.pt_ckpt} ...")
    api_model = DepthAnything3.from_pretrained(args.pt_ckpt)
    api_model.eval()

    if "metric" in modules:
        compare_metric(args.onnx_metric, api_model, images[0], ixts[0])

    if "anyview" in modules:
        compare_anyview(onnx_anyview, api_model, images, exts, ixts)

    if "nested" in modules:
        compare_nested(
            onnx_anyview, args.onnx_metric, api_model, images, exts, ixts,
            align_scale=args.align_scale,
        )


if __name__ == "__main__":
    main()
