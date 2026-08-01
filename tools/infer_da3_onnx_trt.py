#!/usr/bin/env python3
"""Run DepthAnything3 inference on the Astribot dataset via ONNX or TensorRT.

Select the backend (``--backend onnx|trt``) and the module to run
(``--module metric|anyview|nested``); the matching wrapper class is built from
the shared ``model/`` wrappers.  Results are saved as a ``result.npz`` per frame.

- ``metric``  — single-image depth + sky, run per view (mono-sky post-processing
  applied), cropped to each view's tile.
- ``anyview`` — multi-view depth / confidence / predicted cameras, cropped.
- ``nested``  — any-view + metric + alignment (the full pipeline), cropped.

Usage:
    # Nested pipeline, ONNX (default)
    python tools/infer_da3_onnx_trt.py --module nested --camera-set set1 --frame 0

    # Any-view branch, TensorRT
    python tools/infer_da3_onnx_trt.py --backend trt --module anyview --frame 0

    # Any-view with camera-pose priors (selects the -with-camera-pose model)
    python tools/infer_da3_onnx_trt.py --module anyview --use-extrinsics --frame 0

    # Metric model, all frames
    python tools/infer_da3_onnx_trt.py --module metric --all-frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure tools/ is on sys.path for astribot_dataloader and model.* imports
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))

from tools.utils.astribot_dataloader import (  # noqa: E402
    CAMERA_SETS,
    camera_set_for_views,
    count_frames,
    load_images_cam_params,
)
from model.base_da3 import BaseDA3Model  # noqa: E402
from model.da3anyview import DA3AnyViewONNX, DA3AnyViewTRT  # noqa: E402
from model.da3metric import DA3MetricONNX, DA3MetricTRT  # noqa: E402
from model.da3nested import DA3NestedONNX, DA3NestedTRT  # noqa: E402

# Backend-appropriate default metric model paths.
DEFAULT_METRIC = {
    "onnx": "weights/da3_metric_644x490_giant-large-1.1.onnx",
    "trt": "weights/da3_metric_644x490_giant-large-1.1.engine",
}

# Any-view defaults come in two variants: the plain export (model predicts its own
# poses) and the "-with-camera-pose" export selected by --use-extrinsics, which
# additionally consumes camera extrinsics/intrinsics as priors.
_ANYVIEW_STEM = "weights/da3_anyview_n3_644x490_giant-large-1.1"
_BACKEND_EXT = {"onnx": ".onnx", "trt": ".engine"}


def default_anyview_path(backend: str, use_extrinsics: bool) -> str:
    """Backend + variant → default any-view model path."""
    suffix = "-with-camera-pose" if use_extrinsics else ""
    return f"{_ANYVIEW_STEM}{suffix}{_BACKEND_EXT[backend]}"


def build_model(backend: str, module: str, anyview_path: str, metric_path: str, device: str):
    """Construct the wrapper for the selected backend + module."""
    if backend == "onnx":
        if module == "metric":
            return DA3MetricONNX(metric_path, device)
        if module == "anyview":
            return DA3AnyViewONNX(anyview_path, device)
        return DA3NestedONNX(anyview_path, metric_path, device)
    # TensorRT (always CUDA).
    if module == "metric":
        return DA3MetricTRT(metric_path)
    if module == "anyview":
        return DA3AnyViewTRT(anyview_path)
    return DA3NestedTRT(anyview_path, metric_path)


def _expected_views(model, module: str) -> int | None:
    """Fixed view count the model was exported for (``None`` for metric)."""
    if module == "anyview":
        return getattr(model, "num_views", None)
    if module == "nested":
        return getattr(model.av, "num_views", None)
    return None


def _crop_bundle(bundle: dict, metas: list) -> dict:
    """Crop a padded any-view output bundle (batched ``(1, N, …)``) to the tiles.

    Mirrors ``DA3Nested*._crop_result``: per-view crop of depth/conf and un-pad of
    the principal point, with the batch dim squeezed off.
    """
    depth = bundle["depth"][0]  # (N, H, W)
    conf = bundle["depth_conf"][0]
    depth = np.stack([BaseDA3Model.crop_to_tile(depth[i], metas[i]) for i in range(len(metas))])
    conf = np.stack([BaseDA3Model.crop_to_tile(conf[i], metas[i]) for i in range(len(metas))])
    intr = bundle["intrinsics"][0].copy()  # (N, 3, 3)
    for i, m in enumerate(metas):
        intr[i, 0, 2] -= m["pad_left"]
        intr[i, 1, 2] -= m["pad_top"]
    return {
        "depth": depth,
        "depth_conf": conf,
        "extrinsics": bundle["extrinsics"][0],
        "intrinsics": intr,
    }


def _processed_images(img_batch: np.ndarray, metas: list) -> np.ndarray:
    """De-normalize the letterboxed CHW batch, crop to each tile → ``(N, h, w, 3)`` uint8 RGB.

    These are the images the model actually saw (matching the depth resolution), used
    both for the side-by-side depth visualization and as point-cloud colours.
    """
    imgs = img_batch[0]  # (N, 3, H, W)
    out = []
    for i, m in enumerate(metas):
        chw = BaseDA3Model.crop_to_tile(imgs[i], m)  # (3, th, tw)
        hwc = chw.transpose(1, 2, 0) * BaseDA3Model._STD + BaseDA3Model._MEAN
        out.append((hwc * 255.0).clip(0, 255).astype(np.uint8))
    return np.stack(out)


def run_module(
    model,
    module: str,
    images: list,
    exts: np.ndarray | None,
    ixts: np.ndarray,
    align_input_ext_scale: bool,
    align_scale: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run the selected module on one frame.

    ``exts`` is ``None`` when running without camera pose; the model then predicts
    its own poses and the nested pipeline skips input-pose alignment.

    ``align_scale`` (nested + input poses only) selects whether the output uses the
    input poses with rescaled depth (``True``) or keeps the predicted poses aligned
    into the input frame with unscaled depth (``False``).

    Returns ``(result, images_u8)`` — cropped numpy outputs plus the cropped,
    de-normalized RGB views (for visualization).
    """
    img_batch, _, metas = model.preprocess_views(images, ixts)
    images_u8 = _processed_images(img_batch, metas)

    if module == "nested":
        result = model.infer(
            images, exts, ixts,
            align_input_ext_scale=align_input_ext_scale,
            align_scale=align_scale,
        )
    elif module == "anyview":
        bundle = model.infer(images, exts, ixts)  # padded (1, N, …)
        result = _crop_bundle(bundle, metas)
    else:
        # metric: per-view monocular depth + sky (mono-sky on), cropped to each tile.
        depths, skys = [], []
        for i in range(len(images)):
            d, s = model.infer_view(img_batch[0, i])  # letterbox grid; mono-sky applied
            depths.append(model.crop_to_tile(d, metas[i]))
            skys.append(model.crop_to_tile(s, metas[i]))
        result = {"depth": np.stack(depths), "sky": np.stack(skys)}

    return result, images_u8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DepthAnything3 ONNX / TensorRT inference on the Astribot dataset",
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "trt"],
        default="onnx",
        help="Inference backend: ONNX Runtime or TensorRT.",
    )
    parser.add_argument(
        "--module",
        choices=["metric", "anyview", "nested"],
        default="nested",
        help="Which model to run.",
    )
    parser.add_argument(
        "--camera-set",
        choices=["set0", "set1", "set2"],
        default=None,
        help="set0 = (stereo), set1 = 3 views (head_rgbd + stereo L/R), set2 = 4 views. "
        "Default: auto-selected to match the loaded model's view count (metric → set1).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Single frame index (0-based).  Use --all-frames to process all.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process all frames common to the selected cameras.",
    )
    parser.add_argument(
        "--use-extrinsics",
        action="store_true",
        help="Any-view/nested only: use the '-with-camera-pose' model that consumes "
        "camera extrinsics/intrinsics as priors. Selects that checkpoint by default "
        "and feeds the poses; omit to use the plain model (poses are predicted). "
        "Ignored if the loaded model's inputs say otherwise.",
    )
    parser.add_argument(
        "--anyview-model",
        type=str,
        default=None,
        help="Any-view ONNX/engine path (default: backend + --use-extrinsics variant "
        "under weights/).",
    )
    parser.add_argument(
        "--metric-model",
        type=str,
        default=None,
        help="Metric ONNX/engine path (default: backend-appropriate weights/).",
    )
    parser.add_argument(
        "--export-dir",
        default="output_infer",
        help="Directory to save results (NPZ per frame).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX Runtime device (TensorRT always uses CUDA).",
    )
    parser.add_argument(
        "--no-align-input-ext-scale",
        dest="align_input_ext_scale",
        action="store_false",
        help="Disable Umeyama alignment to the input camera poses (nested only; on "
        "by default, matching DepthAnything3.inference).",
    )
    parser.set_defaults(align_input_ext_scale=True)
    parser.add_argument(
        "--keep-predicted-pose",
        dest="align_scale",
        action="store_false",
        help="Nested + --use-extrinsics only: instead of replacing the output poses "
        "with the input poses (and rescaling depth to the input-pose scale), keep "
        "the model's PREDICTED poses rigidly aligned into the input frame (Umeyama "
        "rotation+translation) and leave depth at the predicted metric scale "
        "(align_scale=False). No effect without input poses or with "
        "--no-align-input-ext-scale.",
    )
    parser.set_defaults(align_scale=True)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save colour-coded depth maps (alongside the images) and, for "
        "anyview/nested, a point-cloud scene.glb.",
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Draw camera frustums in the exported GLB (with --visualize).",
    )
    args = parser.parse_args()

    anyview_path = args.anyview_model or default_anyview_path(args.backend, args.use_extrinsics)
    metric_path = args.metric_model or DEFAULT_METRIC[args.backend]

    # --- Build the selected model (first, so the camera set can match its views) ---
    print(f"[MODEL] backend={args.backend} module={args.module}")
    model = build_model(args.backend, args.module, anyview_path, metric_path, args.device)

    # --- Resolve camera set ---
    # Default: auto-select the set whose view count matches the model (metric is
    # single-view, so it falls back to set1). An explicit --camera-set is honoured.
    n_expected = _expected_views(model, args.module)
    if args.camera_set is not None:
        camera_set_name = args.camera_set
    elif n_expected is not None:
        camera_set_name = camera_set_for_views(n_expected)
        print(f"[DATA] Auto-selected --camera-set {camera_set_name} for {n_expected} views.")
    else:
        camera_set_name = "set1"
    camera_names = CAMERA_SETS[camera_set_name]

    # --- Determine frame range ---
    if args.all_frames:
        frame_counts = {key: count_frames(key) for key in camera_names}
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

    # --- Per-frame inference ---
    for frame_idx in frame_indices:
        print(f"\n--- Frame {frame_idx} ---")
        images, exts, ixts = load_images_cam_params(camera_set_name, frame_idx)
        print(f"  Views: {len(images)}")
        for p in images:
            print(f"    {p}")

        if n_expected is not None and len(images) != n_expected:
            raise SystemExit(
                f"[ERROR] Camera set '{camera_set_name}' provides {len(images)} views, "
                f"but the {args.module} model was exported for {n_expected}. Pick a "
                f"--camera-set with {n_expected} views (set0=2, set1=3, set2=4)."
            )

        # Only feed camera extrinsics when --use-extrinsics is set. Without them
        # the model predicts its own poses and the nested pipeline skips the
        # input-pose alignment, so the output keeps the predicted poses — matching
        # DepthAnything3.inference(extrinsics=None) used by infer_pytorch.py.
        exts_in = exts if args.use_extrinsics else None
        result, images_u8 = run_module(
            model, args.module, images, exts_in, ixts,
            args.align_input_ext_scale, args.align_scale,
        )

        for key, val in result.items():
            print(f"  {key:12s} shape: {tuple(val.shape)}")

        # --- Save ---
        out_dir = Path(args.export_dir)
        if len(frame_indices) > 1:
            out_dir = out_dir / f"frame_{frame_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_dir / "result.npz",
            **{k: v.astype(np.float32) for k, v in result.items()},
        )
        print(f"  Saved: {out_dir / 'result.npz'}")

        # --- Visualization ---
        if args.visualize:
            from utils.visualization import export_glb, save_depth_vis  # noqa: PLC0415

            vis_dir = save_depth_vis(images_u8, result["depth"], out_dir)
            print(f"  Saved depth vis: {vis_dir}")
            # Point cloud needs camera params — anyview/nested only (not metric).
            if "intrinsics" in result and "extrinsics" in result:
                glb_path = export_glb(
                    result["depth"],
                    result["intrinsics"],
                    result["extrinsics"],
                    images_u8,
                    conf=result.get("depth_conf"),
                    out_path=out_dir / "scene.glb",
                    show_cameras=args.show_cameras,
                )
                print(f"  Saved GLB: {glb_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
