"""Alignment utilities for combining any-view and metric depth predictions.

Mirrors the post-processing logic of ``NestedDepthAnything3Net`` so that
independently-exported any-view and metric models can be combined outside
the ONNX / TRT graph.
"""

from __future__ import annotations

import numpy as np
import torch


def align_anyview_with_metric(
    anyview_depth: torch.Tensor,
    anyview_conf: torch.Tensor,
    anyview_extrinsics: torch.Tensor,
    anyview_intrinsics: torch.Tensor,
    metric_depth: torch.Tensor,
    metric_sky: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Replicate ``NestedDepthAnything3Net`` alignment logic in standalone Python.

    Mirrors the three post-processing steps so they run **outside** the ONNX
    or TRT graph:

    1. Metric scaling via predicted intrinsics
    2. Least-squares depth alignment
    3. Sky-region handling

    Parameters
    ----------
    anyview_depth : ``(B, N, H, W)`` or ``(N, H, W)``
        Raw any-view depth prediction.
    anyview_conf : ``(B, N, H, W)`` or ``(N, H, W)``
        Any-view depth confidence.
    anyview_extrinsics : ``(B, N, 3, 4)`` or ``(N, 3, 4)``
        Any-view predicted camera extrinsics.
    anyview_intrinsics : ``(B, N, 3, 3)`` or ``(N, 3, 3)``
        Any-view predicted camera intrinsics.
    metric_depth : ``(B, N, H, W)`` or ``(N, H, W)``
        Raw metric-model depth (monocular, unscaled).
    metric_sky : ``(B, N, H, W)`` or ``(N, H, W)``
        Metric-model sky logits (0 = non-sky, 1 = sky).

    Returns
    -------
    dict[str, torch.Tensor]
        ``depth``, ``depth_conf``, ``extrinsics``, ``intrinsics``.
    """
    from depth_anything_3.utils.alignment import (  # noqa: PLC0415
        apply_metric_scaling,
        compute_alignment_mask,
        compute_sky_mask,
        least_squares_scale_scalar,
        sample_tensor_for_quantile,
        set_sky_regions_to_max_depth,
    )

    # ---- step 1: metric scaling --------------------------------------------
    metric_depth = apply_metric_scaling(metric_depth, anyview_intrinsics)

    # ---- step 2: least-squares scale alignment -----------------------------
    non_sky_mask = compute_sky_mask(metric_sky, threshold=0.3)
    if non_sky_mask.sum() <= 10:
        raise RuntimeError("Insufficient non-sky pixels for alignment")

    depth_conf_ns = anyview_conf[non_sky_mask]
    depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100_000)
    median_conf = torch.quantile(depth_conf_sampled, 0.5)

    align_mask = compute_alignment_mask(
        anyview_conf, non_sky_mask, anyview_depth, metric_depth, median_conf,
    )

    scale_factor = least_squares_scale_scalar(
        metric_depth[align_mask], anyview_depth[align_mask],
    )

    anyview_depth = anyview_depth * scale_factor
    anyview_extrinsics = anyview_extrinsics.clone()
    anyview_extrinsics[..., :3, 3] *= scale_factor

    # ---- step 3: sky-region handling ---------------------------------------
    non_sky_depth = anyview_depth[non_sky_mask]
    if non_sky_depth.numel() > 100_000:
        idx = torch.randint(
            0, non_sky_depth.numel(), (100_000,), device=non_sky_depth.device,
        )
        sampled_depth = non_sky_depth[idx]
    else:
        sampled_depth = non_sky_depth
    non_sky_max = min(float(torch.quantile(sampled_depth, 0.99)), 200.0)

    anyview_depth, anyview_conf = set_sky_regions_to_max_depth(
        anyview_depth, anyview_conf, non_sky_mask, max_depth=non_sky_max,
    )

    return {
        "depth": anyview_depth,
        "depth_conf": anyview_conf,
        "extrinsics": anyview_extrinsics,
        "intrinsics": anyview_intrinsics,
    }


def align_to_input_ext_scale(
    pred_depth: np.ndarray,
    pred_extrinsics: np.ndarray,
    input_extrinsics: np.ndarray,
    input_intrinsics: np.ndarray,
    align_scale: bool = True,
    ransac_view_thresh: int = 10,
) -> dict[str, np.ndarray]:
    """Align a prediction to the input camera poses (numpy post-processing).

    Standalone replica of ``DepthAnything3._align_to_input_extrinsics_intrinsics``
    (``api.py``) so it can run **outside** the ONNX / TRT graph, on the outputs of
    :func:`align_anyview_with_metric`.  The Umeyama Sim(3) scale it needs (3x3 SVD,
    optional RANSAC via ``evo``) is not ONNX-exportable, hence the Python helper.

    Parameters
    ----------
    pred_depth : ``(N, H, W)``
        Predicted (metric-scaled) depth from the nested pipeline.
    pred_extrinsics : ``(N, 3, 4)`` or ``(N, 4, 4)``
        Predicted camera extrinsics (world-to-camera), in the model's frame.
    input_extrinsics : ``(N, 4, 4)``
        Original **un-normalised** input extrinsics (world-to-camera).
    input_intrinsics : ``(N, 3, 3)``
        Input intrinsics, scaled to the processing resolution.  Passed straight
        through to the output (mirrors the PyTorch behaviour).
    align_scale : bool, default ``True``
        If ``True``: output extrinsics are the input extrinsics and ``depth`` is
        divided by the Umeyama scale.  If ``False``: output extrinsics are the
        predicted poses aligned into the input frame, and depth is unchanged.
    ransac_view_thresh : int, default 10
        Use RANSAC alignment when the number of views is ``>=`` this threshold.

    Returns
    -------
    dict[str, np.ndarray]
        ``depth``, ``extrinsics`` ``(N, 3, 4)``, ``intrinsics``.
    """
    from depth_anything_3.utils.pose_align import align_poses_umeyama  # noqa: PLC0415

    pred_extrinsics = np.asarray(pred_extrinsics, dtype=np.float64)
    input_extrinsics = np.asarray(input_extrinsics, dtype=np.float64)

    _, _, scale, aligned_extrinsics = align_poses_umeyama(
        pred_extrinsics,
        input_extrinsics,
        ransac=len(input_extrinsics) >= ransac_view_thresh,
        return_aligned=True,
        random_state=42,
    )

    out_depth = np.asarray(pred_depth).copy()
    if align_scale:
        out_extrinsics = input_extrinsics[..., :3, :].copy()
        out_depth = out_depth / scale
    else:
        out_extrinsics = aligned_extrinsics[..., :3, :]

    return {
        "depth": out_depth.astype(np.float32),
        "extrinsics": out_extrinsics.astype(np.float32),
        "intrinsics": np.asarray(input_intrinsics, dtype=np.float32).copy(),
    }
