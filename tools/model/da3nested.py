"""Nested Depth Anything v3 TensorRT wrapper.

Combines an any-view TRT engine with a metric TRT engine and aligns their
outputs to reproduce the behaviour of ``NestedDepthAnything3Net``.

Usage::

    from tools.model.da3nested import DA3NestedModel
    from tools.model.data_structure import CameraIntrinsics

    nested = DA3NestedModel(
        anyview_engine="weights/da3_anyview.trt",
        metric_engine="weights/da3_metric.trt",
        metric_intrinsics=CameraIntrinsics.from_intrinsics_matrix(K),
    )
    result = nested.infer(
        imgs=["cam0.jpg", "cam1.jpg", "cam2.jpg"],
        extrs=extrinsics_array,   # (N, 4, 4)
        intrs=intrinsics_array,   # (N, 3, 3)
    )
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .alignment import align_anyview_with_metric
from .da3anyview import DA3AnyViewModel
from .da3metric import DA3MetricModel
from .data_structure import CameraIntrinsics

# ---------------------------------------------------------------------------
# ImageNet normalisation (must match both TRT engine expectations)
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DA3NestedModel:
    """Combined any-view + metric TRT pipeline replicating ``NestedDepthAnything3Net``.

    Parameters
    ----------
    anyview_engine : str
        Path to the any-view TRT engine (exported via ``--wrapper anyview``).
    metric_engine : str
        Path to the metric TRT engine (exported via ``--wrapper metric``).
    metric_intrinsics : CameraIntrinsics
        Intrinsics used by the metric model for focal-based scaling.
        Only ``fx`` / ``fy`` are relevant; passed through the
        ``CameraIntrinsics`` helper for consistency with ``DA3MetricModel``.
    conf_thresh : float
        Confidence threshold for masking low-confidence depth (default 0.5).
    """

    def __init__(
        self,
        anyview_engine: str,
        metric_engine: str,
        metric_intrinsics: CameraIntrinsics,
        conf_thresh: float = 0.5,
    ) -> None:
        # Any-view branch (multi-view, exact-resize preprocessing)
        self.av = DA3AnyViewModel(anyview_engine, conf_thresh=conf_thresh)

        # Metric branch (single-image, but we bypass its built-in preprocessing)
        self.metric = DA3MetricModel(
            metric_engine, metric_intrinsics, conf_thresh=conf_thresh,
        )

        # Convenience
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        self.num_views = self.av.num_views
        self.conf_thresh = conf_thresh

        # Metric-engine target size (may differ from any-view)
        m_shape = self.metric.inputs[0]["shape"]  # (1, 3, H, W) or symbolic
        self._metric_h = (
            m_shape[2] if isinstance(m_shape[2], int) else self.target_h
        )
        self._metric_w = (
            m_shape[3] if isinstance(m_shape[3], int) else self.target_w
        )
        print(
            f"Nested engine: anyview={self.target_h}x{self.target_w} (N={self.num_views}), "
            f"metric={self._metric_h}x{self._metric_w}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(
        self,
        imgs: list[np.ndarray | str],
        extrs: np.ndarray,
        intrs: np.ndarray,
    ) -> dict:
        """Run the full nested pipeline.

        Parameters
        ----------
        imgs :
            *N* BGR images (``uint8 (H,W,3)`` arrays) or absolute file paths.
        extrs :
            ``(N, 4, 4)`` camera extrinsics (world-to-camera convention).
        intrs :
            ``(N, 3, 3)`` camera intrinsics at the **original** image
            resolution (re-scaled internally).

        Returns
        -------
        dict with keys:
            ``depth``       — aligned metric depth ``(N, H, W)``
            ``depth_conf``  — confidence ``(N, H, W)``
            ``extrinsics``  — predicted extrinsics ``(N, 3, 4)``
            ``intrinsics``  — predicted intrinsics ``(N, 3, 3)``
            ``scale_factor`` — least-squares scale factor (float)
        """
        # ---- 1. Any-view preprocessing + inference -------------------------
        img_batch, intrs_adj, _metas = self.av.preprocess(imgs, intrs)   # (1,N,3,H,W)
        extrs_norm = self.av._normalize_extrinsics(extrs)                 # (N,4,4)

        av_raw = self.av._infer(img_batch, extrs_norm, intrs_adj)        # raw dict

        # Normalise output keys
        av = self._map_anyview_keys(av_raw)

        # ---- 2. Metric inference (one view at a time) ---------------------
        N = len(imgs)
        metric_depths = np.zeros((N, self.target_h, self.target_w), dtype=np.float32)
        metric_skys = np.zeros((N, self.target_h, self.target_w), dtype=np.float32)

        for i, img in enumerate(imgs):
            bgr = cv2.imread(img, cv2.IMREAD_COLOR) if isinstance(img, str) else img.copy()

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_m = cv2.resize(
                rgb, (self._metric_w, self._metric_h), interpolation=cv2.INTER_LINEAR,
            )
            metric_inp = (rgb_m.astype(np.float32) / 255.0 - _MEAN) / _STD

            m_raw = self.metric._infer(metric_inp)
            m_depth, m_sky = self._extract_metric(m_raw)

            if m_depth.shape[-2:] != (self.target_h, self.target_w):
                m_depth = cv2.resize(
                    m_depth, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR,
                )
                m_sky = cv2.resize(
                    m_sky, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR,
                )

            metric_depths[i] = m_depth
            metric_skys[i] = m_sky

        # ---- 3. Alignment via shared `align_anyview_with_metric` ----------
        # Convert numpy → torch, align, convert back
        result = align_anyview_with_metric(
            anyview_depth=torch.from_numpy(av["depth"]),
            anyview_conf=torch.from_numpy(av["depth_conf"]),
            anyview_extrinsics=torch.from_numpy(av["extrinsics"]),
            anyview_intrinsics=torch.from_numpy(av["intrinsics"]),
            metric_depth=torch.from_numpy(metric_depths[None]),   # add B=1
            metric_sky=torch.from_numpy(metric_skys[None]),
        )

        # Convert back to numpy, squeeze batch dim, apply confidence masking
        out: dict = {}
        for k in ("depth", "depth_conf", "extrinsics", "intrinsics"):
            val = result[k]
            if val.ndim >= 4 and val.shape[0] == 1:
                val = val.squeeze(0)
            out[k] = val.float().cpu().numpy()

        # Mask low-confidence depth
        valid = out["depth_conf"] > self.conf_thresh
        out["depth_raw"] = out["depth"].copy()
        out["depth"] = out["depth"].copy()
        out["depth"][~valid] = -1.0

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_anyview_keys(raw: dict) -> dict:
        """Normalise any-view engine output keys to canonical names."""
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

    @staticmethod
    def _extract_metric(raw: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract (depth, sky) from raw metric engine outputs."""
        depth = sky = None
        for name, val in raw.items():
            low = name.lower()
            if "sky" in low:
                sky = val.squeeze().astype(np.float32)
            elif "depth" in low:
                depth = val.squeeze().astype(np.float32)
            elif depth is None:
                depth = val.squeeze().astype(np.float32)
        if sky is None:
            sky = np.zeros_like(depth)
        return depth, sky
