"""Nested Depth Anything 3 ONNX pipeline (any-view + metric + alignment).

Composes ``DA3AnyViewONNX`` and ``DA3MetricONNX`` and aligns their outputs to
reproduce ``NestedDepthAnything3Net`` — the ONNX sibling of the TRT
``DA3NestedModel``.  Output depth is left **unmasked** (PyTorch does not
confidence-mask depth; only sky regions are set to max depth, handled inside
``align_anyview_with_metric``).
"""

from __future__ import annotations

import cv2
import numpy as np

from model.base_da3 import BaseDA3Model
from model.da3anyview_onnx import DA3AnyViewONNX
from model.da3metric_onnx import DA3MetricONNX


class DA3NestedONNX(BaseDA3Model):
    """Any-view + metric ONNX pipeline replicating the nested PyTorch model."""

    def __init__(self, anyview_path: str, metric_path: str, device: str = "cuda") -> None:
        self.av = DA3AnyViewONNX(anyview_path, device)
        self.metric = DA3MetricONNX(metric_path, device)
        # BaseDA3Model methods (normalize_extrinsics, align_*) need target size.
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        print(
            f"[NESTED] anyview N={self.av.num_views} @ {self.av.target_h}x"
            f"{self.av.target_w}, metric @ {self.metric.target_h}x"
            f"{self.metric.target_w}"
        )

    def infer(
        self,
        imgs: list,
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        align_input_ext_scale: bool = True,
    ) -> dict:
        """Run the full nested pipeline; returns numpy depth/conf/extrinsics/intrinsics."""
        n = len(imgs)
        if self.av.num_views is not None and n != self.av.num_views:
            raise ValueError(
                f"Got {n} views but the any-view ONNX model expects "
                f"{self.av.num_views}. Pass exactly {self.av.num_views} views."
            )

        # 1. Any-view: preprocess + normalize + run + map
        img_batch, intrs_adj, _ = self.av.preprocess_views(imgs, intrs)
        extrs_norm = self.normalize_extrinsics(extrs)
        av = self.av.map_anyview_keys(
            self.av.run(
                {
                    "image": img_batch.astype(np.float32),
                    "extrinsics": extrs_norm[None].astype(np.float32),
                    "intrinsics": intrs_adj[None].astype(np.float32),
                }
            )
        )

        # 2. Metric branch (one view at a time)
        metric_depths, metric_skys = self._run_metric_branch(img_batch)

        # 3. Align any-view depth to metric (sky handling inside)
        result = self.align_with_metric(av, metric_depths, metric_skys)

        # 4. Optional Umeyama align to input poses (PyTorch inference default)
        if align_input_ext_scale:
            result = self.align_to_input(result, extrs, intrs_adj)

        return result

    def _run_metric_branch(self, img_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-view metric inference on the already-preprocessed any-view batch.

        ``img_batch`` is ``(1, N, 3, H_av, W_av)``.  Each view is resized to the
        metric model's H/W when it differs, run, then depth/sky are resized back
        to the any-view H/W.  Returns ``(1, N, H_av, W_av)`` depth and sky.
        """
        n = img_batch.shape[1]
        h, w = self.av.target_h, self.av.target_w
        mh, mw = self.metric.target_h, self.metric.target_w
        need_resize = (mh, mw) != (h, w)

        depths = np.zeros((1, n, h, w), dtype=np.float32)
        skys = np.zeros((1, n, h, w), dtype=np.float32)
        for i in range(n):
            view = img_batch[0, i]  # (3, H_av, W_av)
            if need_resize:
                view = cv2.resize(
                    view.transpose(1, 2, 0),
                    (mw, mh),
                    interpolation=cv2.INTER_LINEAR,
                ).transpose(2, 0, 1)
            d, s = self.metric.infer_view(view)
            if need_resize:
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
                s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
            depths[0, i] = d
            skys[0, i] = s
        return depths, skys
