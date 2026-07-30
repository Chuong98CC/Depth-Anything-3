"""Nested Depth Anything v3 TensorRT pipeline (any-view + metric + alignment).

Composes ``DA3AnyViewModel`` and ``DA3MetricModel`` and aligns their outputs to
reproduce ``NestedDepthAnything3Net`` — the TRT sibling of ``DA3NestedONNX``.
Output depth is left **unmasked** (only sky regions are set to max depth inside
``align_anyview_with_metric``).
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.da3anyview import DA3AnyViewModel
from model.da3metric import DA3MetricModel


class DA3NestedModel(BaseDA3Model):
    """Any-view + metric TRT pipeline replicating the nested PyTorch model."""

    def __init__(
        self,
        anyview_engine: str,
        metric_engine: str,
        conf_thresh: float = 0.5,
    ) -> None:
        self.av = DA3AnyViewModel(anyview_engine, conf_thresh=conf_thresh)
        self.metric = DA3MetricModel(metric_engine)
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        print(
            f"[NESTED-TRT] anyview N={self.av.num_views} @ "
            f"{self.av.target_h}x{self.av.target_w}, "
            f"metric @ {self.metric.target_h}x{self.metric.target_w}"
        )

    def infer(
        self,
        imgs: list,
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        align_input_ext_scale: bool = True,
    ) -> dict:
        """Run the full nested pipeline; returns cropped numpy outputs."""
        n = len(imgs)
        if self.av.num_views is not None and n != self.av.num_views:
            raise ValueError(
                f"Got {n} views but the any-view engine expects "
                f"{self.av.num_views}. Pass exactly {self.av.num_views} views."
            )

        # 1. Any-view: letterbox preprocess + normalize + run + map
        img_batch, intrs_adj, metas = self.av.preprocess_views(imgs, intrs)
        extrs_norm = self.normalize_extrinsics(extrs)
        av = self.av.map_anyview_keys(
            self.av._run(
                {
                    "image": img_batch.astype(np.float32),
                    "extrinsics": extrs_norm[None].astype(np.float32),
                    "intrinsics": intrs_adj[None].astype(np.float32),
                }
            )
        )

        # 2. Metric branch (letterbox grid; reuse the any-view padded views)
        metric_depths, metric_skys = self._run_metric_branch(img_batch)

        # 3. Align any-view depth to metric (padded grid; sky handling inside)
        result = self.align_with_metric(av, metric_depths, metric_skys)

        # 4. Optional Umeyama align to input poses (PyTorch inference default)
        if align_input_ext_scale:
            result = self.align_to_input(result, extrs, intrs_adj)

        # 5. Crop padded outputs to the tile; un-pad intrinsics
        return self._crop_result(result, metas)

    def _run_metric_branch(self, av_img_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-view metric inference on the letterbox grid.

        Requires the metric engine to share the any-view target size (fail-fast
        otherwise); reuses the already-letterboxed any-view padded views.
        Returns ``(1, N, H, W)``.
        """
        h, w = self.av.target_h, self.av.target_w
        mh, mw = self.metric.target_h, self.metric.target_w
        if (mh, mw) != (h, w):
            raise NotImplementedError(
                "Nested pipeline requires the metric and any-view engines to "
                f"share the input size; got metric {mh}x{mw} vs any-view {h}x{w}. "
                "Re-export the metric engine at the any-view resolution."
            )
        n = av_img_batch.shape[1]
        depths = np.zeros((1, n, h, w), dtype=np.float32)
        skys = np.zeros((1, n, h, w), dtype=np.float32)
        for i in range(n):
            d, s = self.metric.infer_view(av_img_batch[0, i])
            depths[0, i] = d
            skys[0, i] = s
        return depths, skys

    def _crop_result(self, result: dict, metas: list) -> dict:
        """Crop padded depth/conf to the tile region and un-pad the intrinsics."""
        depth = np.stack(
            [self.av.crop_to_tile(result["depth"][i], metas[i]) for i in range(len(metas))]
        )
        conf = np.stack(
            [self.av.crop_to_tile(result["depth_conf"][i], metas[i]) for i in range(len(metas))]
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
