"""Depth Anything v3 any-view TensorRT wrapper.

Multi-view depth + confidence + predicted camera parameters.  Built on
``TRTModel`` (engine) + ``BaseDA3Model`` (pre/post) — the TRT sibling of
``DA3AnyViewONNX``.
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_trt import TRTModel


class DA3AnyViewModel(TRTModel, BaseDA3Model):
    """Any-view TRT inference: images (+ extrinsics/intrinsics) → depth bundle."""

    def __init__(self, engine_path: str, conf_thresh: float = 0.5) -> None:
        super().__init__(engine_path)
        self.conf_thresh = conf_thresh
        print(f"Any-view engine: {self.target_h}x{self.target_w}, " f"num_views={self.num_views}")

    def infer(
        self,
        imgs: list,
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        normalize_extrinsics: bool = False,
    ) -> dict:
        """Run the any-view engine.

        ``imgs`` are *N* paths or BGR arrays; ``extrs`` ``(N,4,4)``; ``intrs``
        ``(N,3,3)`` at original resolution.  Returns the mapped output dict with
        batched values (``(1,N,…)``).  Extrinsics are normalized here only when
        ``normalize_extrinsics=True`` (the graph does not normalize).
        """
        img_batch, intrs_adj, _ = self.preprocess_views(imgs, intrs)
        ext = self.normalize_extrinsics(extrs) if normalize_extrinsics else extrs
        raw = self._run(
            {
                "image": img_batch.astype(np.float32),
                "extrinsics": ext[None].astype(np.float32),
                "intrinsics": intrs_adj[None].astype(np.float32),
            }
        )
        return self.map_anyview_keys(raw)
