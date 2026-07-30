"""Any-view Depth Anything 3 ONNX wrapper.

Multi-view depth + confidence + predicted camera parameters.  Built on
``ONNXModel`` (session) + ``BaseDA3Model`` (pre/post).
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_onnx import ONNXModel


class DA3AnyViewONNX(ONNXModel, BaseDA3Model):
    """Any-view ONNX inference: images (+ extrinsics/intrinsics) → depth bundle."""

    def infer(
        self,
        imgs: list,
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        normalize_extrinsics: bool = False,
    ) -> dict:
        """Run the any-view model.

        ``imgs`` are *N* paths or BGR arrays; ``extrs`` ``(N,4,4)``; ``intrs``
        ``(N,3,3)`` at original resolution.  Returns the mapped output dict with
        batched values (``(1,N,…)``).  Extrinsics are normalized here only when
        ``normalize_extrinsics=True`` (the ONNX graph does not normalize).
        """
        img_batch, intrs_adj, _ = self.preprocess_views(imgs, intrs)
        ext = self.normalize_extrinsics(extrs) if normalize_extrinsics else extrs
        raw = self.run(
            {
                "image": img_batch.astype(np.float32),
                "extrinsics": ext[None].astype(np.float32),
                "intrinsics": intrs_adj[None].astype(np.float32),
            }
        )
        return self.map_anyview_keys(raw)
