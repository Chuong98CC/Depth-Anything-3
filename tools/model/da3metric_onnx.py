"""Metric Depth Anything 3 ONNX wrapper (single-image, raw depth + sky)."""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_onnx import ONNXModel


class DA3MetricONNX(ONNXModel, BaseDA3Model):
    """Metric ONNX inference on a single already-preprocessed view."""

    def infer_view(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``img`` is a preprocessed ``(3, H, W)`` CHW array → ``(depth, sky)``."""
        raw = self.run({"image": img[None].astype(np.float32)})
        return self.extract_metric(raw)
