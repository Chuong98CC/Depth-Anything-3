"""Metric Depth Anything v3 wrappers (single-image, raw depth + sky).

``DA3MetricTRT`` runs a TensorRT engine and ``DA3MetricONNX`` runs an ONNX
session; both share ``BaseDA3Model`` post-processing.  Metric depth in metres is
a caller-side ``focal * depth / 300`` step (the model returns raw network depth
+ sky).
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_onnx import ONNXModel
from model.base_trt import TRTModel


class DA3MetricTRT(TRTModel, BaseDA3Model):
    """Metric TRT inference on a single already-preprocessed view."""

    def infer_view(
        self, img: np.ndarray, apply_mono_sky: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """``img`` is a preprocessed ``(3, H, W)`` CHW array → ``(depth, sky)``.

        ``apply_mono_sky`` clamps sky-region depth to match the full PyTorch metric
        forward; the nested pipeline passes ``False`` to feed the raw depth to
        alignment.
        """
        raw = self._run({self.inputs[0]["name"]: img[None].astype(np.float32)})
        depth, sky = self.extract_metric(raw)
        if apply_mono_sky:
            depth = self.apply_mono_sky(depth, sky)
        return depth, sky


class DA3MetricONNX(ONNXModel, BaseDA3Model):
    """Metric ONNX inference on a single already-preprocessed view."""

    def infer_view(
        self, img: np.ndarray, apply_mono_sky: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """``img`` is a preprocessed ``(3, H, W)`` CHW array → ``(depth, sky)``.

        ``apply_mono_sky`` clamps sky-region depth to match the full PyTorch metric
        forward; the nested pipeline passes ``False`` to feed the raw depth to
        alignment.
        """
        raw = self.run({"image": img[None].astype(np.float32)})
        depth, sky = self.extract_metric(raw)
        if apply_mono_sky:
            depth = self.apply_mono_sky(depth, sky)
        return depth, sky
