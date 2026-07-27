import cv2
import numpy as np

from .data_structure import CameraIntrinsics
from .base_trt import MonoDepthTRT

class DA3MetricModel(MonoDepthTRT):
	"""Depth Anything v3 metric TensorRT wrapper.

	Preprocess and postprocess are aligned with tools/export_models/da3/export_onnx.py:
	- preprocess: resize to model input size, RGB conversion, ImageNet normalization
	- postprocess: extract depth/sky outputs and convert depth to metric scale using focal
	"""

	_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	def __init__(self, engine_path: str, camera_intrinsics: CameraIntrinsics, conf_thresh: float = 0.5):
		super().__init__(engine_path)
		self.camera_intrinsics = camera_intrinsics
		self.conf_thresh = conf_thresh

	def preprocess(self, img_bgr: np.ndarray):
		"""Preprocess image for DA3 model input."""
		if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
			raise ValueError("Expected a BGR image with shape (H, W, 3).")
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		img_padded, meta = self.resize_img(img_rgb)
		img_normalized = img_padded.astype(np.float32) / 255.0
		img_normalized = (img_normalized - self._MEAN) / self._STD
		return img_normalized, meta

	def parse_outputs(self, outputs: dict, meta_info: dict):
		"""Postprocess raw model outputs to depth/sky/metric depth."""
		if not outputs:
			raise RuntimeError("Model returned no outputs.")

		depth_name = None
		sky_name = None
		names = list(outputs.keys())

		for name in names:
			lname = name.lower()
			if depth_name is None and "depth" in lname:
				depth_name = name
			if sky_name is None and "sky" in lname:
				sky_name = name

		if depth_name is None:
			depth_name = names[0]
		if sky_name is None and len(names) > 1:
			sky_name = names[1]

		depth_full = outputs[depth_name].squeeze().astype(np.float32)
		sky_full = outputs[sky_name].squeeze().astype(np.float32) if sky_name is not None else None

		pad_top = int(meta_info["pad_top"])
		pad_left = int(meta_info["pad_left"])
		tile_h = int(meta_info["tile_h"])
		tile_w = int(meta_info["tile_w"])

		s = (slice(pad_top, pad_top + tile_h), slice(pad_left, pad_left + tile_w))
		depth = depth_full[s]
		sky = sky_full[s] if sky_full is not None else None

		# get scaled metric depth
		scale_factor = float(meta_info["scale_factor"])
		focal = self.camera_intrinsics.fxy_scaled(tile_w, tile_h)
		metric_depth = focal * depth / 300.0

		conf = 1 - sky if sky is not None else np.ones_like(metric_depth)
		valid = conf > self.conf_thresh
		metric_depth[~valid] = -1.0
		return metric_depth, conf, scale_factor

	def infer(self, img_bgr: np.ndarray):
		"""Run DA3 TRT inference and return metric depth plus auxiliary outputs."""
		input_img, meta = self.preprocess(img_bgr)
		outputs = self._infer(input_img)
		return self.parse_outputs(outputs, meta_info=meta)
