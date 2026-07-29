"""Depth Anything v3 any-view TensorRT wrapper.

Takes multi-view images with extrinsics and intrinsics, returns depth,
confidence, and predicted camera parameters.  Designed for TRT engines
built from the ONNX model exported by ``tools/export_onnx.py --wrapper anyview``.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorrt as trt
import torch

from .base_trt import TRTModel, trt_to_torch_dtype


class DA3AnyViewModel(TRTModel):
    """Depth Anything v3 any-view TensorRT wrapper.

    Preprocess and postprocess are aligned with the any-view ONNX export path
    and the ``DepthAnything3AnyViewOnnxWrapper`` in ``tools/export_onnx.py``.

    Parameters
    ----------
    engine_path : str
        Path to the TensorRT engine file.
    conf_thresh : float
        Confidence threshold for masking low-confidence depth values (default 0.5).
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, engine_path: str, conf_thresh: float = 0.5):
        super().__init__(engine_path)
        self.conf_thresh = conf_thresh

        # The image input is 5-D: (B, N, 3, H, W).  Override the 4-D defaults
        # set by TRTModel which assume NCHW layout.
        img_shape = self.inputs[0]["shape"]
        # img_shape = (B, N, C, H_target, W_target)  or symbolic
        self.num_views = img_shape[1] if isinstance(img_shape[1], int) else None
        self.target_h = img_shape[3] if isinstance(img_shape[3], int) else None
        self.target_w = img_shape[4] if isinstance(img_shape[4], int) else None
        if self.target_h is None or self.target_w is None:
            raise ValueError(
                "Engine image input shape must have concrete H, W dimensions. "
                f"Got: {img_shape}"
            )
        print(
            f"Any-view engine: {self.target_h}x{self.target_w}, "
            f"num_views={self.num_views}"
        )

    # ---- public API ---------------------------------------------------------

    def infer(
        self,
        imgs: list[np.ndarray | str],
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        normalize_extrinsics: bool = False,
    ) -> dict:
        """Run full any-view TRT inference pipeline.

        Parameters
        ----------
        imgs :
            List of *N* BGR images (``uint8`` arrays of shape ``(H, W, 3)``)
            or absolute file paths.
        extrs :
            Camera extrinsics, shape ``(N, 4, 4)`` (world-to-camera).
        intrs :
            Camera intrinsics, shape ``(N, 3, 3)`` (original resolution).
        normalize_extrinsics :
            If True, normalise extrinsics so the first camera sits at the
            origin and the median camera distance is 1 (matching
            ``DepthAnything3._normalize_extrinsics``).  Default ``False``
            because the ONNX wrapper does *not* normalise internally —
            normalisation is the caller's responsibility.

        Returns
        -------
        dict with keys ``depth``, ``depth_conf``, ``extrinsics``, ``intrinsics``.
        """
        img_batch, intrs_adj, metas = self.preprocess(imgs, intrs)
        if normalize_extrinsics:
            extrs = self._normalize_extrinsics(extrs)
        outputs = self._infer(img_batch, extrs, intrs_adj)
        return self.parse_outputs(outputs, metas)

    # ---- preprocessing ------------------------------------------------------

    def preprocess(
        self,
        imgs: list[np.ndarray | str],
        intrs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Resize, normalise, and batch *N* images for the TRT engine.

        Parameters
        ----------
        imgs :
            List of *N* BGR images or file paths.
        intrs :
            ``(N, 3, 3)`` intrinsics at the *original* image resolution.
            These are re-scaled to the target size and returned.

        Returns
        -------
        img_batch : np.ndarray
            ``(1, N, 3, H, W)`` float32, ImageNet-normalised.
        intrs_adj : np.ndarray
            ``(N, 3, 3)`` float32, intrinsics re-scaled to the target size.
        metas : list[dict]
            Per-view metadata with ``orig_h``, ``orig_w``, ``scale_x``,
            ``scale_y`` for reversing the resize in post-processing.
        """
        N = len(imgs)
        proc_imgs = np.zeros((N, 3, self.target_h, self.target_w), dtype=np.float32)
        intrs_out = np.zeros((N, 3, 3), dtype=np.float32)
        metas: list[dict] = []

        for i in range(N):
            proc_imgs[i], intrs_out[i], meta = self._preprocess_one(
                imgs[i], intrs[i],
            )
            metas.append(meta)

        # Add batch dimension: (N, 3, H, W) → (1, N, 3, H, W)
        img_batch = proc_imgs[None]
        return img_batch, intrs_out, metas

    def _preprocess_one(
        self,
        img: np.ndarray | str,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Preprocess a single view."""
        # Load
        if isinstance(img, str):
            img_bgr = cv2.imread(img, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {img}")
        elif isinstance(img, np.ndarray):
            img_bgr = img.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H,W,3), got {img_bgr.shape}")

        orig_h, orig_w = img_bgr.shape[:2]

        # RGB conversion + resize to exact target dimensions
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(
            img_rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR,
        )

        # Normalise
        img_f = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_f - self._MEAN) / self._STD
        # → (H, W, 3) → (3, H, W)
        img_chw = img_normalized.transpose(2, 0, 1).astype(np.float32)

        # Scale intrinsics
        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h
        K_adj = K.copy().astype(np.float32)
        K_adj[0, 0] *= scale_x  # fx
        K_adj[0, 2] *= scale_x  # cx
        K_adj[1, 1] *= scale_y  # fy
        K_adj[1, 2] *= scale_y  # cy

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
        return img_chw, K_adj, meta

    # ---- inference ----------------------------------------------------------

    def _infer(
        self,
        img_batch: np.ndarray,
        extrs: np.ndarray,
        intrs: np.ndarray,
        np_output: bool = True,
    ) -> dict:
        """Generic multi-input TRT inference.

        Parameters
        ----------
        img_batch : np.ndarray
            ``(1, N, 3, H, W)`` float32.
        extrs : np.ndarray
            ``(N, 4, 4)`` float32.
        intrs : np.ndarray
            ``(N, 3, 3)`` float32.
        np_output : bool
            Return numpy arrays (``True``) or torch tensors.

        Returns
        -------
        dict mapping output tensor name → numpy array.
        """
        # Build ordered list of numpy inputs matching engine input order
        input_map = {"image": img_batch, "extrinsics": extrs, "intrinsics": intrs}

        # Convert & set each input (keep tensors alive during execution)
        input_tensors: list[torch.Tensor] = []
        for inp_info in self.inputs:
            name = inp_info["name"]
            data = input_map[name]
            # Add batch dim if needed (extrinsics/intrinsics come as (N, ...))
            if data.ndim == len(inp_info["shape"]) - 1:
                data = data[None]  # add B=1
            tensor = torch.from_numpy(data.copy()).cuda().contiguous()
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
            input_tensors.append(tensor)

        # Allocate outputs
        output_tensors: dict[str, torch.Tensor] = {}
        for out_info in self.outputs:
            name = out_info["name"]
            shape = self.context.get_tensor_shape(name)
            dtype = trt_to_torch_dtype(out_info["dtype"])
            t = torch.empty(tuple(shape), dtype=dtype, device="cuda").contiguous()
            output_tensors[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        # Execute
        stream = torch.cuda.current_stream()
        success = self.context.execute_async_v3(stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT inference execution failed")
        torch.cuda.synchronize()

        if np_output:
            return self.output2numpy(output_tensors)
        return output_tensors

    def output2numpy(self, output_tensors: dict) -> dict:
        """Convert GPU output tensors to CPU numpy arrays."""
        return {name: tensor.cpu().numpy() for name, tensor in output_tensors.items()}

    # ---- postprocessing -----------------------------------------------------

    def parse_outputs(self, outputs: dict, metas: list[dict]) -> dict:
        """Extract and post-process raw TRT outputs.

        Parameters
        ----------
        outputs : dict
            Raw output dict from ``_infer``.
        metas : list[dict]
            Per-view metadata from ``preprocess``.

        Returns
        -------
        dict with ``depth``, ``depth_conf``, ``extrinsics``, ``intrinsics``.
        """
        if not outputs:
            raise RuntimeError("Model returned no outputs.")

        # Resolve output names by keyword matching (robust to prefix changes)
        def _find_key(needle: str) -> str | None:
            for k in outputs:
                if needle in k.lower():
                    return k
            return None

        depth_key = _find_key("depth")
        # Make sure we don't match "depth_conf" when looking for "depth"
        depth_keys = [k for k in outputs if "depth" in k.lower()]
        if len(depth_keys) >= 2:
            depth_key = next(
                (k for k in depth_keys if "conf" not in k.lower()), depth_key,
            )
        conf_key = _find_key("conf") or _find_key("depth_conf")
        extrs_key = (
            _find_key("pred_extrinsics") or _find_key("extrinsics")
        )

        # For intrinsics output, be careful not to match the INPUT "intrinsics"
        ixts_key = _find_key("pred_intrinsics")
        if ixts_key is None:
            # Fall back: the output "intrinsics" that is NOT an input
            input_names = {inp["name"] for inp in self.inputs}
            ixts_key = next(
                (k for k in outputs if "intrinsics" in k.lower() and k not in input_names),
                None,
            )

        if depth_key is None:
            depth_key = list(outputs.keys())[0]

        # Squeeze batch dim: (1, N, ...) → (N, ...)
        depth = outputs[depth_key].squeeze(0).astype(np.float32)            # (N, H, W)
        conf = (
            outputs[conf_key].squeeze(0).astype(np.float32)
            if conf_key is not None
            else np.ones_like(depth)
        )
        pred_extrs = (
            outputs[extrs_key].squeeze(0).astype(np.float32)
            if extrs_key is not None
            else np.zeros((depth.shape[0], 3, 4), dtype=np.float32)
        )
        pred_ixts = (
            outputs[ixts_key].squeeze(0).astype(np.float32)
            if ixts_key is not None
            else np.zeros((depth.shape[0], 3, 3), dtype=np.float32)
        )

        # Mask low-confidence depth (set to -1)
        valid = conf > self.conf_thresh
        depth_masked = depth.copy()
        depth_masked[~valid] = -1.0

        return {
            "depth": depth_masked,
            "depth_raw": depth,
            "depth_conf": conf,
            "extrinsics": pred_extrs,
            "intrinsics": pred_ixts,
        }

    # ---- extrinsics normalisation (mirrors api.py) -------------------------

    @staticmethod
    def _normalize_extrinsics(extrs: np.ndarray) -> np.ndarray:
        """Normalise extrinsics so the first camera is at the origin and the
        median camera distance equals 1.

        This is a numpy re-implementation of
        ``DepthAnything3._normalize_extrinsics``.
        """
        if extrs is None:
            return None
        # extrs: (N, 4, 4) — world-to-camera convention
        ex_t = extrs.copy()
        # First camera → identity transform (invert first extrinsics)
        transform = np.linalg.inv(ex_t[0])  # (4, 4)
        ex_t_norm = ex_t @ transform         # all cameras relative to first

        # Compute camera-to-world (inverse of w2c)
        c2ws = np.linalg.inv(ex_t_norm)     # (N, 4, 4)
        translations = c2ws[..., :3, 3]      # (N, 3)
        dists = np.linalg.norm(translations, axis=-1)  # (N,)
        median_dist = max(float(np.median(dists)), 1e-1)

        ex_t_norm[..., :3, 3] /= median_dist
        return ex_t_norm
