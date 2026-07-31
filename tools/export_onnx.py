#!/usr/bin/env python3
"""
Export a Depth Anything 3 checkpoint to ONNX.

Example (metric model saved to ./DA3METRIC-LARGE):
    python export.py --model-dir DA3METRIC-LARGE --height 518 --width 518
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from depth_anything_3.api import DepthAnything3

PATCH_SIZE = 14

# Reference-view strategy pinned for the any-view export.
#
# The default ``saddle_balanced`` strategy selects a reference view via a
# data-dependent ``argmin`` over a per-view "balance score". When one of that
# score's sub-metrics is near-constant across views (e.g. the class-token
# variance, whose spread can be ~1e-9 against values ~1e-3), ``normalize_metric``
# subtracts nearly equal fp32 numbers — catastrophic cancellation — so the score,
# and therefore the ``argmin``, is decided by rounding noise. PyTorch fp32 and
# ONNX Runtime fp32 round that noise differently and pick *different* reference
# views, which reorders the views only on one side and makes the whole multi-view
# prediction diverge. (This block runs only when no camera pose is supplied; the
# with-extrinsics graph skips it, which is why that export is faithful.)
#
# Pinning "first" removes the argmin entirely (view 0 is the reference, the
# reorder becomes an identity), giving a deterministic graph that matches PyTorch.
# Only affects the no-extrinsics export; harmless for the with-pose one.
EXPORT_REF_VIEW_STRATEGY = "first"

# ImageNet normalisation — shared by all preprocessing paths.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#See readme at: https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt/tree/main/onnx


def _get_da3_submodel(api_model: DepthAnything3) -> nn.Module:
    """Return the any-view sub-model from a (possibly nested) checkpoint.

    For nested models (``NestedDepthAnything3Net``) this is ``.model.da3``;
    for plain any-view models it is ``.model`` itself.
    """
    return getattr(api_model.model, "da3", api_model.model)


def _get_metric_submodel(api_model: DepthAnything3) -> nn.Module:
    """Return the metric sub-model from a (possibly nested) checkpoint.

    For nested models (``NestedDepthAnything3Net``) this is ``.model.da3_metric``;
    for plain metric models it is ``.model`` itself.
    """
    return getattr(api_model.model, "da3_metric", api_model.model)


def _forward_metric_submodel(api_model: DepthAnything3, image: torch.Tensor):
    """Run the metric branch on a ``(B, 3, H, W)`` image, returning (depth, sky).

    Bypasses the API-level autocast (which forces bf16 on CUDA) by calling the
    underlying ``DepthAnything3Net`` directly — its depth head already runs in
    ``autocast(..., enabled=False)``, so the graph stays float32 and
    ONNX-Runtime-CPU-compatible.  For nested checkpoints only the metric branch
    (``.da3_metric``) is traced, never the any-view giant backbone.

    The exported graph returns the **raw** network depth.  Metric depth in metres
    is a caller-side post-processing step (``metric_depth = focal * net_output /
    300``, ``focal = (fx + fy) / 2``); the metric sub-model itself never consumes
    the intrinsic matrix (its ``intrinsics`` argument is unused unless extrinsics
    or GS are enabled — see ``NestedDepthAnything3Net``), so intrinsics stay out of
    the ONNX graph entirely.
    """
    model_in = image.unsqueeze(1)  # add single-view dimension → (B, 1, 3, H, W)
    metric_model = _get_metric_submodel(api_model)

    # IMPORTANT: avoid DepthAnything3Net.forward() here.
    # The full forward applies mono sky post-processing with quantile/sort ops,
    # which export to ONNX Sort/TopK and can exceed TensorRT TopK limits.
    # For metric export we only need raw head outputs (depth + sky).
    feats, _ = metric_model.backbone(
        model_in,
        cam_token=None,
        export_feat_layers=[],
        ref_view_strategy="saddle_balanced",
    )
    H, W = model_in.shape[-2], model_in.shape[-1]
    with torch.autocast(device_type=model_in.device.type, enabled=False):
        output = metric_model._process_depth_head(feats, H, W)
    return output["depth"], output["sky"]  # depth: (B, 1, H, W)


class DepthAnything3MetricOnnxWrapper(nn.Module):
    """Metric (or monocular) wrapper: ``image`` → ``(depth, sky)``.

    Returns the raw (un-scaled) network depth and the sky mask.  Metric depth in
    metres is obtained by the caller via ``metric_depth = focal * depth / 300``
    (the intrinsic matrix is only used in that post-processing formula, never in
    the exported graph).
    """

    def __init__(self, api_model: DepthAnything3) -> None:
        super().__init__()
        self.model = api_model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        depth, sky = _forward_metric_submodel(self.model, image)
        return depth, sky


class DepthAnything3AnyViewOnnxWrapper(nn.Module):
    """Wraps the any-view sub-model (da3) from a nested checkpoint for ONNX export.

    Takes multi-view images and returns depth, confidence, and predicted camera
    parameters.  The number of views *N* is fixed at export time.

    When ``use_extrinsics`` is True the graph additionally consumes camera
    ``extrinsics`` / ``intrinsics`` as priors (fed to the camera encoder).  When
    False the graph takes only ``image`` and the model predicts poses itself —
    matching the ``extrinsics=None`` inference path.  Note the da3 sub-model only
    consumes ``intrinsics`` via the camera encoder (gated on ``extrinsics`` being
    provided), so intrinsics are baked in together with extrinsics, never alone.
    """

    def __init__(self, api_model: DepthAnything3, use_extrinsics: bool = True) -> None:
        super().__init__()
        self.model = api_model
        self.use_extrinsics = use_extrinsics

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # image:      (B, N, 3, H, W)
        # extrinsics: (B, N, 4, 4)  — only consumed when use_extrinsics is True
        # intrinsics: (B, N, 3, 3)  — only consumed when use_extrinsics is True
        if not self.use_extrinsics:
            extrinsics = None
            intrinsics = None
        output = _get_da3_submodel(self.model)(
            image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_feat_layers=[],
            infer_gs=False,
            ref_view_strategy=EXPORT_REF_VIEW_STRATEGY,
        )
        return (
            output["depth"],
            output["depth_conf"],
            output["extrinsics"],
            output["intrinsics"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Depth Anything 3 to ONNX.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Local checkpoint directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--wrapper",
        type=str,
        choices=["metric", "anyview"],
        default="metric",
        help="Wrapper type: 'metric' for DA3METRIC-LARGE, "
             "'anyview' for the any-view branch of DA3NESTED-GIANT-LARGE.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=1,
        help="Number of views N for the anyview wrapper (fixed at export time).",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default='weights/onnx_save/da3metric_large_644x490.onnx',
        help="Where to write the ONNX file (defaults to <model-name>.onnx).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=490,
        help="Input height. Must be divisible by 14.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=644,
        help="Input width. Must be divisible by 14.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the dummy export input.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version to target.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to export on (cpu or cuda).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save ONNX outputs.",
    )
    parser.add_argument(
        "--use-extrinsics", action="store_true",
        help="Any-view wrapper only: bake camera extrinsics/intrinsics into the "
             "ONNX graph as inputs (fed to the camera encoder as priors). When "
             "omitted, the graph takes only `image` and the model predicts poses "
             "itself.",
    )
    parser.add_argument(
        "--check-accuracy",
        action="store_true",
        help="After export, compare ONNX vs PyTorch outputs using Astribot "
             "set1 / frame 0 (preprocessed at the exported --height/--width).",
    )
    return parser.parse_args()


def load_model(model_dir: Path, device: torch.device) -> DepthAnything3:
    api_model = DepthAnything3.from_pretrained(model_dir.as_posix())
    api_model = api_model.to(device)
    api_model.eval()
    return api_model


def export_onnx(
    model_dir: str,
    onnx_path: Path,
    height: int,
    width: int,
    batch_size: int,
    opset: int,
    device: torch.device,
    wrapper: str = "metric",
    num_views: int = 1,
    use_extrinsics: bool = True,
) -> None:
    if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
        raise ValueError(f"height and width must be divisible by {PATCH_SIZE}.")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {model_dir} to {device}...")
    api_model = load_model(Path(model_dir), device)

    param_count = sum(p.numel() for p in api_model.parameters())
    print(f"Model parameters: {param_count/1e6:.2f}M")

    if wrapper == "anyview":
        _export_anyview(
            api_model, onnx_path, height, width, batch_size, num_views, opset, device,
            use_extrinsics=use_extrinsics,
        )
    else:
        _export_metric(api_model, onnx_path, height, width, batch_size, opset, device)

    print(f"ONNX model written to {onnx_path.resolve()}")

    print("Validating exported ONNX model...")
    _validate_onnx(onnx_path)


def _export_metric(
    api_model: DepthAnything3,
    onnx_path: Path,
    height: int,
    width: int,
    batch_size: int,
    opset: int,
    device: torch.device,
) -> None:
    """Export the metric (or monocular) model.

    Inputs ``image`` → outputs raw ``depth`` + ``sky``.  Metric depth in metres is
    a caller-side post-processing step (``metric_depth = focal * depth / 300``);
    the intrinsic matrix is not part of the exported graph.
    """
    # The metric branch may use RoPE (e.g. DinoV2-L backbone); patch it so
    # ``int(positions.max())`` doesn't crash the tracer.
    metric_sub = _get_metric_submodel(api_model)
    _patch_rope_for_export(api_model, height, width, submodel=metric_sub)

    dummy_image = torch.zeros(batch_size, 3, height, width, device=device)
    wrapper = DepthAnything3MetricOnnxWrapper(api_model).to(device)

    with torch.no_grad():
        wrapper(dummy_image)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image,),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["depth", "sky"],
            training=torch.onnx.TrainingMode.EVAL,
        )


def _replace_swiglu_fused_with_pytorch(api_model: DepthAnything3) -> None:
    """Replace xFormers ``SwiGLUFFNFused`` modules with plain-PyTorch ``SwiGLUFFN``.

    The xFormers fused kernel inspects concrete tensor properties (``data_ptr``,
    strides) at call time to dispatch to an optimized implementation.  That
    materialisation crashes during ``torch.export`` tracing because the tensors
    are ``FakeTensor``\\s.  ``SwiGLUFFN`` is a pure-PyTorch equivalent whose ops
    are all natively supported by the ONNX exporter.
    """
    from depth_anything_3.model.dinov2.layers.swiglu_ffn import (  # noqa: PLC0415
        SwiGLUFFN,
        SwiGLUFFNFused,
    )

    replaced = 0
    # The any-view branch — handles both nested (.model.da3) and plain (.model)
    da3_model = _get_da3_submodel(api_model)
    for name, child in da3_model.named_modules():
        if not isinstance(child, SwiGLUFFNFused):
            continue

        in_features = child.in_features
        hidden_features = child.hidden_features
        out_features = child.out_features
        bias = child.w12.bias is not None

        new_ffn = SwiGLUFFN(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
        # xFormers SwiGLU stores w12 (packed) + w3 by default — same layout as
        # SwiGLUFFN, so copy weights directly without repacking.
        new_ffn.w12.weight.data.copy_(child.w12.weight.data)
        new_ffn.w3.weight.data.copy_(child.w3.weight.data)
        if bias:
            new_ffn.w12.bias.data.copy_(child.w12.bias.data)
            new_ffn.w3.bias.data.copy_(child.w3.bias.data)

        # Navigate the dotted name to replace the module in-place
        parent = da3_model
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, leaf, new_ffn)
        replaced += 1

    if replaced:
        print(f"[INFO] Replaced {replaced} SwiGLUFFNFused → SwiGLUFFN for ONNX export.")


def _patch_rope_for_export(
    api_model: DepthAnything3, height: int, width: int, submodel: nn.Module | None = None,
) -> None:
    """Patch RoPE forward to avoid ``int(positions.max())`` during tracing.

    The original forward calls ``int(positions.max()) + 1`` to size the frequency
    lookup table.  That materialises a symbolic integer during ``torch.export`` and
    raises ``GuardOnDataDependentSymNode``.  Because the patch grid is fixed at
    export time we can pre-compute the value and use it directly.

    Parameters
    ----------
    submodel :
        The module tree to search.  Defaults to ``_get_da3_submodel(api_model)``
        for backward compatibility; pass ``_get_metric_submodel(api_model)`` when
        exporting the metric branch from a nested checkpoint.
    """
    from depth_anything_3.model.dinov2.layers.rope import (  # noqa: PLC0415
        RotaryPositionEmbedding2D,
    )

    if submodel is None:
        submodel = _get_da3_submodel(api_model)

    patches_h = height // PATCH_SIZE
    patches_w = width // PATCH_SIZE
    # ``patch_start_idx=1`` shifts patch positions by 1 (special tokens at 0).
    max_position = max(patches_h, patches_w) + 1

    def _make_patched_forward(max_pos: int):
        def patched_forward(self, tokens, positions):
            assert tokens.size(-1) % 2 == 0
            assert positions.ndim == 3 and positions.shape[-1] == 2
            feature_dim = tokens.size(-1) // 2
            cos_comp, sin_comp = self._compute_frequency_components(
                feature_dim, max_pos, tokens.device, tokens.dtype,
            )
            vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
            vertical_features = self._apply_1d_rope(
                vertical_features, positions[..., 0], cos_comp, sin_comp,
            )
            horizontal_features = self._apply_1d_rope(
                horizontal_features, positions[..., 1], cos_comp, sin_comp,
            )
            return torch.cat((vertical_features, horizontal_features), dim=-1)
        return patched_forward

    patched = 0
    for module in submodel.modules():
        if not isinstance(module, RotaryPositionEmbedding2D):
            continue
        module.forward = _make_patched_forward(max_position).__get__(
            module, RotaryPositionEmbedding2D,
        )
        patched += 1

    if patched:
        print(f"[INFO] Patched {patched} RoPE modules for ONNX export.")


def _export_anyview(
    api_model: DepthAnything3,
    onnx_path: Path,
    height: int,
    width: int,
    batch_size: int,
    num_views: int,
    opset: int,
    device: torch.device,
    use_extrinsics: bool = True,
) -> None:
    """Export the any-view sub-model via ``DepthAnything3AnyViewOnnxWrapper``.

    When ``use_extrinsics`` is True the exported graph takes
    ``(image, extrinsics, intrinsics)``; otherwise it takes only ``image`` and
    the model predicts camera poses itself.
    """
    _replace_swiglu_fused_with_pytorch(api_model)
    _patch_rope_for_export(api_model, height, width)
    wrapper = DepthAnything3AnyViewOnnxWrapper(api_model, use_extrinsics=use_extrinsics).to(device)
    dummy_image = torch.zeros(batch_size, num_views, 3, height, width, device=device)

    if use_extrinsics:
        dummy_extrinsics = torch.eye(4, device=device)[None, None].repeat(
            batch_size, num_views, 1, 1
        )
        dummy_intrinsics = torch.eye(3, device=device)[None, None].repeat(
            batch_size, num_views, 1, 1
        )
        export_args = (dummy_image, dummy_extrinsics, dummy_intrinsics)
        input_names = ["image", "extrinsics", "intrinsics"]
    else:
        export_args = (dummy_image,)
        input_names = ["image"]

    with torch.no_grad():
        wrapper(*export_args)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            export_args,
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["depth", "depth_conf", "pred_extrinsics", "pred_intrinsics"],
            training=torch.onnx.TrainingMode.EVAL,
        )

def _print_io_shapes(onnx_model) -> None:
    def _dims(tensor):
        dims = []
        for d in tensor.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        return dims

    for inp in onnx_model.graph.input:
        print(f"Input {inp.name}: {_dims(inp)}")
    for out in onnx_model.graph.output:
        print(f"Output {out.name}: {_dims(out)}")


def _validate_onnx(onnx_path: Path) -> None:
    """Validate an ONNX model.

    For models whose total size exceeds protobuf's 2 GiB inline limit we skip
    the in-process checker (which internally calls ``SerializeToString``) and
    verify via the command-line ``onnx-check`` tool instead.
    """
    total_size_gb = sum(
        f.stat().st_size for f in onnx_path.parent.glob(onnx_path.name + "*")
    ) / (1024**3)
    print(f"ONNX model size: {total_size_gb:.2f} GiB")

    try:
        model = onnx.load(onnx_path.as_posix(), load_external_data=True)
    except Exception:
        print("[WARN] Could not load ONNX model into memory for validation.")
        print("[WARN] The model was exported successfully but is too large for")
        print("[WARN] in-process protobuf operations.  Verify manually with:")
        print(f"[WARN]   onnx-check {onnx_path}")
        return

    if total_size_gb > 1.9:
        print("[INFO] Skipping in-process check_model (model > 2 GiB protobuf limit).")
        print(f"[INFO] Validate manually with:  onnx-check {onnx_path}")
    else:
        onnx.checker.check_model(model)
    _print_io_shapes(model)


def _load_astribot_frame0() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load Astribot set1 / frame 0 image paths, extrinsics, and intrinsics."""
    from tools.utils.astribot_dataloader import load_images_cam_params  # noqa: PLC0415

    return load_images_cam_params("set1", 0)


def _preprocess_views(
    image_paths: list[str], intrs: np.ndarray, height: int, width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-resize *N* images to ``(N, 3, H, W)`` and scale their intrinsics.

    Matches the preprocessing baked into the exported ONNX models (plain resize
    to the export resolution, ImageNet normalisation).
    """
    N = len(image_paths)
    proc = np.zeros((N, 3, height, width), dtype=np.float32)
    intrs_out = np.zeros((N, 3, 3), dtype=np.float32)

    for i, path in enumerate(image_paths):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not load: {path}")
        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        img_f = rgb_r.astype(np.float32) / 255.0
        proc[i] = ((img_f - _MEAN) / _STD).transpose(2, 0, 1)

        K = intrs[i].copy().astype(np.float32)
        sx, sy = width / orig_w, height / orig_h
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        intrs_out[i] = K

    return proc, intrs_out


def _find_onnx_key(needle: str, d: dict) -> np.ndarray | None:
    """Case-insensitive key lookup in an ONNX output dict."""
    for k, v in d.items():
        if needle in k.lower():
            return v
    return None


def run_metric_accuracy_check(
    onnx_path: Path,
    api_model: DepthAnything3,
    height: int,
    width: int,
    device: str = "cuda",
) -> None:
    """Compare metric ONNX outputs against PyTorch using Astribot set1 / frame 0.

    Uses the first camera view (head_rgbd), preprocesses at the exported
    ``height``/``width`` for both backends, and reports raw depth / sky error
    stats.  (Metric depth in metres is a caller-side ``focal * depth / 300`` step,
    not part of the exported graph, so it is not exercised here.)
    """
    print("\n" + "=" * 72)
    print("[ACCURACY] Loading Astribot set1 / frame 0 (first view only) ...")

    images, _, ixts_np = _load_astribot_frame0()
    img_path = images[0]
    print(f"[ACCURACY] Image: {img_path}")

    # --- Preprocess (exact resize to the exported resolution) ---------------
    proc, _ = _preprocess_views([img_path], ixts_np[:1], height, width)
    img_chw = proc[0]              # (3, H, W)
    print(f"[ACCURACY] Preprocessed to: {height}x{width}")

    # --- PyTorch forward ----------------------------------------------------
    dev = torch.device(device)
    # DepthAnything3Net expects (B, N, 3, H, W)
    img_t = torch.from_numpy(img_chw)[None, None].to(dev).float()  # (1, 1, 3, H, W)

    metric_model = _get_metric_submodel(api_model)
    print("[ACCURACY] Running PyTorch metric forward ...")
    with torch.no_grad():
        pt_out = metric_model(
            img_t,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
        )

    # Snapshot PyTorch outputs to CPU numpy, then free all GPU memory (model +
    # activations + input tensor) BEFORE the ONNX Runtime CUDA session allocates
    # its arena, so the two backends don't contend for VRAM (see the any-view
    # accuracy check for the same OOM guard).
    pt_np = {k: pt_out[k].float().cpu().numpy().squeeze() for k in ["depth", "sky"]}
    del pt_out, img_t
    api_model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- ONNX forward -------------------------------------------------------
    print("[ACCURACY] Running ONNX metric forward ...")
    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=providers)
    print(f"[ACCURACY] ONNX Runtime using: {sess.get_providers()[0]}")

    onnx_inputs = {"image": img_chw[None].astype(np.float32)}  # (1, 3, H, W)
    onnx_outputs = sess.run([o.name for o in sess.get_outputs()], onnx_inputs)
    onx = dict(zip([o.name for o in sess.get_outputs()], onnx_outputs))

    # --- Compare ------------------------------------------------------------
    print("\n[ACCURACY] Per-output comparison (PyTorch vs ONNX):")
    for key in ["depth", "sky"]:
        pt_val = pt_np[key]
        onx_val = _find_onnx_key(key, onx)
        if onx_val is None:
            print(f"  {key:15s}  SKIP (not found in ONNX outputs)")
            continue
        onx_val = onx_val.squeeze()

        abs_diff = np.abs(pt_val - onx_val)
        rel_err = abs_diff / (np.abs(pt_val) + 1e-6)
        print(
            f"  {key:15s}  "
            f"max_abs={float(abs_diff.max()):.4e}  "
            f"mean_abs={float(abs_diff.mean()):.4e}  "
            f"max_rel={float(rel_err.max()):.4e}  "
            f"mean_rel={float(rel_err.mean()):.4e}  "
            f"shape={list(pt_val.shape)}"
        )

    # Allclose
    depth_onx = _find_onnx_key("depth", onx).squeeze()
    print(f"\n[ACCURACY] depth    allclose(atol=1e-2): "
          f"{np.allclose(pt_np['depth'], depth_onx, atol=1e-2)}")

    sky_onx = _find_onnx_key("sky", onx)
    if sky_onx is not None:
        print(f"[ACCURACY] sky      allclose(atol=1e-2): "
              f"{np.allclose(pt_np['sky'], sky_onx.squeeze(), atol=1e-2)}")
    print("=" * 72)


def run_anyview_accuracy_check(
    onnx_path: Path,
    api_model: DepthAnything3,
    height: int,
    width: int,
    num_views: int,
    device: str = "cuda",
    use_extrinsics: bool = True,
) -> None:
    """Compare any-view ONNX outputs against PyTorch using real multi-view data.

    Loads ``set1`` / frame 0 from the Astribot dataset, preprocesses at the
    exported ``height``/``width`` for both backends, and reports per-output error
    statistics.  When ``use_extrinsics`` is False the extrinsics/intrinsics priors
    are not fed to either backend (the model predicts poses), matching the graph
    exported without ``--use-extrinsics``.
    """
    print("\n" + "=" * 72)
    print("[ACCURACY] Loading Astribot set1 / frame 0 ...")

    images, exts_np, ixts_np = _load_astribot_frame0()
    print(f"[ACCURACY] Loaded {len(images)} views")
    for p in images:
        print(f"  {p}")

    # --- Preprocess (exact resize to the exported resolution) ---------------
    proc, ixts_scaled = _preprocess_views(images, ixts_np, height, width)
    print(f"[ACCURACY] Preprocessed to: {height}x{width}")

    # Prepare model inputs (move to device, add batch dim)
    dev = torch.device(device)
    imgs = torch.from_numpy(proc)[None].to(dev).float()              # (1, N, 3, H, W)
    ex_t = torch.from_numpy(exts_np)[None].to(dev).float()           # (1, N, 4, 4)
    in_t = torch.from_numpy(ixts_scaled)[None].to(dev).float()       # (1, N, 3, 3)

    # Normalize extrinsics (same as model._normalize_extrinsics)
    from depth_anything_3.utils.geometry import affine_inverse  # noqa: PLC0415

    transform = affine_inverse(ex_t[:, :1])
    ex_t_norm = ex_t @ transform
    c2ws = affine_inverse(ex_t_norm)
    dists = c2ws[..., :3, 3].norm(dim=-1)
    median_dist = torch.median(dists).clamp(min=1e-1)
    ex_t_norm[..., :3, 3] /= median_dist

    # --- PyTorch forward (the same call the ONNX wrapper traces) --------------
    print("[ACCURACY] Running PyTorch forward ...")
    with torch.no_grad():
        pt_out = _get_da3_submodel(api_model)(
            imgs,
            extrinsics=ex_t_norm if use_extrinsics else None,
            intrinsics=in_t if use_extrinsics else None,
            export_feat_layers=[],
            infer_gs=False,
            ref_view_strategy=EXPORT_REF_VIEW_STRATEGY,
        )

    # Snapshot the ONNX inputs and PyTorch outputs as CPU numpy, then release all
    # GPU memory (the giant model, its activations, and the input tensors) BEFORE
    # the ONNX Runtime CUDA session allocates its arena.  Otherwise the two
    # backends contend for VRAM and the any-view attention Softmax OOMs.
    pt_np = {
        k: pt_out[k].float().cpu().numpy()
        for k in ["depth", "depth_conf", "extrinsics", "intrinsics"]
    }
    onnx_inputs = {"image": imgs.float().cpu().numpy()}
    if use_extrinsics:
        onnx_inputs["extrinsics"] = ex_t_norm.float().cpu().numpy()
        onnx_inputs["intrinsics"] = in_t.float().cpu().numpy()
    del pt_out, imgs, ex_t, in_t, ex_t_norm, c2ws, dists, transform, median_dist
    api_model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- ONNX forward --------------------------------------------------------
    print("[ACCURACY] Running ONNX forward ...")
    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=providers)
    print(f"[ACCURACY] ONNX Runtime using: {sess.get_providers()[0]}")

    onnx_outputs = sess.run(
        ["depth", "depth_conf", "pred_extrinsics", "pred_intrinsics"], onnx_inputs,
    )
    onx = dict(zip(["depth", "depth_conf", "extrinsics", "intrinsics"], onnx_outputs))

    # --- Compare -------------------------------------------------------------
    print("\n[ACCURACY] Per-output comparison (PyTorch vs ONNX):")
    for key in ["depth", "depth_conf", "extrinsics", "intrinsics"]:
        pt_val = pt_np[key]
        onx_val = onx[key]
        abs_diff = np.abs(pt_val - onx_val)
        rel_err = abs_diff / (np.abs(pt_val) + 1e-6)
        print(
            f"  {key:15s}  "
            f"max_abs={float(abs_diff.max()):.4e}  "
            f"mean_abs={float(abs_diff.mean()):.4e}  "
            f"max_rel={float(rel_err.max()):.4e}  "
            f"mean_rel={float(rel_err.mean()):.4e}  "
            f"shape={list(pt_val.shape)}"
        )

    # Also check structural equivalence
    depth_match = np.allclose(pt_np["depth"], onx["depth"], atol=1e-2)
    print(f"\n[ACCURACY] depth    allclose(atol=1e-2): {depth_match}")
    conf_match = np.allclose(pt_np["depth_conf"], onx["depth_conf"], atol=1e-2)
    print(f"[ACCURACY] depth_conf allclose(atol=1e-2): {conf_match}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    model_name = (
        Path(args.model_dir).name
        if Path(args.model_dir).exists()
        else args.model_dir.rstrip("/").split("/")[-1]
    )
    out_dir = Path(args.output_dir)
    onnx_path = Path(args.onnx_path) if args.onnx_path else out_dir / f"{model_name}.onnx"
    export_onnx(
        model_dir=args.model_dir,
        onnx_path=onnx_path,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        opset=args.opset,
        device=torch.device(args.device),
        wrapper=args.wrapper,
        num_views=args.num_views,
        use_extrinsics=args.use_extrinsics,
    )
    if args.check_accuracy:
        # Reload a fresh model for the PyTorch baseline (the export path
        # mutates modules in-place for ONNX compatibility).
        print("\n[ACCURACY] Reloading model for PyTorch baseline ...")
        fresh_model = load_model(Path(args.model_dir), torch.device(args.device))
        if args.wrapper == "anyview":
            run_anyview_accuracy_check(
                onnx_path=onnx_path,
                api_model=fresh_model,
                height=args.height,
                width=args.width,
                num_views=args.num_views,
                device=args.device,
                use_extrinsics=args.use_extrinsics,
            )
        else:
            run_metric_accuracy_check(
                onnx_path=onnx_path,
                api_model=fresh_model,
                height=args.height,
                width=args.width,
                device=args.device,
            )


if __name__ == "__main__":
    main()