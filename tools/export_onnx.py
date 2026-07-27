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

import torch
from torch import nn

import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

PATCH_SIZE = 14

#See readme at: https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt/tree/main/onnx

class DepthAnything3OnnxWrapper(nn.Module):
    """Simplified forward that takes (B, 3, H, W) and returns depth (+ sky mask if available)."""

    def __init__(self, api_model: DepthAnything3) -> None:
        super().__init__()
        self.model = api_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # The caller is expected to validate shapes before export; keep traced graph minimal.
        model_in = image.unsqueeze(1)  # add single-view dimension

        # Bypass the API-level autocast (which forces bf16 on CUDA) — call the
        # underlying DepthAnything3Net directly.  It already wraps the depth head
        # in ``autocast(..., enabled=False)``, so the full graph stays float32
        # and remains compatible with ONNX Runtime CPU providers.
        output = self.model.model(
            model_in,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
        )
        depth = output["depth"]  # [B, 1, H, W] for monocular models
        sky = output["sky"]
        return depth, sky


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Depth Anything 3 to ONNX.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/da3metric-large",
        help="Local checkpoint directory or Hugging Face repo id.",
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
        "--demo-image",
        type=str,
        default="data/frame_0100.jpeg",
        help="Path to demo image for ONNX forward pass demo.",
    )
    parser.add_argument(
        "--demo-fx",
        type=float,
        default=400.88,
        help="Original focal length fx (pixels) for metric scaling; scaled by resize factor.",
    )
    parser.add_argument(
        "--demo-output",
        type=str,
        default=None,
        help="Path to save the demo depth visualization (PNG). Defaults to <image>_depth.png.",
    )
    parser.add_argument(
        "--demo-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for ONNX Runtime inference demo (default: cuda).",
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

    wrapper = DepthAnything3OnnxWrapper(api_model).to(device)
    dummy_input = torch.zeros(batch_size, 3, height, width, device=device)

    with torch.no_grad():
        output = wrapper(dummy_input)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["depth", "sky"],
            training=torch.onnx.TrainingMode.EVAL,
        )

    print(f"ONNX model written to {onnx_path.resolve()}")

    print("Validating exported ONNX model...")
    model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(model)
    _print_io_shapes(model)

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


def _infer_size_from_input(sess_input, default_h: int, default_w: int) -> tuple[int, int]:
    shape = sess_input.shape
    # Expect [B, 3, H, W]; use defaults if symbolic/None.
    h = shape[2] if isinstance(shape[2], int) else default_h
    w = shape[3] if isinstance(shape[3], int) else default_w
    return int(h), int(w)


def preprocess_image(image_path: Path, target_h: int, target_w: int):
    """Load and preprocess an image to (1, 3, H, W) normalized float32."""
    img = Image.open(image_path).convert("RGB").resize((target_w, target_h), Image.BILINEAR)
    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    tensor = transform(img).unsqueeze(0)  # (1,3,H,W)
    return tensor.numpy().astype(np.float32), tensor


def run_onnx_demo(
    onnx_path: Path,
    image_path: Path,
    depth_out_path: Path | None = None,
    fx_orig: float = 858.0,
    grid_rows: int = 20,
    grid_cols: int = 20,
    device: str = "cuda",
) -> dict:
    """Run a single ONNX forward pass for verification and print depth stats."""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Prefer GPU providers; fall back to CPU gracefully.
    if device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=providers)
    actual_provider = sess.get_providers()[0]
    print(f"[DEMO] ONNX Runtime using: {actual_provider}")
    input_name = sess.get_inputs()[0].name
    target_h, target_w = _infer_size_from_input(sess.get_inputs()[0], default_h=518, default_w=518)

    inp_np, _ = preprocess_image(image_path, target_h, target_w)

    outputs = sess.run([o.name for o in sess.get_outputs()], {input_name: inp_np})
    out_dict = dict(zip([o.name for o in sess.get_outputs()], outputs))

    depth = out_dict["depth"].squeeze().astype(np.float32)  # (H,W)
    # Save visualization instead of raw npy
    if depth_out_path is None:
        depth_out_path = image_path.with_name(f"{image_path.stem}_depth.png")
    depth_vis = visualize_depth(depth, ret_type=np.uint8)
    vis_img = Image.fromarray(depth_vis)

    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive integers.")

    draw = ImageDraw.Draw(vis_img)
    font = ImageFont.load_default()
    row_positions = np.linspace(0, depth.shape[0] - 1, num=grid_rows, dtype=int)
    col_positions = np.linspace(0, depth.shape[1] - 1, num=grid_cols, dtype=int)

    for y in row_positions:
        for x in col_positions:
            d = float(depth[y, x])
            label = f"{d:.2f}"

            # Draw point marker with contrast to stay visible on any colormap.
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 255), outline=(0, 0, 0))

            text_x = min(x + 3, depth.shape[1] - 1)
            text_y = min(y + 3, depth.shape[0] - 1)
            draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))

    vis_img.save(depth_out_path)


    sky = out_dict["sky"].squeeze()
    print(
        f"[DEMO] Sky stats: min={float(sky.min()):.4f}, "
        f"max={float(sky.max()):.4f}, mean={float(sky.mean()):.4f}"
    )

    # Compute focal scaling based on original intrinsics and resize
    orig_w, orig_h = Image.open(image_path).size
    proc_h, proc_w = depth.shape
    scale_x = proc_w / orig_w
    scale_y = proc_h / orig_h
    fx_scaled = fx_orig * scale_x
    fy_scaled = fx_orig * scale_y
    focal = (fx_scaled + fy_scaled) / 2.0
    metric_depth = focal * depth / 300.0
    print(
        f"[DEMO] Metric depth (focal={focal:.2f}): "
        f"min={metric_depth.min():.4f} m, max={metric_depth.max():.4f} m"
    )


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
    )
    if args.demo_image:
        run_onnx_demo(
            onnx_path=onnx_path,
            image_path=Path(args.demo_image),
            depth_out_path=Path(args.demo_output) if args.demo_output else None,
            fx_orig=args.demo_fx,
            device=args.demo_device,
        )


if __name__ == "__main__":
    main()