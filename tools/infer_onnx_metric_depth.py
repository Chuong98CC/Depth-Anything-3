#!/usr/bin/env python3
"""Run inference with a Depth Anything 3 metric ONNX model.

Loads a single-image metric ONNX model, runs a forward pass on one image or a
directory of images, and saves a colour-mapped depth visualization (with sampled
depth values overlaid on a grid) per image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from depth_anything_3.utils.visualize import visualize_depth


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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run ONNX inference for Depth Anything 3.")
	parser.add_argument(
		"--onnx-path",
		type=str,
		default="weights/onnx_save/da3metric_large_644x490_simplified.onnx",
		help="Path to an exported ONNX model.",
	)

	input_group = parser.add_mutually_exclusive_group(required=True)
	input_group.add_argument(
		"--image",
		type=str,
		default=None,
		help="Path to a single input image.",
	)
	input_group.add_argument(
		"--image-dir",
		type=str,
		default=None,
		help="Directory containing input images.",
	)

	parser.add_argument(
		"--glob",
		type=str,
		default="*.jpeg",
		help="Glob used with --image-dir (for example: '*.jpg', '*.png').",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default='outputs',
		help="Directory where depth visualizations are saved.",
	)
	parser.add_argument(
		"--fx",
		type=float,
		default=400.88,
		help="Original focal length fx (pixels), used by metric depth scaling printout.",
	)
	parser.add_argument(
		"--grid",
		type=int,
		nargs=2,
		metavar=("ROWS", "COLS"),
		default=[20, 20],
		help="Grid sampling size as two integers: --grid ROWS COLS (default: 20 20).",
	)
	return parser.parse_args()


def _collect_images(single_image: str | None, image_dir: str | None, pattern: str) -> list[Path]:
	if single_image is not None:
		return [Path(single_image)]

	assert image_dir is not None
	root = Path(image_dir)
	if not root.is_dir():
		raise NotADirectoryError(f"Image directory not found: {root}")
	images = sorted(root.glob(pattern))
	if not images:
		raise FileNotFoundError(f"No images matched pattern '{pattern}' in: {root}")
	return images


def main() -> None:
	args = parse_args()
	grid_rows, grid_cols = args.grid
	if grid_rows <= 0 or grid_cols <= 0:
		raise ValueError("--grid values must be positive integers")

	onnx_path = Path(args.onnx_path)
	images = _collect_images(args.image, args.image_dir, args.glob)

	output_dir = Path(args.output_dir) if args.output_dir else None
	if output_dir is not None:
		output_dir.mkdir(parents=True, exist_ok=True)

	for image_path in images:
		depth_out_path = None
		if output_dir is not None:
			depth_out_path = output_dir / f"{image_path.stem}_depth.png"

		run_onnx_demo(
			onnx_path=onnx_path,
			image_path=image_path,
			depth_out_path=depth_out_path,
			fx_orig=args.fx,
			grid_rows=grid_rows,
			grid_cols=grid_cols,
		)
		saved_path = depth_out_path if depth_out_path is not None else image_path.with_name(
			f"{image_path.stem}_depth.png"
		)
		print(f"[DONE] {image_path} -> {saved_path}")


if __name__ == "__main__":
	main()
