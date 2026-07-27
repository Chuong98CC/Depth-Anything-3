#!/usr/bin/env python3
"""Run inference with a Depth Anything 3 ONNX model.

This script intentionally reuses utilities from tools/export_onnx.py to avoid
duplicating preprocessing and postprocessing logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from export_onnx import run_onnx_demo


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
