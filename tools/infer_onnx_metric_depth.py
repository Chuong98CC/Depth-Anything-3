#!/usr/bin/env python3
"""Run inference with a Depth Anything 3 metric ONNX model.

Loads a single-image metric ONNX model via ``DA3MetricONNX`` (shared letterbox
preprocessing + session), runs a forward pass on one image or a directory of
images, and saves a colour-mapped depth visualization (with sampled depth values
overlaid on a grid) per image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure tools/ is on sys.path for the model.* imports.
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))

from depth_anything_3.utils.visualize import visualize_depth  # noqa: E402
from model.da3metric import DA3MetricONNX  # noqa: E402


def process_image(
    model: DA3MetricONNX,
    image_path: Path,
    depth_out_path: Path,
    fx_orig: float,
    grid_rows: int,
    grid_cols: int,
) -> None:
    """Run one metric forward pass, save the depth visualization, print stats."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive integers.")

    # Only fx/fy feed the metric focal; the principal point is never read back,
    # so a bare fx == fy matrix suffices (the wrapper scales it by the letterbox).
    K = np.array([[fx_orig, 0.0, 0.0], [0.0, fx_orig, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Letterbox preprocess (aspect-preserving resize + pad) via the shared mixin.
    img_batch, intrs_adj, metas = model.preprocess_views([image_path.as_posix()], K[None])
    depth, sky = model.infer_view(img_batch[0, 0])

    # Crop the padded outputs back to the unpadded tile region.
    meta = metas[0]
    depth = model.crop_to_tile(depth, meta)
    sky = model.crop_to_tile(sky, meta)

    depth_vis = visualize_depth(depth, ret_type=np.uint8)
    vis_img = Image.fromarray(depth_vis)

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
            draw.text(
                (text_x, text_y),
                label,
                font=font,
                fill=(255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )

    vis_img.save(depth_out_path)

    print(
        f"[DEMO] Sky stats: min={float(sky.min()):.4f}, "
        f"max={float(sky.max()):.4f}, mean={float(sky.mean()):.4f}"
    )

    # Metric depth uses the letterbox-scaled focal (uniform scale on fx == fy).
    focal = float((intrs_adj[0, 0, 0] + intrs_adj[0, 1, 1]) / 2.0)
    metric_depth = focal * depth / 300.0
    print(
        f"[DEMO] Metric depth (focal={focal:.2f}): "
        f"min={metric_depth.min():.4f} m, max={metric_depth.max():.4f} m"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metric ONNX inference for Depth Anything 3.")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="weights/da3_metric_644x490_giant-large-1.1.onnx",
        help="Path to an exported metric ONNX model.",
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
        default="outputs",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX Runtime device.",
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
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    images = _collect_images(args.image, args.image_dir, args.glob)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the ONNX session once; reuse across all images.
    model = DA3MetricONNX(onnx_path.as_posix(), device=args.device)

    for image_path in images:
        depth_out_path = output_dir / f"{image_path.stem}_depth.png"
        process_image(
            model=model,
            image_path=image_path,
            depth_out_path=depth_out_path,
            fx_orig=args.fx,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        print(f"[DONE] {image_path} -> {depth_out_path}")


if __name__ == "__main__":
    main()
