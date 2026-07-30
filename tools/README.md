# Tools

Standalone command-line scripts for exporting, running, and validating Depth
Anything 3 models. Each script imports from the installed `depth_anything_3`
package (not via relative imports) and should be **run from the repository
root**.

```bash
# Always run from repo root, e.g.
python tools/export_onnx.py --help
```

This README documents the top-level scripts in `tools/` that expose a
`main()` entry point and a command-line argument parser. Helper modules
(`astribot_dataloader.py`, `scale_intrinsics.py`) and the `tools/model/`,
`tools/scripts/` subfolders are not covered here.

---

## Export

### `export_onnx.py`

Export a Depth Anything 3 checkpoint to ONNX. Supports two wrapper types and an
optional PyTorch-vs-ONNX accuracy check on real multi-view data.

- **`metric`** wrapper — single-image depth + sky (`DA3METRIC-LARGE`, or the
  metric branch of a nested checkpoint). Input `image`; outputs raw network
  `depth` + `sky`. Metric depth in metres is a **caller-side** post-processing
  step (`metric_depth = focal * depth / 300`, `focal = (fx + fy) / 2`); the
  intrinsic matrix is only used in that formula, never inside the ONNX graph, so
  no intrinsics input is exported.
- **`anyview`** wrapper — multi-view depth, confidence, and predicted camera
  parameters (the any-view branch of a nested checkpoint). Handles xFormers
  SwiGLU and RoPE tracing issues automatically.

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Local checkpoint dir or HuggingFace repo id |
| `--wrapper` | `metric` | `metric` or `anyview` |
| `--num-views` | `3` | Number of views (cameras) `N` for the anyview wrapper (fixed at export). Default 3 for Head RGB, Head Stereo Left and Right. |
| `--onnx-path` | `path/to/output_ckpt.onnx` | Output ONNX path |
| `--height` | `490` | Input height (must be divisible by 14) |
| `--width` | `644` | Input width (must be divisible by 14) |
| `--batch-size` | `1` | Dummy export batch size |
| `--opset` | `20` | ONNX opset version |
| `--device` | `cuda` | Export device |
| `--output-dir` | `.` | Output directory (used if `--onnx-path` unset) |
| `--check-accuracy` | off | Compare ONNX vs PyTorch on Astribot set1 / frame 0 (preprocessed at the exported `--height`/`--width`) |

**Metric model I/O:**

| Inputs | Outputs |
|---|---|
| `image [B,3,H,W]` | `depth [B,1,H,W]` (raw), `sky [B,1,H,W]` |

```bash
# Metric model (raw depth; apply focal * depth / 300 downstream for metres)
python tools/export_onnx.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --wrapper metric --height 490 --width 644 \
    --onnx-path weights/da3_metric_644x490.onnx --check-accuracy

# Any-view branch of a nested checkpoint (3 views)
python tools/export_onnx.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --wrapper anyview --num-views 3 --height 490 --width 644 \
    --onnx-path weights/da3_anyview_n3_644x490.onnx --check-accuracy
```

### `export_trt.py`

Build a TensorRT engine from an ONNX file via `trtexec`. The output engine path
defaults to `<onnx_dir>/<onnx_stem>_<precision>.engine`.

| Argument | Default | Description |
|---|---|---|
| `onnx_path` (positional) | — | Path to the input ONNX file |
| `--trt_path` | `None` | Output engine path (auto-derived if omitted) |
| `--precision` | `fp16` | `fp16`, `tf32`, or `fp32` |

Basic usage
```bash
python tools/export_trt.py weights/da3_anyview_n3_644x490.onnx --precision fp16
```
To avoid environment set-up headache, please use the `tools/scripts/export_trt_docker.sh` instead,
To inference the TRT model, install the matching TRT version in your python env by:
```bash
pip install tensorrt-cu12==10.16.1.11
```
---

## Inference

### `infer_pytorch.py`

Run the PyTorch model on the Astribot stereo dataset. Loads camera extrinsics /
intrinsics (scaled to 640×480), selects a camera set, and feeds synchronized
multi-view frames to `model.inference`.

| Argument | Default | Description |
|---|---|---|
| `--camera-set` | `set1` | `set1` (3 views) or `set2` (4 views) |
| `--frame` | `0` | Single frame index (0-based) |
| `--model-name` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Model name or HuggingFace Hub id |
| `--export-dir` | `output` | Directory to export results |
| `--export-format` | `mini_npz-glb-depth_vis` | Hyphen-separated export formats |
| `--process-res` | `644` | Base processing resolution |
| `--infer-gs` | off | Enable Gaussian Splatting branch |
| `--device` | auto | Torch device (auto-detect cuda/cpu) |
| `--use-ray-pose` | off | Use ray pose for inference |
| `--show-cameras` | off | Show camera frustums in exported GLB |

```bash
python tools/infer_pytorch.py --camera-set set1 --frame 0 \
    --model-name depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

### `infer_onnx_nested.py`

Run the nested model via two split ONNX models (any-view + metric), aligning
their outputs with `align_anyview_with_metric`. Saves a `result.npz` per frame
matching the PyTorch output fields (depth, depth_conf, extrinsics, intrinsics).

| Argument | Default | Description |
|---|---|---|
| `--camera-set` | `set1` | `set1` (3 views) or `set2` (4 views) |
| `--frame` | `None` | Single frame index (use `--all-frames` for all) |
| `--all-frames` | off | Process all frames common to the selected cameras |
| `--onnx-anyview` | `weights/da3_anyview_n3_644x490_giant-large-1.1.onnx` | Any-view ONNX path |
| `--onnx-metric` | `weights/da3_metric_644x490_giant-large-1.1.onnx` | Metric ONNX path |
| `--export-dir` | `output_onnx` | Directory to save `result.npz` per frame |
| `--device` | `cuda` | ONNX Runtime device |
| `--no-align-input-ext-scale` | off (alignment on) | Disable the Umeyama alignment of the prediction to the input camera poses. On by default, matching `DepthAnything3.inference(align_to_input_ext_scale=True)`; when enabled, output extrinsics are the input poses and depth is rescaled to the input pose scale. |

```bash
python tools/infer_onnx_nested.py --camera-set set1 --frame 0
```

### `infer_onnx_metric_depth.py`

Run a single-image metric ONNX model on one image or a directory of images.
Reuses the preprocessing / postprocessing from `export_onnx.py` and saves a
depth visualization PNG per image. Exactly one of `--image` / `--image-dir` is
required.

| Argument | Default | Description |
|---|---|---|
| `--onnx-path` | `weights/onnx_save/da3metric_large_644x490_simplified.onnx` | Path to the ONNX model |
| `--image` | — | Single input image (mutually exclusive with `--image-dir`) |
| `--image-dir` | — | Directory of input images (mutually exclusive with `--image`) |
| `--glob` | `*.jpeg` | Glob pattern used with `--image-dir` |
| `--output-dir` | `outputs` | Directory for depth-visualization PNGs |
| `--fx` | `400.88` | Original focal length fx for the metric-depth printout |
| `--grid` | `20 20` | Grid sample size as `ROWS COLS` |

```bash
python tools/infer_onnx_metric_depth.py \
    --onnx-path weights/da3_metric_644x490.onnx --image data/frame_0100.jpeg
```

---

## Validation / comparison

### `compare_nested_onnx_pt.py`

Compare the end-to-end PyTorch nested model against the split-ONNX pipeline
(any-view ONNX + metric ONNX + `align_anyview_with_metric`) on Astribot set1 /
frame 0. Reports per-output absolute / relative error and `allclose` checks.

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | PyTorch nested checkpoint (HuggingFace id or local dir) |
| `--onnx-anyview` | *required* | Any-view ONNX path |
| `--onnx-metric` | *required* | Metric ONNX path |
| `--device` | `cuda` | Device for PyTorch / ONNX Runtime |
| `--height` | `490` | Target height (must match ONNX inputs) |
| `--width` | `644` | Target width (must match ONNX inputs) |

```bash
python tools/compare_nested_onnx_pt.py \
    --onnx-anyview weights/da3_anyview_n3_644x490_giant-large-1.1.onnx \
    --onnx-metric  weights/da3_metric_644x490_giant-large-1.1.onnx
```

### `compare_onnx_trt.py`

Compare any-view ONNX vs TensorRT outputs on Astribot set1 / frame 0 using
identical shared preprocessing. Reports per-output error statistics and
`allclose` checks.

| Argument | Default | Description |
|---|---|---|
| `--onnx-path` | *required* | Any-view ONNX path |
| `--trt-path` | *required* | Any-view TensorRT engine path |
| `--device` | `cuda` | ONNX Runtime device (TRT always uses CUDA) |

```bash
python tools/compare_onnx_trt.py \
    --onnx-path weights/da3_anyview_n3_644x490_small2.onnx \
    --trt-path  weights/da3_anyview_n3_644x490_small2_fp16.engine
```
