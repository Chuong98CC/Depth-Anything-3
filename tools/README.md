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
  SwiGLU and RoPE tracing issues automatically. By default the graph takes only
  `image` and the model predicts its own camera poses. Pass **`--use-extrinsics`**
  to additionally bake `extrinsics`/`intrinsics` inputs into the graph (fed to the
  camera encoder as priors) — export it to a **`-with-camera-pose`**-suffixed path
  so the inference/comparison scripts can find it.

  > The any-view model consumes `intrinsics` only through the camera encoder,
  > which is gated on `extrinsics` being provided. So the two are baked in together
  > by `--use-extrinsics` or not at all — there is no intrinsics-only mode.

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Local checkpoint dir or HuggingFace repo id |
| `--wrapper` | `metric` | `metric` or `anyview` |
| `--num-views` | `3` | Number of views (cameras) `N` for the anyview wrapper (fixed at export). Default 3 for Head RGB, Head Stereo Left and Right. |
| `--use-extrinsics` | off | Any-view only: add `extrinsics`/`intrinsics` inputs (camera-pose priors). Omit → `image`-only graph, poses predicted. |
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

**Any-view model I/O:**

| Variant | Inputs | Outputs |
|---|---|---|
| default | `image [B,N,3,H,W]` | `depth`, `depth_conf`, `pred_extrinsics`, `pred_intrinsics` |
| `--use-extrinsics` | `image [B,N,3,H,W]`, `extrinsics [B,N,4,4]`, `intrinsics [B,N,3,3]` | `depth`, `depth_conf`, `pred_extrinsics`, `pred_intrinsics` |

```bash
# Metric model (raw depth; apply focal * depth / 300 downstream for metres)
python tools/export_onnx.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --wrapper metric --height 490 --width 644 \
    --onnx-path weights/da3_metric_644x490.onnx --check-accuracy

# Any-view branch, default (image-only; model predicts poses)
python tools/export_onnx.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --wrapper anyview --num-views 3 --height 490 --width 644 \
    --onnx-path weights/da3_anyview_n3_644x490_giant-large-1.1.onnx --check-accuracy

# Any-view branch, with camera pose (extrinsics/intrinsics as priors)
python tools/export_onnx.py --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --wrapper anyview --num-views 3 --height 490 --width 644 --use-extrinsics \
    --onnx-path weights/da3_anyview_n3_644x490_giant-large-1.1-with-camera-pose.onnx \
    --check-accuracy
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
To avoid environment set-up headache, please use the docker command instead:
```
./tools/scripts/export_trt_docker.sh <absolute/path/to/checkpoint.onnx>
```
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

### `infer_da3_onnx_trt.py`

Run a selected module (`metric`, `anyview`, or `nested`) on the Astribot dataset
via a selectable backend (`onnx` or `trt`), using the shared `model/` wrappers.
Saves a `result.npz` per frame — cropped to each view's tile:

- `metric` — per-view depth + sky (mono-sky post-processing applied).
- `anyview` — multi-view depth, depth_conf, predicted extrinsics/intrinsics.
- `nested` — any-view + metric + alignment (the full pipeline).

| Argument | Default | Description |
|---|---|---|
| `--backend` | `onnx` | Inference backend: `onnx` or `trt` |
| `--module` | `nested` | Module to run: `metric`, `anyview`, or `nested` |
| `--use-extrinsics` | off | Any-view/nested: select the `-with-camera-pose` model and feed camera extrinsics/intrinsics as priors. Omit → plain model, poses predicted. |
| `--keep-predicted-pose` | off | Nested + `--use-extrinsics`: keep the model's **predicted** poses rigidly aligned into the input frame (Umeyama R+t) and leave depth at the predicted metric scale, instead of replacing the output with the input poses and rescaling depth (`align_scale=False`). No effect without input poses. |
| `--camera-set` | `set1` | `set0` (2 views), `set1` (3), or `set2` (4); must match the export |
| `--frame` | `None` | Single frame index (use `--all-frames` for all) |
| `--all-frames` | off | Process all frames common to the selected cameras |
| `--anyview-model` | backend + `--use-extrinsics` default | Any-view ONNX/engine path |
| `--metric-model` | backend default `.onnx`/`.engine` | Metric ONNX/engine path |
| `--export-dir` | `output_infer` | Directory to save `result.npz` per frame |
| `--device` | `cuda` | ONNX Runtime device (TensorRT always uses CUDA) |
| `--no-align-input-ext-scale` | off (alignment on) | Nested only: disable the Umeyama alignment to the input camera poses. On by default, matching `DepthAnything3.inference(align_to_input_ext_scale=True)`. |

Whether camera pose is actually fed is read from the loaded model's inputs, so
`--use-extrinsics` mainly picks the default checkpoint. With `--use-extrinsics`
the any-view defaults become the `-with-camera-pose` variants:
`weights/da3_anyview_n3_644x490_giant-large-1.1-with-camera-pose.onnx` /
`.engine`.

```bash
# Nested pipeline, ONNX (plain any-view, poses predicted)
python tools/infer_da3_onnx_trt.py --module nested --camera-set set1 --frame 0

# Any-view branch, TensorRT
python tools/infer_da3_onnx_trt.py --backend trt --module anyview --frame 0

# Any-view branch with camera-pose priors (selects the -with-camera-pose model)
python tools/infer_da3_onnx_trt.py --module anyview --use-extrinsics --frame 0
```

---

## Validation / comparison

### `compare_onnx_pt.py`

Compare PyTorch vs ONNX outputs across the **metric**, **any-view**, and
**nested** modules on a selectable Astribot camera set / frame. The ONNX side
uses the shared wrapper classes (`DA3MetricONNX`, `DA3AnyViewONNX`,
`DA3NestedONNX`); the PyTorch side runs the matching sub-model of a single nested
checkpoint. Both sides consume identical letterbox-preprocessed inputs. Reports
per-output `max`/`median` absolute / relative error and `allclose` checks. Runs
on CUDA; the giant PyTorch model and the ONNX session never co-reside on the GPU
(each module runs ONNX first, frees it, then the PyTorch forward), avoiding the
any-view attention-Softmax OOM.

Whether the any-view branch uses camera pose is read from the loaded ONNX graph
and applied to the PyTorch reference too, so the two sides always match.
`--use-extrinsics` selects the `-with-camera-pose` any-view checkpoint by default.

| Argument | Default | Description |
|---|---|---|
| `--pt-ckpt` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | PyTorch nested checkpoint (HuggingFace id or local dir); used for all modules |
| `--use-extrinsics` | off | Compare the `-with-camera-pose` any-view model (default `--onnx-anyview` path) |
| `--onnx-anyview` | `--use-extrinsics` variant | Any-view ONNX path |
| `--onnx-metric` | `weights/da3_metric_644x490_giant-large-1.1.onnx` | Metric ONNX path |
| `--modules` | `metric anyview nested` | Subset of modules to compare |
| `--camera-set` | `set1` | Astribot camera set (`set0`/`set1`/`set2`; view count must match the any-view export) |
| `--frame-idx` | `0` | Frame index to load (0-based) |

```bash
# Default (plain any-view; poses predicted)
python tools/compare_onnx_pt.py --camera-set set1 --frame-idx 0

# With camera pose (selects the -with-camera-pose any-view model)
python tools/compare_onnx_pt.py --use-extrinsics --camera-set set1 --frame-idx 0
```

### `compare_onnx_trt.py`

Compare ONNX vs TensorRT outputs across the **metric**, **any-view**, and
**nested** modules on a selectable Astribot camera set / frame. Every module
reuses the shared wrapper classes (`DA3Metric*`, `DA3AnyView*`, `DA3Nested*`) so
the two backends differ only in the inference engine. Reports per-output
`max`/`median` absolute and relative error (median is the meaningful fp16 parity
signal — `max` is inflated by fp16 tail noise) plus `allclose` checks. Modules
run cheapest-first with GPU memory freed between them.

Both backends use the shared wrappers, which feed camera pose only if the loaded
model declares those inputs. `--use-extrinsics` selects the `-with-camera-pose`
any-view ONNX and engine by default.

| Argument | Default | Description |
|---|---|---|
| `--use-extrinsics` | off | Compare the `-with-camera-pose` any-view model (default `--onnx-anyview`/`--trt-anyview` paths) |
| `--onnx-anyview` | `--use-extrinsics` variant | Any-view ONNX path |
| `--trt-anyview` | `--use-extrinsics` variant | Any-view TensorRT engine path |
| `--onnx-metric` | `weights/da3_metric_644x490_giant-large-1.1.onnx` | Metric ONNX path |
| `--trt-metric` | `weights/da3_metric_644x490_giant-large-1.1.engine` | Metric TensorRT engine path |
| `--modules` | `metric anyview nested` | Subset of modules to compare |
| `--camera-set` | `set1` | Astribot camera set (view count must match the any-view export) |
| `--frame-idx` | `0` | Frame index to load (0-based) |

Runs on CUDA. The metric module derives its focal as `(fx + fy) / 2` from the
selected view's real camera intrinsics.

```bash
# Default (plain any-view)
python tools/compare_onnx_trt.py --camera-set set1 --frame-idx 0

# With camera pose (selects the -with-camera-pose any-view ONNX + engine)
python tools/compare_onnx_trt.py --use-extrinsics --camera-set set1 --frame-idx 0
```
