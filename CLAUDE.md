# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# Install (editable + all extras)
pip install -e ".[all]"

# Lint / format
pip install pre-commit && pre-commit run --all-files
black --line-length 99 src/ tools/
isort --profile black src/ tools/
flake8 src/ tools/

# Type-check (mypy configured in pyproject.toml with jaxtyping plugin)
mypy src/

# CLI entry point
da3 --help
da3 image <path>                     # single image
da3 images <dir>                     # directory of images
da3 video <path>                     # video file
da3 colmap <dir>                     # COLMAP reconstruction
da3 backend --model-dir <...>        # FastAPI server
da3 gradio --model-dir <...>         # Gradio web UI

# ONNX export (tools/)
python tools/export_onnx.py --height 518 --width 518
python tools/export_onnx.py --demo-image <path.jpg> --demo-device cuda
python tools/infer_onnx.py --onnx-path <...> --image <...>
python tools/infer_pytorch.py --model-dir <...> --image <...>

# Extract frames from Astribot stereo recordings (tools/)
./tools/extract_frames.sh --start 1 --stop 10 --step 2
```

## Architecture

### Layering

```
CLI (cli.py) / Gradio (app/) / Backend (services/backend.py)
  └─ DepthAnything3 API (api.py) — the only public Python surface
       ├─ InputProcessor (utils/io/input_processor.py) — resize, normalize
       ├─ DepthAnything3Net / NestedDepthAnything3Net (model/da3.py)
       │    ├─ DinoV2 backbone (model/dinov2/) — custom ViT implementation
       │    ├─ DPT / DualDPT head (model/dpt.py, model/dualdpt.py)
       │    ├─ Camera encoder/decoder (model/cam_enc.py, model/cam_dec.py)
       │    └─ GS head (model/gsdpt.py, model/gs_adapter.py) — giant model only
       └─ OutputProcessor (utils/io/output_processor.py) — raw → Prediction

Configs (configs/*.yaml) — OmegaConf-based, support __inherit__ and __object__
Model registry (registry.py) — auto-discovers YAML configs → MODEL_REGISTRY
```

- **Any-view models** (da3-giant/large/base/small): `DualDPT` head → depth + ray + pose.
- **Metric/Mono models** (da3metric-large, da3mono-large): `DPT` head → depth only, 1-channel.
- **Nested models** (da3nested-giant-large): any-view giant + metric large stitched together with least-squares scale alignment.

### Adapters (utils/alignment.py)

Key post-processing utilities shared across models:
- `compute_sky_mask(sky_pred, threshold)` → `sky_pred < threshold` (non-sky = True)
- `set_sky_regions_to_max_depth(depth, depth_conf, non_sky_mask, max_depth)` — sets sky pixels to max depth
- `least_squares_scale_scalar(pred, target, mask)` — optimal scaling factor
- `apply_metric_scaling(depth, intrinsics, scale_factor)` — `focal * depth / scale_factor`

### Config system (cfg.py)

YAML configs support `__inherit__: <parent.yaml>` for cascading overrides and `__object__: {class: dotted.Path, ...}` for constructing arbitrary Python objects at init time. `create_object(cfg)` in `cfg.py` is the factory.

### Model loading from HuggingFace

Pretrained weights live at `depth-anything/<model-name>` on HuggingFace Hub. Loading uses `PyTorchModelHubMixin` from `huggingface_hub`. The `-1.1` suffix models fix a training bug — prefer them (e.g., `DA3NESTED-GIANT-LARGE-1.1`).

## Pitfalls and conventions

### torch.export / ONNX: no data-dependent control flow

`torch.onnx.export` uses `torch.export.export` internally. Python `if` statements that depend on tensor **values** (not just shapes) will fail with `GuardOnDataDependentSymNode`. Use `torch.where`-based selection instead of `if` branching, and avoid `torch.randint` (whose upper bound becomes symbolic). See the fix in `model/da3.py:_process_mono_sky_estimation` for the pattern.

### Autocast in the API forward

`DepthAnything3.forward()` (`api.py:128`) wraps the model call in `torch.autocast(dtype=bfloat16)` when on CUDA. When exporting to ONNX, bypass this by calling `api_model.model(...)` (the underlying `DepthAnything3Net`) directly — otherwise the ONNX graph captures bf16 ops that ONNX Runtime CPU cannot execute.

### No test suite

The repository has no automated tests. Any behavioral change must be verified manually by running the CLI or a `tools/` script against real inputs.

### Line length

Black is configured at 99 chars, flake8 at 100. Pre-commit runs autoflake → pyupgrade → isort → black → flake8.

### Metric depth formula

For `DA3METRIC-LARGE`: `metric_depth = focal_length_px * net_output / 300.0`. The default focal in `export_onnx.py` is 400.88 px (scaled by resize factor at inference time).

### Tools directory conventions

Scripts in `tools/` are standalone — they import from the installed `depth_anything_3` package, not via relative imports. Always run them from the repo root. The `extract_frames.sh` script operates on `data/astribot_stereo_lrb/` by default.
