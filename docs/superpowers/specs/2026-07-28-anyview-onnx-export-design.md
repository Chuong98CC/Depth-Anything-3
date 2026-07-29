# Any-View ONNX Export from Nested Checkpoint

**Date:** 2026-07-28
**Context:** Export the any-view sub-model from `DA3NESTED-GIANT-LARGE` to ONNX,
keeping the alignment logic in Python post-processing.

## Motivation

`NestedDepthAnything3Net` combines two independent sub-models:
- `self.da3` — any-view giant (DualDPT head, camera encoder/decoder)
- `self.da3_metric` — metric large (DPT head with sky)

The metric model is already exportable via `DepthAnything3OnnxWrapper`. We need
to export the any-view branch and replicate the alignment steps in Python.

## Design

### 1. `DepthAnything3AnyViewOnnxWrapper` (new class in `tools/export_onnx.py`)

Wraps only `api_model.model.da3` — no autocast, no nested alignment.

- **Inputs:** `image (B,N,3,H,W)`, `extrinsics (B,N,4,4)`, `intrinsics (B,N,3,3)`
- **Outputs:** `depth (B,N,H,W)`, `depth_conf (B,N,H,W)`, `extrinsics (B,N,4,4)`, `intrinsics (B,N,3,3)`
- `N` is fixed at export time via `--num-views`.

### 2. `align_anyview_with_metric()` (standalone function)

Python-only post-processing that replicates `NestedDepthAnything3Net`:
1. `_apply_metric_scaling` — scale metric depth by `focal / 300.0`
2. `_apply_depth_alignment` — least-squares scale factor
3. `_handle_sky_regions` — set sky pixels to max depth

### 3. CLI changes

| Arg | Detail |
|---|---|
| `--wrapper` | `"metric"` (default) or `"anyview"` (new) |
| `--num-views` | Fixed N, default 1 |

## Why Python for alignment

The alignment logic uses `torch.randint`, `assert` statements, and data-dependent
branching — all incompatible with `torch.export` / ONNX tracing. Keeping it in
Python avoids these issues without modifying the upstream model code.

## Outputs

- Aligned metric depth
- Predicted camera extrinsics and intrinsics
- Depth confidence
