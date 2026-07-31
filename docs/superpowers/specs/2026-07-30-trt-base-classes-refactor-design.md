# TRT base-class refactor — design

**Date:** 2026-07-30
**Scope:** `tools/model/` — rework the three TensorRT DA3 wrapper classes to inherit
`TRTModel` (base_trt.py) + `BaseDA3Model` (base_da3.py), mirroring the ONNX side,
and generalize `base_trt.py` so it stays shared across monocular / stereo /
any-view engines.

## Problem

The ONNX refactor (2026-07-29) extracted shared DA3 pre/post-processing into
`BaseDA3Model` and migrated the ONNX classes onto it. The TensorRT classes were
left untouched and still duplicate that logic:

- `DA3AnyViewTRT(TRTModel)` — its own **exact-resize** `preprocess`/
  `_preprocess_one`, `_normalize_extrinsics`, `parse_outputs` key-finding,
  `_MEAN`/`_STD`, and a multi-input `_infer`.
- `DA3MetricTRT(MonoDepthTRT)` — its own normalization, focal metric scaling,
  crop, `_MEAN`/`_STD`.
- `DA3NestedTRT` (plain object) — its own `_map_anyview_keys`,
  `_extract_metric`.

`base_trt.TRTModel` also hardcodes 4-D NCHW geometry
(`self.inputs[0]['shape'][2],[3]`), which `DA3AnyViewTRT.__init__` has to
monkey-patch for its 5-D `(1,N,3,H,W)` input, and the TRT execute mechanics
(set-shape / set-address / allocate / `execute_async_v3`) are copy-pasted across
`MonoDepthTRT._infer`, `StereoDepthTRT._infer`, and `DA3AnyViewTRT._infer`.

## Goals

1. The three DA3 TRT classes inherit `TRTModel` + `BaseDA3Model` and delegate all
   shared pre/post to the bases — method-for-method mirror of the ONNX classes.
2. `base_trt.py` becomes general (image-count-agnostic): 4-D + 5-D geometry and a
   single name-matched execution helper shared by mono/stereo/anyview.
3. Verified by TRT-vs-ONNX parity (engines present in `weights/`).

## Non-goals

- No change to `base_da3.py` (already the shared mixin) or the ONNX classes.
- `MonoDepthTRT` / `StereoDepthTRT` keep their public `_infer` signatures and
  behavior — they stay as the general non-DA3 examples.
- `data_structure.CameraIntrinsics` stays in the tree; it simply stops being
  imported by the reworked DA3 classes.

## Decisions (from brainstorming)

- **File layout:** rework `da3anyview.py`, `da3metric.py`, `da3nested.py` in
  place; keep class names (`DA3AnyViewTRT`, `DA3MetricTRT`, `DA3NestedTRT`)
  and import paths.
- **base_trt generality:** generalize BOTH (a) 4-D/5-D geometry and (b) a general
  `_run(named_inputs)` execution helper.
- **Letterbox:** `base_trt.resize_img` (general single-image uint8 letterbox for
  mono/stereo) and `base_da3` letterbox (DA3 multi-view normalized-float) both
  stay. `base_da3` is backend-agnostic and must not call a TRT method, so the
  ~15-line geometry overlap is inherent to the two-base split and accepted (no
  base_trt→base_da3 coupling).
- **Metric:** `DA3MetricTRT` mirrors `DA3MetricONNX` exactly — `infer_view`
  raw `(depth, sky)` only; the `CameraIntrinsics` focal-scaling standalone path
  is dropped (metric depth in metres is a caller-side `focal*depth/300` step, per
  README and the ONNX metric decision).
- **Verification:** TRT-vs-ONNX parity, fp16 tolerances.

## Architecture

```
tools/model/
  base_trt.py       GENERALIZED  TRTModel: _resolve_input_geometry (4-D+5-D),
                                 general _run(named_inputs); Mono/Stereo unchanged behavior
  base_da3.py       UNCHANGED    shared DA3 mixin
  da3anyview.py     REWORK  DA3AnyViewTRT(TRTModel, BaseDA3Model)
  da3metric.py      REWORK  DA3MetricTRT(TRTModel, BaseDA3Model)   # was MonoDepthTRT
  da3nested.py      REWORK  DA3NestedTRT(BaseDA3Model)  — composes the two
  da3*_onnx.py      UNCHANGED (parity reference)
  data_structure.py UNCHANGED (no longer imported by DA3 classes)
tools/
  compare_onnx_trt.py  UPDATE the DA3AnyViewTRT._infer call site
```

Each DA3 TRT class multiple-inherits `TRTModel` (engine mechanics) + `BaseDA3Model`
(letterbox `preprocess_views`, `crop_to_tile`, `normalize_extrinsics`,
`map_anyview_keys`, `extract_metric`, `align_with_metric`, `align_to_input`),
exactly as `DA3AnyViewONNX(ONNXModel, BaseDA3Model)` does on the ONNX side.

## Component: `base_trt.py` (`TRTModel`)

**(a) Geometry.** Replace the hardcoded `target_w/target_h` with
`_resolve_input_geometry()` mirroring `ONNXModel`:
- 5-D `(1,N,3,H,W)`: `num_views=shape[1]`, `target_h=shape[3]`, `target_w=shape[4]`.
- 4-D `(1,3,H,W)`: `num_views=None`, `target_h=shape[2]`, `target_w=shape[3]`.
- Non-int (dynamic) H/W → raise (preprocessing needs concrete sizes).

This removes the `DA3AnyViewTRT.__init__` monkey-patch.

**(b) General execution.** Add:
```python
def _run(self, named_inputs: dict[str, np.ndarray | torch.Tensor],
         np_output: bool = True) -> dict:
    # for each engine input (self.inputs, matched BY NAME):
    #   tensor = self._to_input_tensor(named_inputs[name], dtype=input dtype)
    #   set_input_shape(name, tensor.shape); set_tensor_address(name, ptr)
    # for each output: allocate cuda tensor, set_tensor_address
    # execute_async_v3(current stream); synchronize
    # return output2numpy(outs) if np_output else outs
```
`_to_input_tensor(array, dtype)` moves an arbitrary numpy/torch array to a
contiguous CUDA tensor of the engine's expected dtype (generalizes the image-only
`img2tensor`, which stays for the mono/stereo image path).

`MonoDepthTRT._infer(img)` / `StereoDepthTRT._infer(left, right)` keep their
signatures; bodies become: build the `{name: array}` dict (single image; or
`left`/`right` or concatenated per the existing 2-input vs 6-channel logic) and
call `_run`. Behavior unchanged.

## Component: `da3anyview.py` (`DA3AnyViewTRT`)

`class DA3AnyViewTRT(TRTModel, BaseDA3Model)`. Mirrors `DA3AnyViewONNX`:

```python
def __init__(self, engine_path, conf_thresh=0.5):
    super().__init__(engine_path)     # geometry now resolved in base (5-D aware)
    self.conf_thresh = conf_thresh

def infer(self, imgs, extrs, intrs, *, normalize_extrinsics=False):
    img_batch, intrs_adj, _ = self.preprocess_views(imgs, intrs)
    ext = self.normalize_extrinsics(extrs) if normalize_extrinsics else extrs
    raw = self._run({"image": img_batch.astype(np.float32),
                     "extrinsics": ext[None].astype(np.float32),
                     "intrinsics": intrs_adj[None].astype(np.float32)})
    return self.map_anyview_keys(raw)
```

Returns the mapped **raw** dict — **no** confidence masking (drops the old
`depth[~valid] = -1`; masking is TRT-only, absent from PyTorch/ONNX). Deletes the
old `preprocess`/`_preprocess_one`/`_normalize_extrinsics`/`parse_outputs`/
`_infer`/`_MEAN`/`_STD`.

## Component: `da3metric.py` (`DA3MetricTRT`)

`class DA3MetricTRT(TRTModel, BaseDA3Model)` (was `MonoDepthTRT`). Mirrors
`DA3MetricONNX`:

```python
def infer_view(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """img is a preprocessed (3, H, W) CHW array → (depth, sky)."""
    raw = self._run({"image": img[None].astype(np.float32)})
    return self.extract_metric(raw)
```

Drops `CameraIntrinsics`, focal scaling, crop, masking, `preprocess`,
`parse_outputs`, `_MEAN`/`_STD`. Metric depth in metres is caller-side.

## Component: `da3nested.py` (`DA3NestedTRT`)

`class DA3NestedTRT(BaseDA3Model)`. Mirrors `DA3NestedONNX` exactly:

```python
def __init__(self, anyview_engine, metric_engine, conf_thresh=0.5):
    self.av = DA3AnyViewTRT(anyview_engine)
    self.metric = DA3MetricTRT(metric_engine)
    self.target_h, self.target_w = self.av.target_h, self.av.target_w

def infer(self, imgs, extrs, intrs, *, align_input_ext_scale=True):
    # view-count guard (n != self.av.num_views -> ValueError, before inference)
    # 1 av.preprocess_views -> img_batch, intrs_adj, metas
    # 2 extrs_norm = normalize_extrinsics(extrs)
    # 3 av = map_anyview_keys(av._run({image, extrinsics, intrinsics}))
    # 4 metric_depths, metric_skys = _run_metric_branch(img_batch)
    # 5 result = align_with_metric(av, metric_depths, metric_skys)
    # 6 if align_input_ext_scale: result = align_to_input(result, extrs, intrs_adj)
    # 7 return _crop_result(result, metas)
```

`_run_metric_branch(av_img_batch)`: requires metric target == any-view target
(fail-fast `NotImplementedError` otherwise, exactly like the ONNX nested's final
form); per-view `self.metric.infer_view(av_img_batch[0, i])`; returns
`(1, N, H, W)`. `_crop_result` crops depth/conf to the tile and un-pads the
intrinsics principal point. Signature changes: `__init__` no longer takes
`metric_intrinsics: CameraIntrinsics`.

## Data flow (nested)

Identical to the ONNX nested pipeline; only `av._run` / `metric._run` use TRT
engines instead of ONNX sessions.

## Verification (TRT-vs-ONNX parity, fp16)

Assets present: `weights/da3_anyview_n3_644x490_giant-large-1.1.{onnx,engine}` and
`weights/da3_metric_644x490_giant-large-1.1.{onnx,engine}`; tensorrt 10.16
importable.

1. **Any-view:** `DA3AnyViewTRT(engine).infer(...)` vs
   `DA3AnyViewONNX(onnx).infer(...)` on the SAME letterbox-preprocessed set1/frame0
   input; compare `depth/depth_conf/extrinsics/intrinsics`.
2. **Metric:** `DA3MetricTRT(engine).infer_view(chw)` vs
   `DA3MetricONNX(onnx).infer_view(chw)` on the same preprocessed view.
3. **Nested:** `DA3NestedTRT(both engines).infer(...)` vs
   `DA3NestedONNX(both onnx).infer(...)`.

**fp16 tolerances** (engines are fp16; looser than ONNX fp32). Gate on relative
medians + shape + no-NaN, report max stats without hard-failing on small tails
(as done for ONNX intrinsics):
- depth / depth_conf / sky: median relative error < 2e-2.
- extrinsics: max abs < 5e-2.
- intrinsics: max relative < 1e-2.
Thresholds are pragmatic fp16 bounds; if a key blows past them, STOP and
investigate rather than loosen silently.

To avoid GPU co-residency OOM (TRT engine arena + ONNX CUDA arena + any torch),
run the TRT and ONNX halves in **separate processes** (process A saves inputs +
TRT outputs to an npz; process B loads it, runs ONNX, compares) — the pattern
used for the ONNX PyTorch-parity check.

## Migration impact

- `tools/compare_onnx_trt.py` calls `DA3AnyViewTRT._infer(img_batch, extrs,
  intrs)` (old multi-input signature). After the rework `_infer` is gone; update
  the call to `model._run({"image":..., "extrinsics":..., "intrinsics":...})` or
  to `model.infer(...)`, whichever matches its intent (it feeds an
  already-preprocessed batch, so `_run` is the closer fit).
- `da3nested.py` / `da3metric.py` stop importing `CameraIntrinsics`. Leave
  `data_structure.py` in place.
- No other importer relies on the old masked/scaled outputs (the metric-scaling
  consumer `infer_depth.py` was already removed).

## Risks

- **fp16 accuracy:** TRT fp16 vs ONNX fp32 gaps are larger than the ONNX-vs-PyTorch
  bf16 gaps; tolerances above are estimates and may need one calibration pass
  against real output (report-then-adjudicate, not silent loosening).
- **`_run` dtype handling:** engine inputs may be fp16 or fp32; `_to_input_tensor`
  must cast to each input's own dtype (extrinsics/intrinsics are typically fp32
  even in an fp16 engine). Match per-input dtype from `self.inputs[i]['dtype']`.
- **Multi-inheritance MRO:** `BaseDA3Model` has no `__init__`, so
  `TRTModel.__init__` runs; `DA3NestedTRT` sets `target_*` explicitly. Same
  pattern as the ONNX side.
