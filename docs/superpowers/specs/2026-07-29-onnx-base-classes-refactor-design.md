# ONNX base-class refactor — design

**Date:** 2026-07-29
**Scope:** `tools/` — refactor duplicated ONNX inference code behind two new base
classes, add three concrete ONNX model classes, and migrate the main nested
ONNX inference script onto them.

## Problem

The ONNX inference logic in `tools/` is copy-pasted across several scripts and
partially duplicated against the TensorRT wrapper classes. An exploration pass
found:

- **ImageNet preprocessing** (RGB→resize→normalize→CHW + intrinsics scaling):
  4 near-identical copies (`da3anyview.py`, `infer_onnx_nested.py`,
  `compare_nested_onnx_pt.py`, `compare_onnx_trt.py`), plus a 5th PIL-based
  variant in `infer_onnx_metric_depth.py`.
- **Extrinsics normalization** (first camera to origin, median distance = 1,
  clamp 1e-1): 5 copies (4 numpy — two byte-identical — + 1 torch).
- **ONNX session creation** (provider selection + reading H/W/N from
  `get_inputs()[0].shape`): ~4-5 slightly varying copies.
- **Output-name mapping** (`_map_*_keys`, `_extract_metric*`): 3 identical
  copies each, plus keyword-search variants.
- **Align → squeeze-batch → mask** post-processing: duplicated between
  `da3nested.py` and `infer_onnx_nested.py`.
- `_MEAN`/`_STD` constants redeclared 7 times.

## Goals

1. One canonical home for ONNX-session mechanics and for DA3 pre/post-processing.
2. Three concrete ONNX model classes that read like their TRT siblings.
3. Migrate `infer_onnx_nested.py` onto the new classes with identical output.

## Non-goals (explicitly deferred)

- **Not** touching the TRT classes (`base_trt.py`, `da3anyview.py`,
  `da3metric.py`, `da3nested.py`). `base_da3` is built backend-agnostic so TRT
  can adopt it in a **later** pass, but no TRT file changes in this pass.
- **Not** migrating `compare_nested_onnx_pt.py`, `compare_onnx_trt.py`, or
  `infer_onnx_metric_depth.py`. They keep their current copies for now.
- **Not** moving the alignment function bodies. `alignment.py` stays as-is
  (TRT's `da3nested.py` imports it); `base_da3` calls into it.

## Decisions (from brainstorming)

- **Refactor scope:** "ONNX now, TRT later."
- **Class + script scope:** build 3 ONNX classes; migrate only
  `infer_onnx_nested.py`.

## Architecture

Two-axis split:

- `base_onnx.ONNXModel` — *how we run a model* (ONNX Runtime session). Sibling of
  `base_trt.TRTModel`.
- `base_da3.BaseDA3Model` — *what a DA3 model does around the graph*
  (preprocess, normalize extrinsics, map outputs, align). Backend-agnostic
  mixin, no session code, no `__init__`.

A concrete class multiple-inherits both:
`class DA3AnyViewONNX(ONNXModel, BaseDA3Model)`. `ONNXModel.__init__` sets
`target_h`/`target_w`/`num_views`; `BaseDA3Model` methods consume those
attributes.

### File layout

```
tools/model/
  base_onnx.py         NEW  ONNXModel
  base_da3.py          NEW  BaseDA3Model (mixin)
  da3anyview_onnx.py   NEW  DA3AnyViewONNX(ONNXModel, BaseDA3Model)
  da3metric_onnx.py    NEW  DA3MetricONNX(ONNXModel, BaseDA3Model)
  da3nested_onnx.py    NEW  DA3NestedONNX(BaseDA3Model)  # composes the two above
  alignment.py         KEEP unchanged
  base_trt.py, da3anyview.py, da3metric.py, da3nested.py   UNTOUCHED
tools/
  infer_onnx_nested.py MIGRATE to a thin driver over DA3NestedONNX
```

## Component: `base_onnx.py` (`ONNXModel`)

```python
class ONNXModel:
    def __init__(self, onnx_path: str, device: str = "cuda"):
        providers = ([("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.inputs  = [{"name": i.name, "shape": i.shape} for i in self.session.get_inputs()]
        self.outputs = [{"name": o.name, "shape": o.shape} for o in self.session.get_outputs()]
        self._resolve_input_geometry()   # sets target_h, target_w, num_views

    @property
    def input_width(self):  return self.target_w
    @property
    def input_height(self): return self.target_h

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        names = [o["name"] for o in self.outputs]
        return dict(zip(names, self.session.run(names, feed)))
```

- `_resolve_input_geometry()`: for a 5-D input `(1,N,3,H,W)` set
  `num_views=shape[1]`, `target_h=shape[3]`, `target_w=shape[4]`; for 4-D
  `(1,3,H,W)` set `num_views=None`, `target_h=shape[2]`, `target_w=shape[3]`.
  Non-static (symbolic/str) dims → `None`.
- Prints input/output name+shape on load, like `TRTModel`.

## Component: `base_da3.py` (`BaseDA3Model`)

Mixin; relies on `self.target_h`, `self.target_w`.

```python
class BaseDA3Model:
    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], np.float32)

    def preprocess_views(self, imgs, intrs, target_h=None, target_w=None):
        """N imgs (paths or BGR arrays) → (1,N,3,H,W) float32, scaled (N,3,3) intrs, metas."""
    def _preprocess_one(self, img, K, target_h, target_w):
        """RGB→resize→ImageNet-norm→CHW; scale fx,fy,cx,cy; return (chw, K_adj, meta)."""

    @staticmethod
    def normalize_extrinsics(extrs):     # (N,4,4) → (N,4,4)

    @staticmethod
    def map_anyview_keys(raw):           # → {depth, depth_conf, extrinsics, intrinsics}
    @staticmethod
    def extract_metric(raw):             # → (depth, sky)

    def align_with_metric(self, av, metric_depths, metric_skys):
        """numpy→torch, align_anyview_with_metric, squeeze (1,N,…)→(N,…), → numpy dict."""
    def align_to_input(self, result, input_extrinsics, input_intrinsics):
        """align_to_input_ext_scale on (N,…) numpy arrays."""
```

- `preprocess_views` accepts optional `target_h/target_w` overrides so the
  nested pipeline can preprocess the metric branch at the metric model's size
  (which may differ from the any-view size). Default `None` → `self.target_*`.
- `align_with_metric` folds in the numpy→torch→align→squeeze boilerplate
  duplicated between `da3nested.py` and `infer_onnx_nested.py`.
- `align_with_metric` / `align_to_input` are thin wrappers over the existing
  `alignment.align_anyview_with_metric` / `alignment.align_to_input_ext_scale`.

## Component: concrete ONNX classes

```python
# da3anyview_onnx.py
class DA3AnyViewONNX(ONNXModel, BaseDA3Model):
    def infer(self, imgs, extrs, intrs, *, normalize_extrinsics=False):
        img_batch, intrs_adj, _ = self.preprocess_views(imgs, intrs)
        if normalize_extrinsics:
            extrs = self.normalize_extrinsics(extrs)
        raw = self.run({"image": img_batch,
                        "extrinsics": extrs[None], "intrinsics": intrs_adj[None]})
        return self.map_anyview_keys(raw)

# da3metric_onnx.py
class DA3MetricONNX(ONNXModel, BaseDA3Model):
    def infer_view(self, img):            # single preprocessed CHW view → (depth, sky)
        return self.extract_metric(self.run({"image": img[None]}))

# da3nested_onnx.py  — mirrors TRT DA3NestedModel (composition, not a session itself)
class DA3NestedONNX(BaseDA3Model):
    def __init__(self, anyview_path, metric_path, device="cuda"):
        self.av = DA3AnyViewONNX(anyview_path, device)
        self.metric = DA3MetricONNX(metric_path, device)
        self.target_h, self.target_w = self.av.target_h, self.av.target_w

    def infer(self, imgs, extrs, intrs, *, align_input_ext_scale=True):
        # view-count guard: len(imgs) must == self.av.num_views
        img_batch, intrs_adj, _ = self.av.preprocess_views(imgs, intrs)
        extrs_norm = self.normalize_extrinsics(extrs)
        av = self.av.map_anyview_keys(self.av.run(
            {"image": img_batch, "extrinsics": extrs_norm[None], "intrinsics": intrs_adj[None]}))
        metric_depths, metric_skys = self._run_metric_branch(img_batch)   # (1,N,H,W)
        result = self.align_with_metric(av, metric_depths, metric_skys)
        if align_input_ext_scale:
            result = self.align_to_input(result, extrs, intrs_adj)
        return result   # {depth, depth_conf, extrinsics, intrinsics}
```

- `_run_metric_branch(img_batch)`: per-view loop that resizes each
  already-preprocessed view to the metric model's H/W when it differs, runs
  `DA3MetricONNX.infer_view`, and resizes depth/sky back to the any-view H/W —
  the logic currently inline in `NestedONNXInference.infer`.

## Data flow (nested)

```
images/exts/ixts
  → av.preprocess_views          → (1,N,3,H,W), scaled intrs
  → normalize_extrinsics(exts)   → (N,4,4)
  → av.run + map_anyview_keys    → {depth, depth_conf, extrinsics, intrinsics}
  → _run_metric_branch           → metric depth/sky (1,N,H,W)
  → align_with_metric            → aligned {depth,…} squeezed to (N,…)
  → align_to_input (optional)    → depth rescaled + input extrinsics
  → result.npz
```

## Migration: `infer_onnx_nested.py`

- **Remove**: `NestedONNXInference`, `preprocess_views`,
  `_normalize_extrinsics_numpy`, `_map_av_keys`, `_extract_metric`.
- **Keep**: arg parsing (`--camera-set`, `--frame`/`--all-frames`,
  `--onnx-anyview`, `--onnx-metric`, `--export-dir`, `--device`,
  `--no-align-input-ext-scale`), the astribot data loop, `result.npz` saving.
- **New body**: build `DA3NestedONNX(...)` once; per frame call
  `nested.infer(images, exts, ixts, align_input_ext_scale=args.align_input_ext_scale)`.
- Output keys and `result.npz` layout unchanged.

## Error handling

- `ONNXModel`: raise a clear error when input geometry is non-static (symbolic
  H/W), rather than silently defaulting.
- `DA3NestedONNX.infer`: fail early if `len(imgs) != self.av.num_views`, with a
  message naming the expected view count (as in the guard recently added to
  `compare_nested_onnx_pt.py`).

## Verification (manual — repo has no test suite)

1. Run `infer_onnx_nested.py --camera-set set1 --frame 0` before and after the
   migration; diff the saved `depth/depth_conf/extrinsics/intrinsics` in
   `result.npz` — must match within float tolerance.
2. Import smoke-test the 3 new classes.
3. Confirm `compare_nested_onnx_pt.py` still runs (unchanged).

## Risks

- Multiple-inheritance MRO: `BaseDA3Model` has no `__init__`, so
  `ONNXModel.__init__` runs; low risk. Concrete classes must set `target_*`
  before any `BaseDA3Model` method is called (guaranteed by `ONNXModel.__init__`;
  `DA3NestedONNX` sets them explicitly in its own `__init__`).
- Preprocessing parity: the new `preprocess_views` must reproduce the exact
  cv2 resize + normalization used today (INTER_LINEAR, `/255`, ImageNet
  mean/std) so numeric outputs match — covered by verification step 1.
```
