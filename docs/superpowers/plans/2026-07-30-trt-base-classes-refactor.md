# TRT Base-Class Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the three TensorRT DA3 wrapper classes to inherit `TRTModel` + `BaseDA3Model` (mirroring the ONNX classes), and generalize `base_trt.py` so it stays shared across monocular / stereo / any-view engines.

**Architecture:** `base_trt.TRTModel` gains 4-D+5-D geometry resolution and one name-matched `_run(named_inputs)` execution helper (replacing three copy-pasted `_infer` bodies). The three DA3 TRT classes (`DA3AnyViewTRT`, `DA3MetricTRT`, `DA3NestedTRT`) are reworked in place to multiple-inherit `TRTModel` + `BaseDA3Model`, deleting their duplicated preprocessing / extrinsics-norm / key-mapping / masking / focal-scaling, exactly mirroring `DA3AnyViewONNX` / `DA3MetricONNX` / `DA3NestedONNX`.

**Tech Stack:** Python 3.10, numpy, opencv (cv2), torch, tensorrt 10.16. Run from repo root; `tools/` is on `sys.path[0]` so imports are `from model.X import Y` (no `tools.` prefix).

## Global Constraints

- **Run from repo root** `/home/chuong/workspace/depth_models/Depth-Anything-3`. Bare imports (`from model.X import Y`), never `tools.` prefix.
- **No pytest suite exists.** Verification is manual: import smoke-tests + TRT-vs-ONNX parity checks against the real engines/ONNX in `weights/`. Put throwaway scripts / npz in the scratchpad `/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad`, never in the repo.
- **Black line length 99; flake8 100.** Run `black --line-length 99` and confirm `black --check` + `flake8 --max-line-length 100` before every commit.
- **Never `git add`** `*.onnx`, `*.onnx.data`, `*.engine`, or model weights. Commit only source files.
- **Commit trailer** exactly: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Mirror the ONNX classes** (`tools/model/da3anyview_onnx.py`, `da3metric_onnx.py`, `da3nested_onnx.py`, `base_onnx.py`) method-for-method; only the engine `_run` differs from the ONNX `run`.
- **Unmasked depth:** the reworked classes must NOT confidence-mask depth (drop the old `depth[~valid] = -1`); only sky handling inside `align_anyview_with_metric` touches depth.
- **Assets present:** `weights/da3_anyview_n3_644x490_giant-large-1.1.{onnx,engine}` (5-D input, N=3, 490×644) and `weights/da3_metric_644x490_giant-large-1.1.{onnx,engine}` (4-D input). Astribot data at `/home/chuong/workspace/demo_data/astribot_stereo_lrb`. `tensorrt` 10.16 imports.
- **fp16 tolerances are provisional** — if a parity gate is exceeded, STOP and report the actual numbers for adjudication; do NOT loosen silently.

---

### Task 1: generalize `base_trt.py`

**Files:**
- Modify: `tools/model/base_trt.py`

**Interfaces:**
- Consumes: nothing.
- Produces on `TRTModel`:
  - `self.num_views: int|None`, `self.target_h: int`, `self.target_w: int` (via `_resolve_input_geometry`)
  - `_to_input_tensor(array, trt_dtype) -> torch.Tensor` (CUDA, engine dtype)
  - `_run(named_inputs: dict[str, np.ndarray | torch.Tensor], np_output: bool = True) -> dict` — matches engine inputs BY NAME.
  - `_infer` is no longer abstract (subclasses may omit it).
- `MonoDepthTRT._infer(img)`, `StereoDepthTRT._infer(left, right)`, `StereoDepthTRT.preprocess` keep their signatures.

- [ ] **Step 1: Rewrite the file**

Replace the entire contents of `tools/model/base_trt.py` with:

```python
import numpy as np
import cv2
import tensorrt as trt
import torch
from abc import ABC


# Convert numpy dtype name to torch dtype
def trt_to_torch_dtype(trt_dtype):
    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.UINT8: torch.uint8,
    }
    return dtype_map.get(trt_dtype, torch.float32)


class TRTModel(ABC):
    """TensorRT engine wrapper.

    Owns engine load, IO-tensor metadata, input-geometry resolution (4-D NCHW
    for mono/stereo, 5-D (1,N,3,H,W) for any-view), and a single name-matched
    execution helper ``_run``.  Kept image-count-agnostic so monocular, stereo,
    and any-view models all share it.  DA3-specific pre/post lives in
    ``BaseDA3Model``.
    """

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.context = None
        self.engine = None

        print(f"Loading engine from: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {engine_path}.\n"
                f"The engine is incompatible with TensorRT {trt.__version__}.\n"
                f"Rebuild the engine with your current TensorRT version."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Store tensor info for TensorRT 10.3+ (no manual memory allocation).
        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_info = {"name": tensor_name, "shape": shape, "dtype": dtype}
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
                print(f"Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
            else:
                self.outputs.append(tensor_info)
                print(f"Output: {tensor_name}, shape: {shape}, dtype: {dtype}")

        self._resolve_input_geometry()

    def _resolve_input_geometry(self) -> None:
        """Resolve target H/W (and num_views) from the first input shape.

        Supports 5-D any-view ``(1, N, 3, H, W)`` and 4-D ``(1, 3, H, W)``.
        Dynamic dims (TRT reports ``-1``) are treated as unresolved.
        """
        shape = self.inputs[0]["shape"]

        def _dim(idx: int) -> int | None:
            if idx >= len(shape):
                return None
            v = int(shape[idx])
            return v if v > 0 else None

        if len(shape) == 5:  # (B, N, 3, H, W) — any-view
            self.num_views = _dim(1)
            self.target_h = _dim(3)
            self.target_w = _dim(4)
        elif len(shape) == 4:  # (B, 3, H, W) — mono / stereo / metric
            self.num_views = None
            self.target_h = _dim(2)
            self.target_w = _dim(3)
        else:
            raise ValueError(
                f"Unexpected input rank {len(shape)} for "
                f"{self.inputs[0]['name']}: {shape}"
            )

        if self.target_h is None or self.target_w is None:
            raise ValueError(
                f"Engine input '{self.inputs[0]['name']}' has non-static H/W "
                f"{shape}; rebuild with fixed height/width."
            )

    @property
    def input_width(self):
        return self.target_w

    @property
    def input_height(self):
        return self.target_h

    def resize_img(self, img: np.ndarray):
        """General single-image letterbox (aspect-preserve + center-pad).

        Returns a padded uint8 image and meta.  Used by the non-DA3 mono/stereo
        paths; the DA3 classes use ``BaseDA3Model.preprocess_views`` instead.
        """
        orig_h, orig_w = img.shape[:2]
        scale_w = self.target_w / orig_w
        scale_h = self.target_h / orig_h
        raw_scale = min(scale_w, scale_h)
        scale_factor = np.floor(raw_scale * 100.0) / 100.0
        if scale_factor <= 0:
            scale_factor = raw_scale

        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h))

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_factor": float(scale_factor),
            "tile_h": new_h,
            "tile_w": new_w,
            "pad_top": int(pad_top),
            "pad_left": int(pad_left),
        }
        return img_padded, meta

    def img2tensor(self, img: np.ndarray):
        """HWC image → (1, 3, H, W) CUDA tensor at the first input's dtype."""
        input_dtype = trt_to_torch_dtype(self.inputs[0]["dtype"])
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).contiguous()
        return tensor.cuda().to(input_dtype)

    def _to_input_tensor(self, array, trt_dtype) -> torch.Tensor:
        """Move an already model-shaped numpy/torch array to a contiguous CUDA
        tensor of the engine input's dtype (no layout change)."""
        if isinstance(array, torch.Tensor):
            t = array
        else:
            t = torch.from_numpy(np.ascontiguousarray(array))
        return t.cuda().to(trt_to_torch_dtype(trt_dtype)).contiguous()

    def _run(self, named_inputs: dict, np_output: bool = True) -> dict:
        """Generic execution: bind inputs BY NAME, allocate outputs, execute.

        ``named_inputs`` maps each engine input name to an array already in the
        engine's expected layout (e.g. ``image`` (1,N,3,H,W), ``extrinsics``
        (1,N,4,4)).  Returns ``{output_name: array}`` (numpy if ``np_output``).
        """
        input_tensors = []  # keep alive until execution completes
        for inp in self.inputs:
            name = inp["name"]
            if name not in named_inputs:
                raise KeyError(f"Missing engine input '{name}' in named_inputs")
            t = self._to_input_tensor(named_inputs[name], inp["dtype"])
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            input_tensors.append(t)

        output_tensors = {}
        for out in self.outputs:
            name = out["name"]
            shape = self.context.get_tensor_shape(name)
            dtype = trt_to_torch_dtype(out["dtype"])
            t = torch.empty(tuple(shape), dtype=dtype, device="cuda").contiguous()
            output_tensors[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference execution failed")
        torch.cuda.synchronize()

        return self.output2numpy(output_tensors) if np_output else output_tensors

    def output2numpy(self, output_tensors: dict):
        return {name: t.cpu().numpy() for name, t in output_tensors.items()}

    def __del__(self):
        if hasattr(self, "context") and self.context is not None:
            del self.context
        if hasattr(self, "engine") and self.engine is not None:
            del self.engine

    def parse_outputs(self, *args, **kwargs):
        """Extract/crop/mask outputs for a frame. Overridden by subclasses."""
        pass


class MonoDepthTRT(TRTModel):
    def _infer(self, img: np.ndarray, np_output: bool = True):
        """Single-image inference. Input: uint8 HWC RGB."""
        img_tensor = self.img2tensor(img)
        return self._run({self.inputs[0]["name"]: img_tensor}, np_output)


class StereoDepthTRT(TRTModel):
    def _infer(self, left_img: np.ndarray, right_img: np.ndarray, np_output: bool = True):
        """Stereo inference. Two-input engine or concatenated 6-channel input."""
        left_tensor = self.img2tensor(left_img)
        right_tensor = self.img2tensor(right_img)
        if len(self.inputs) == 2:
            named = {
                self.inputs[0]["name"]: left_tensor,
                self.inputs[1]["name"]: right_tensor,
            }
        else:
            named = {self.inputs[0]["name"]: torch.cat([left_tensor, right_tensor], dim=1)}
        return self._run(named, np_output)

    def preprocess(self, left_img, right_img):
        left_resized, meta_info = self.resize_img(left_img)
        right_resized, _ = self.resize_img(right_img)
        return left_resized, right_resized, meta_info
```

- [ ] **Step 2: Black + flake8**

```bash
black --line-length 99 tools/model/base_trt.py
black --check --line-length 99 tools/model/base_trt.py
flake8 --max-line-length 100 tools/model/base_trt.py
```
Expected: black "unchanged", flake8 silent.

- [ ] **Step 3: Geometry + `_run` smoke-test on real engines**

`MonoDepthTRT` is concrete; use it to exercise `TRTModel.__init__`/geometry on both engines, and `_run` on the 4-D metric engine.

```bash
python -c "
import sys; sys.path.insert(0, 'tools')
import numpy as np
from model.base_trt import MonoDepthTRT

# 5-D any-view engine: geometry resolves N/H/W (do not call _infer on it)
av = MonoDepthTRT('weights/da3_anyview_n3_644x490_giant-large-1.1.engine')
print('anyview geom:', av.num_views, av.target_h, av.target_w)
assert (av.num_views, av.target_h, av.target_w) == (3, 490, 644), (av.num_views, av.target_h, av.target_w)

# 4-D metric engine: geometry + generic _run on a dummy image via _infer
m = MonoDepthTRT('weights/da3_metric_644x490_giant-large-1.1.engine')
print('metric geom:', m.num_views, m.target_h, m.target_w)
assert m.num_views is None and (m.target_h, m.target_w) == (490, 644)
dummy = np.zeros((490, 644, 3), np.uint8)
out = m._infer(dummy)
print('metric _run outputs:', {k: v.shape for k, v in out.items()})
assert len(out) >= 1
print('OK')
"
```
Expected: any-view geom `3 490 644`, metric geom `None 490 644`, metric `_run` returns at least one output array, `OK`. (Ignore any TRT stdout noise.)

- [ ] **Step 4: Commit**

```bash
git add tools/model/base_trt.py
git commit -m "refactor(tools): generalize TRTModel (4-D/5-D geometry + name-matched _run)

Resolves input geometry for both 4-D (mono/stereo) and 5-D (any-view) inputs,
and replaces the three copy-pasted _infer bodies with one name-matched _run
execution helper. Mono/Stereo _infer keep their signatures (behaviour
unchanged). _infer is no longer abstract so the DA3 classes can call _run
directly.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: rework `da3anyview.py` (`DA3AnyViewTRT`) + fix `compare_onnx_trt.py`

**Files:**
- Modify: `tools/model/da3anyview.py` (full rewrite)
- Modify: `tools/compare_onnx_trt.py` (`run_trt` call site)

**Interfaces:**
- Consumes: `TRTModel` (Task 1: `_run`, `num_views`, `target_h/w`), `BaseDA3Model` (`preprocess_views`, `normalize_extrinsics`, `map_anyview_keys`).
- Produces: `class DA3AnyViewTRT(TRTModel, BaseDA3Model)` with `__init__(engine_path, conf_thresh=0.5)` and `infer(imgs, extrs, intrs, *, normalize_extrinsics=False) -> dict` (mapped raw `{depth, depth_conf, extrinsics, intrinsics}`, batched values).

- [ ] **Step 1: Rewrite `tools/model/da3anyview.py`**

Replace the entire file with:

```python
"""Depth Anything v3 any-view TensorRT wrapper.

Multi-view depth + confidence + predicted camera parameters.  Built on
``TRTModel`` (engine) + ``BaseDA3Model`` (pre/post) — the TRT sibling of
``DA3AnyViewONNX``.
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_trt import TRTModel


class DA3AnyViewTRT(TRTModel, BaseDA3Model):
    """Any-view TRT inference: images (+ extrinsics/intrinsics) → depth bundle."""

    def __init__(self, engine_path: str, conf_thresh: float = 0.5) -> None:
        super().__init__(engine_path)
        self.conf_thresh = conf_thresh
        print(
            f"Any-view engine: {self.target_h}x{self.target_w}, "
            f"num_views={self.num_views}"
        )

    def infer(
        self,
        imgs: list,
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        normalize_extrinsics: bool = False,
    ) -> dict:
        """Run the any-view engine.

        ``imgs`` are *N* paths or BGR arrays; ``extrs`` ``(N,4,4)``; ``intrs``
        ``(N,3,3)`` at original resolution.  Returns the mapped output dict with
        batched values (``(1,N,…)``).  Extrinsics are normalized here only when
        ``normalize_extrinsics=True`` (the graph does not normalize).
        """
        img_batch, intrs_adj, _ = self.preprocess_views(imgs, intrs)
        ext = self.normalize_extrinsics(extrs) if normalize_extrinsics else extrs
        raw = self._run({
            "image": img_batch.astype(np.float32),
            "extrinsics": ext[None].astype(np.float32),
            "intrinsics": intrs_adj[None].astype(np.float32),
        })
        return self.map_anyview_keys(raw)
```

This deletes the old `preprocess`/`_preprocess_one`, `_normalize_extrinsics`, `parse_outputs` (with its confidence masking), multi-input `_infer`, and `_MEAN`/`_STD`.

- [ ] **Step 2: Update `compare_onnx_trt.py`'s `run_trt`**

In `tools/compare_onnx_trt.py`, the `run_trt` function calls the removed `model._infer(...)`. Replace the call so it uses the generic `_run` on the already-preprocessed batch. Change:

```python
    # The TRT wrapper's _infer expects (N, ...) extrs/intrs (no batch dim)
    raw = model._infer(
        img_batch.astype(np.float32),
        extrs.astype(np.float32),
        intrs.astype(np.float32),
    )
    return raw
```

to:

```python
    # DA3AnyViewTRT now inherits the generic TRTModel._run; feed the
    # already-preprocessed batch and add the batch dim to extrs/intrs.
    raw = model._run({
        "image": img_batch.astype(np.float32),
        "extrinsics": extrs[None].astype(np.float32),
        "intrinsics": intrs[None].astype(np.float32),
    })
    return raw
```

- [ ] **Step 3: Black + flake8**

```bash
black --line-length 99 tools/model/da3anyview.py tools/compare_onnx_trt.py
black --check --line-length 99 tools/model/da3anyview.py tools/compare_onnx_trt.py
flake8 --max-line-length 100 tools/model/da3anyview.py tools/compare_onnx_trt.py
```

- [ ] **Step 4: Any-view TRT-vs-ONNX parity (two processes to avoid GPU co-residency OOM)**

`SCRATCH` = the scratchpad dir. Process A (TRT) saves outputs; Process B (ONNX) compares. Both call `.infer(..., normalize_extrinsics=True)` so preprocessing is identical (deterministic `base_da3`).

Process A:
```bash
SCRATCH=/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad
python - "$SCRATCH" <<'PY'
import sys; sys.path.insert(0, 'tools')
scratch = sys.argv[1]
import numpy as np
from astribot_dataloader import load_images_cam_params
from model.da3anyview import DA3AnyViewTRT
images, exts, ixts = load_images_cam_params('set1', 0)
av = DA3AnyViewTRT('weights/da3_anyview_n3_644x490_giant-large-1.1.engine')
out = av.infer(images, exts, ixts, normalize_extrinsics=True)
np.savez(scratch + '/trt_av.npz', **{k: np.asarray(v) for k, v in out.items()})
print('saved TRT any-view:', {k: out[k].shape for k in out})
PY
```

Process B:
```bash
SCRATCH=/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad
python - "$SCRATCH" <<'PY'
import sys; sys.path.insert(0, 'tools')
scratch = sys.argv[1]
import numpy as np
from astribot_dataloader import load_images_cam_params
from model.da3anyview_onnx import DA3AnyViewONNX
images, exts, ixts = load_images_cam_params('set1', 0)
onx = DA3AnyViewONNX('weights/da3_anyview_n3_644x490_giant-large-1.1.onnx', device='cuda')
o = onx.infer(images, exts, ixts, normalize_extrinsics=True)
t = np.load(scratch + '/trt_av.npz')
def med_rel(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.median(np.abs(a - b) / (np.abs(b) + 1e-6)))
def mx_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
for k in ('depth', 'depth_conf', 'extrinsics', 'intrinsics'):
    a, b = t[k], o[k]
    print(f"{k:12s} shape={tuple(a.shape)} med_rel={med_rel(a,b):.3e} max_abs={mx_abs(a,b):.3e}")
# fp16 gates (provisional; if exceeded STOP and report numbers)
assert med_rel(t['depth'], o['depth']) < 3e-2, 'depth med_rel'
assert mx_abs(t['depth_conf'], o['depth_conf']) < 1e-1, 'depth_conf max_abs'
assert mx_abs(t['extrinsics'], o['extrinsics']) < 1e-1, 'extrinsics max_abs'
assert med_rel(t['intrinsics'], o['intrinsics']) < 2e-2, 'intrinsics med_rel'
print('ANYVIEW TRT-vs-ONNX PARITY OK')
PY
```
Expected: per-key stats printed; gates pass; `ANYVIEW TRT-vs-ONNX PARITY OK`. If any gate fails, STOP and report the printed numbers (fp16 tolerances are provisional — adjudicate, do not loosen silently).

- [ ] **Step 5: Commit**

```bash
git add tools/model/da3anyview.py tools/compare_onnx_trt.py
git commit -m "refactor(tools): DA3AnyViewTRT inherits TRTModel + BaseDA3Model

Deletes the duplicated exact-resize preprocess, extrinsics-norm, parse_outputs
masking, and multi-input _infer in favour of the shared bases; mirrors
DA3AnyViewONNX (unmasked mapped outputs). Updates compare_onnx_trt.run_trt to
the generic _run. Verified TRT-vs-ONNX any-view parity (fp16).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: rework `da3metric.py` (`DA3MetricTRT`)

**Files:**
- Modify: `tools/model/da3metric.py` (full rewrite)

**Interfaces:**
- Consumes: `TRTModel` (`_run`), `BaseDA3Model` (`extract_metric`).
- Produces: `class DA3MetricTRT(TRTModel, BaseDA3Model)` with `infer_view(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (img is a preprocessed `(3,H,W)` CHW array → `(depth, sky)`).

- [ ] **Step 1: Rewrite `tools/model/da3metric.py`**

Replace the entire file with:

```python
"""Metric Depth Anything v3 TensorRT wrapper (single-image, raw depth + sky).

TRT sibling of ``DA3MetricONNX``.  Metric depth in metres is a caller-side
``focal * depth / 300`` step (the engine returns raw network depth + sky).
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_trt import TRTModel


class DA3MetricTRT(TRTModel, BaseDA3Model):
    """Metric TRT inference on a single already-preprocessed view."""

    def infer_view(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``img`` is a preprocessed ``(3, H, W)`` CHW array → ``(depth, sky)``."""
        raw = self._run({self.inputs[0]["name"]: img[None].astype(np.float32)})
        return self.extract_metric(raw)
```

This deletes `CameraIntrinsics`, focal scaling, crop, masking, `preprocess`, `parse_outputs`, `_MEAN`/`_STD`, and the `MonoDepthTRT` base.

- [ ] **Step 2: Black + flake8**

```bash
black --line-length 99 tools/model/da3metric.py
black --check --line-length 99 tools/model/da3metric.py
flake8 --max-line-length 100 tools/model/da3metric.py
```

- [ ] **Step 3: Metric TRT-vs-ONNX parity (two processes)**

Both run `infer_view` on the SAME preprocessed view (built with `base_da3.preprocess_views` at the metric target size).

Process A:
```bash
SCRATCH=/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad
python - "$SCRATCH" <<'PY'
import sys; sys.path.insert(0, 'tools')
scratch = sys.argv[1]
import numpy as np
from astribot_dataloader import load_images_cam_params
from model.da3metric import DA3MetricTRT
images, exts, ixts = load_images_cam_params('set1', 0)
m = DA3MetricTRT('weights/da3_metric_644x490_giant-large-1.1.engine')
view = m.preprocess_views(images[:1], ixts[:1], target_h=m.target_h, target_w=m.target_w)[0][0, 0]
d, s = m.infer_view(view)
np.savez(scratch + '/trt_metric.npz', view=view, depth=d, sky=s)
print('saved TRT metric:', d.shape, s.shape)
PY
```

Process B:
```bash
SCRATCH=/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad
python - "$SCRATCH" <<'PY'
import sys; sys.path.insert(0, 'tools')
scratch = sys.argv[1]
import numpy as np
from model.da3metric_onnx import DA3MetricONNX
t = np.load(scratch + '/trt_metric.npz')
m = DA3MetricONNX('weights/da3_metric_644x490_giant-large-1.1.onnx', device='cuda')
d, s = m.infer_view(t['view'])
def med_rel(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.median(np.abs(a - b) / (np.abs(b) + 1e-6)))
def mx_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
def med_abs(a, b):
    return float(np.median(np.abs(a.astype(np.float64) - b.astype(np.float64))))
print(f"depth shape={d.shape} med_rel={med_rel(t['depth'],d):.3e} med_abs={med_abs(t['depth'],d):.3e} max_abs={mx_abs(t['depth'],d):.3e}")
print(f"sky   shape={s.shape} med_abs={med_abs(t['sky'],s):.3e} max_abs={mx_abs(t['sky'],s):.3e}")
# fp16 gate: robust median stats (max/tail diffs are fp16-noisy on bounded sky)
assert med_rel(t['depth'], d) < 3e-2, 'depth med_rel'
assert med_abs(t['sky'], s) < 5e-3, 'sky med_abs'
print('METRIC TRT-vs-ONNX PARITY OK')
PY
```
Expected: stats printed, gates pass, `METRIC TRT-vs-ONNX PARITY OK`. If a gate fails, STOP and report numbers.

- [ ] **Step 4: Commit**

```bash
git add tools/model/da3metric.py
git commit -m "refactor(tools): DA3MetricTRT inherits TRTModel + BaseDA3Model

Mirrors DA3MetricONNX: infer_view(chw) -> (depth, sky) raw. Drops the
CameraIntrinsics focal-scaling/crop/mask standalone path (metric depth in
metres is now a caller-side focal*depth/300 step). Verified TRT-vs-ONNX metric
parity (fp16).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: rework `da3nested.py` (`DA3NestedTRT`)

**Files:**
- Modify: `tools/model/da3nested.py` (full rewrite)

**Interfaces:**
- Consumes: `DA3AnyViewTRT` (Task 2), `DA3MetricTRT` (Task 3), `BaseDA3Model` (`normalize_extrinsics`, `map_anyview_keys`, `align_with_metric`, `align_to_input`, `crop_to_tile`), `TRTModel._run`.
- Produces: `class DA3NestedTRT(BaseDA3Model)` with `__init__(anyview_engine, metric_engine, conf_thresh=0.5)` and `infer(imgs, extrs, intrs, *, align_input_ext_scale=True) -> dict` (cropped tile-resolution `{depth (N,tile_h,tile_w), depth_conf, extrinsics (N,3,4), intrinsics (N,3,3)}`).

- [ ] **Step 1: Rewrite `tools/model/da3nested.py`**

Replace the entire file with:

```python
"""Nested Depth Anything v3 TensorRT pipeline (any-view + metric + alignment).

Composes ``DA3AnyViewTRT`` and ``DA3MetricTRT`` and aligns their outputs to
reproduce ``NestedDepthAnything3Net`` — the TRT sibling of ``DA3NestedONNX``.
Output depth is left **unmasked** (only sky regions are set to max depth inside
``align_anyview_with_metric``).
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.da3anyview import DA3AnyViewTRT
from model.da3metric import DA3MetricTRT


class DA3NestedTRT(BaseDA3Model):
    """Any-view + metric TRT pipeline replicating the nested PyTorch model."""

    def __init__(
        self, anyview_engine: str, metric_engine: str, conf_thresh: float = 0.5,
    ) -> None:
        self.av = DA3AnyViewTRT(anyview_engine, conf_thresh=conf_thresh)
        self.metric = DA3MetricTRT(metric_engine)
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        print(
            f"[NESTED-TRT] anyview N={self.av.num_views} @ "
            f"{self.av.target_h}x{self.av.target_w}, "
            f"metric @ {self.metric.target_h}x{self.metric.target_w}"
        )

    def infer(
        self,
        imgs: list,
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        align_input_ext_scale: bool = True,
    ) -> dict:
        """Run the full nested pipeline; returns cropped numpy outputs."""
        n = len(imgs)
        if self.av.num_views is not None and n != self.av.num_views:
            raise ValueError(
                f"Got {n} views but the any-view engine expects "
                f"{self.av.num_views}. Pass exactly {self.av.num_views} views."
            )

        # 1. Any-view: letterbox preprocess + normalize + run + map
        img_batch, intrs_adj, metas = self.av.preprocess_views(imgs, intrs)
        extrs_norm = self.normalize_extrinsics(extrs)
        av = self.av.map_anyview_keys(self.av._run({
            "image": img_batch.astype(np.float32),
            "extrinsics": extrs_norm[None].astype(np.float32),
            "intrinsics": intrs_adj[None].astype(np.float32),
        }))

        # 2. Metric branch (letterbox grid; reuse the any-view padded views)
        metric_depths, metric_skys = self._run_metric_branch(img_batch)

        # 3. Align any-view depth to metric (padded grid; sky handling inside)
        result = self.align_with_metric(av, metric_depths, metric_skys)

        # 4. Optional Umeyama align to input poses (PyTorch inference default)
        if align_input_ext_scale:
            result = self.align_to_input(result, extrs, intrs_adj)

        # 5. Crop padded outputs to the tile; un-pad intrinsics
        return self._crop_result(result, metas)

    def _run_metric_branch(self, av_img_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-view metric inference on the letterbox grid.

        Requires the metric engine to share the any-view target size (fail-fast
        otherwise); reuses the already-letterboxed any-view padded views.
        Returns ``(1, N, H, W)``.
        """
        h, w = self.av.target_h, self.av.target_w
        mh, mw = self.metric.target_h, self.metric.target_w
        if (mh, mw) != (h, w):
            raise NotImplementedError(
                "Nested pipeline requires the metric and any-view engines to "
                f"share the input size; got metric {mh}x{mw} vs any-view {h}x{w}. "
                "Re-export the metric engine at the any-view resolution."
            )
        n = av_img_batch.shape[1]
        depths = np.zeros((1, n, h, w), dtype=np.float32)
        skys = np.zeros((1, n, h, w), dtype=np.float32)
        for i in range(n):
            d, s = self.metric.infer_view(av_img_batch[0, i])
            depths[0, i] = d
            skys[0, i] = s
        return depths, skys

    def _crop_result(self, result: dict, metas: list) -> dict:
        """Crop padded depth/conf to the tile region and un-pad the intrinsics."""
        depth = np.stack([
            self.av.crop_to_tile(result["depth"][i], metas[i]) for i in range(len(metas))
        ])
        conf = np.stack([
            self.av.crop_to_tile(result["depth_conf"][i], metas[i]) for i in range(len(metas))
        ])
        intr = result["intrinsics"].copy()
        for i, m in enumerate(metas):
            intr[i, 0, 2] -= m["pad_left"]
            intr[i, 1, 2] -= m["pad_top"]
        return {
            "depth": depth,
            "depth_conf": conf,
            "extrinsics": result["extrinsics"],
            "intrinsics": intr,
        }
```

This deletes the old `_map_anyview_keys`, `_extract_metric`, the `CameraIntrinsics` param and import, the inline align/squeeze/mask, and the `_MEAN`/`_STD` module constants.

- [ ] **Step 2: Black + flake8**

```bash
black --line-length 99 tools/model/da3nested.py
black --check --line-length 99 tools/model/da3nested.py
flake8 --max-line-length 100 tools/model/da3nested.py
```

- [ ] **Step 3: Nested smoke-test (cropped shapes + guard)**

```bash
python -c "
import sys; sys.path.insert(0, 'tools')
from astribot_dataloader import load_images_cam_params
from model.da3nested import DA3NestedTRT
images, exts, ixts = load_images_cam_params('set1', 0)
n = DA3NestedTRT('weights/da3_anyview_n3_644x490_giant-large-1.1.engine',
                   'weights/da3_metric_644x490_giant-large-1.1.engine')
r = n.infer(images, exts, ixts, align_input_ext_scale=True)
for k in ('depth','depth_conf','extrinsics','intrinsics'):
    print(k, r[k].shape, r[k].dtype)
assert r['depth'].shape == (3, 480, 640), r['depth'].shape
assert r['extrinsics'].shape == (3,3,4) and r['intrinsics'].shape == (3,3,3)
# guard fires on wrong view count
img2, e2, i2 = load_images_cam_params('set0', 0)
try:
    n.infer(img2, e2, i2); print('GUARD FAIL')
except ValueError as ex:
    print('OK guard:', ex)
print('NESTED-TRT SMOKE OK')
" 2>&1 | tail -8
```
Expected: depth/conf `(3, 480, 640)`, extrinsics `(3,3,4)`, intrinsics `(3,3,3)`, `OK guard: ...`, `NESTED-TRT SMOKE OK`.

- [ ] **Step 4: Nested TRT-vs-ONNX parity (two processes)**

Process A:
```bash
SCRATCH=/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad
python - "$SCRATCH" <<'PY'
import sys; sys.path.insert(0, 'tools')
scratch = sys.argv[1]
import numpy as np
from astribot_dataloader import load_images_cam_params
from model.da3nested import DA3NestedTRT
images, exts, ixts = load_images_cam_params('set1', 0)
n = DA3NestedTRT('weights/da3_anyview_n3_644x490_giant-large-1.1.engine',
                   'weights/da3_metric_644x490_giant-large-1.1.engine')
r = n.infer(images, exts, ixts, align_input_ext_scale=True)
np.savez(scratch + '/trt_nested.npz', **{k: np.asarray(v) for k, v in r.items()})
print('saved TRT nested:', {k: r[k].shape for k in r})
PY
```

Process B:
```bash
SCRATCH=/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad
python - "$SCRATCH" <<'PY'
import sys; sys.path.insert(0, 'tools')
scratch = sys.argv[1]
import numpy as np
from astribot_dataloader import load_images_cam_params
from model.da3nested_onnx import DA3NestedONNX
images, exts, ixts = load_images_cam_params('set1', 0)
onx = DA3NestedONNX('weights/da3_anyview_n3_644x490_giant-large-1.1.onnx',
                    'weights/da3_metric_644x490_giant-large-1.1.onnx', device='cuda')
o = onx.infer(images, exts, ixts, align_input_ext_scale=True)
t = np.load(scratch + '/trt_nested.npz')
def med_rel(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.median(np.abs(a - b) / (np.abs(b) + 1e-6)))
def mx_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
for k in ('depth', 'depth_conf', 'extrinsics', 'intrinsics'):
    a, b = t[k], o[k]
    print(f"{k:12s} shape={tuple(a.shape)} med_rel={med_rel(a,b):.3e} max_abs={mx_abs(a,b):.3e}")
assert med_rel(t['depth'], o['depth']) < 3e-2, 'depth med_rel'
assert mx_abs(t['extrinsics'], o['extrinsics']) < 1e-1, 'extrinsics max_abs'
assert med_rel(t['intrinsics'], o['intrinsics']) < 2e-2, 'intrinsics med_rel'
print('NESTED TRT-vs-ONNX PARITY OK')
PY
```
Expected: per-key stats; gates pass; `NESTED TRT-vs-ONNX PARITY OK`. If a gate fails, STOP and report numbers (fp16 tolerances provisional).

- [ ] **Step 5: Commit**

```bash
git add tools/model/da3nested.py
git commit -m "refactor(tools): DA3NestedTRT inherits BaseDA3Model, composes TRT models

Mirrors DA3NestedONNX exactly (letterbox preprocess -> anyview+metric engines ->
align -> crop). Drops the duplicated key-mapping/extract-metric/inline-align and
the CameraIntrinsics param. Verified TRT-vs-ONNX nested parity (fp16).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Import style:** run from repo root; `from model.X import Y`, never `from tools.model...`.
- **Two-process parity:** always run the TRT half and the ONNX half in separate `python` invocations — a giant TRT engine arena + an ONNX CUDA arena co-resident on the GPU will OOM. Process A saves an npz; Process B loads + compares.
- **fp16 tolerances are provisional.** The gates (depth med_rel < 3e-2, extrinsics max_abs < 1e-1, intrinsics med_rel < 2e-2, conf/sky max_abs < 1e-1) are engineering estimates for fp16-vs-fp32. If a real run exceeds one, STOP and report the printed numbers — the controller adjudicates whether it's fp16 noise (loosen with justification) or a real bug. Never loosen a gate silently.
- **Do not add confidence masking** anywhere. Depth stays unmasked (only sky handling inside `align_anyview_with_metric`).
- **`_run` inputs are pre-shaped.** `preprocess_views` returns `image` as `(1,N,3,H,W)`; extrinsics/intrinsics need a leading batch dim (`ext[None]`). `_to_input_tensor` does not transpose — it only moves to CUDA + casts dtype.
```
