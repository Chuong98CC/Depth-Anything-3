# ONNX Base-Class Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract duplicated ONNX inference code in `tools/` behind two new base classes (`ONNXModel`, `BaseDA3Model`), add three concrete ONNX model classes, and migrate `infer_onnx_nested.py` onto them with byte-identical output.

**Architecture:** Two-axis split. `base_onnx.ONNXModel` owns ONNX Runtime session mechanics (sibling of `base_trt.TRTModel`). `base_da3.BaseDA3Model` is a backend-agnostic mixin owning DA3 preprocessing, extrinsics normalization, output-key mapping, and alignment orchestration (thin wrappers over the existing `alignment.py`). Concrete classes multiple-inherit both; the nested class composes the two session models.

**Tech Stack:** Python 3.10, numpy, onnxruntime, opencv (cv2), torch. Scripts run from repo root; `tools/` is on `sys.path[0]` so imports are `from model.X import Y` and `from astribot_dataloader import Z` (no `tools.` prefix, no `__init__` package path).

## Global Constraints

- **Run from repo root.** All `tools/` scripts import from the installed `depth_anything_3` package and use bare `model.*` / `astribot_dataloader` imports (no `tools.` prefix).
- **No pytest suite exists.** Verification is manual per CLAUDE.md: import smoke-tests via `python -c`, instantiating real ONNX models in `weights/`, and golden-output `.npz` diffs. Put any throwaway scripts in the scratchpad, never in the repo.
- **Black line length 99; flake8 100.** Match surrounding style.
- **Preserve behavior exactly.** The nested ONNX output must remain **unmasked** depth + confidence (PyTorch does NOT confidence-mask depth; the `depth[~valid]=-1` masking in the TRT wrappers is TRT-only and must NOT be added). Sky-region handling stays via `align_anyview_with_metric`.
- **Do NOT modify** `base_trt.py`, `da3anyview.py`, `da3metric.py`, `da3nested.py`, `alignment.py`, or the `compare_*` / `infer_onnx_metric_depth` scripts. TRT migration is a later pass.
- **ImageNet normalization** is `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`, `img/255`, cv2 `INTER_LINEAR`, BGR→RGB. Reproduce exactly for numeric parity.
- **Default nested ONNX models** present at `weights/da3_anyview_n3_644x490_giant-large-1.1.onnx` (5-D input `(1,3,3,490,644)`, N=3) and `weights/da3_metric_644x490_giant-large-1.1.onnx` (4-D input). Astribot data present at `/home/chuong/workspace/demo_data/astribot_stereo_lrb`.

---

### Task 1: `base_onnx.py` — `ONNXModel`

**Files:**
- Create: `tools/model/base_onnx.py`

**Interfaces:**
- Consumes: nothing (leaf).
- Produces:
  - `class ONNXModel(onnx_path: str, device: str = "cuda")`
  - attributes: `self.session` (`ort.InferenceSession`), `self.inputs: list[dict]` and `self.outputs: list[dict]` (each `{"name": str, "shape": list}`), `self.target_h: int|None`, `self.target_w: int|None`, `self.num_views: int|None`
  - properties: `input_width -> int|None`, `input_height -> int|None`
  - method: `run(feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write the module**

Create `tools/model/base_onnx.py`:

```python
"""ONNX Runtime base wrapper for Depth Anything 3 models.

Sibling of ``base_trt.TRTModel``: owns the ONNX Runtime session, exposes input/
output tensor metadata, resolves the model's fixed input geometry, and runs
inference.  DA3-specific pre/post-processing lives in ``base_da3.BaseDA3Model``;
concrete models multiple-inherit both.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


class ONNXModel:
    """Thin ONNX Runtime session wrapper.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` file (external ``.data`` weights resolved by ORT).
    device : str
        ``"cuda"`` (CUDA EP with CPU fallback) or ``"cpu"``.
    """

    def __init__(self, onnx_path: str, device: str = "cuda") -> None:
        if device == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[ONNX] Loaded {onnx_path}")
        print(f"[ONNX] Provider: {self.session.get_providers()[0]}")

        self.inputs = [{"name": i.name, "shape": i.shape} for i in self.session.get_inputs()]
        self.outputs = [{"name": o.name, "shape": o.shape} for o in self.session.get_outputs()]
        for i in self.inputs:
            print(f"[ONNX] Input : {i['name']}  shape={i['shape']}")
        for o in self.outputs:
            print(f"[ONNX] Output: {o['name']}  shape={o['shape']}")

        self._resolve_input_geometry()

    def _resolve_input_geometry(self) -> None:
        """Set ``target_h``/``target_w``/``num_views`` from the first input shape.

        Supports 5-D any-view input ``(B, N, 3, H, W)`` and 4-D metric input
        ``(B, 3, H, W)``.  Symbolic (non-int) dims become ``None``.
        """
        shape = self.inputs[0]["shape"]

        def _dim(idx: int) -> int | None:
            return shape[idx] if idx < len(shape) and isinstance(shape[idx], int) else None

        if len(shape) == 5:      # (B, N, 3, H, W) — any-view
            self.num_views = _dim(1)
            self.target_h = _dim(3)
            self.target_w = _dim(4)
        elif len(shape) == 4:    # (B, 3, H, W) — metric / mono
            self.num_views = None
            self.target_h = _dim(2)
            self.target_w = _dim(3)
        else:
            raise ValueError(
                f"Unexpected input rank {len(shape)} for {self.inputs[0]['name']}: {shape}"
            )

        # Preprocessing needs concrete H/W; fail clearly rather than defaulting.
        if self.target_h is None or self.target_w is None:
            raise ValueError(
                f"Input '{self.inputs[0]['name']}' has non-static H/W {shape}; "
                "export the model with fixed height/width."
            )

    @property
    def input_width(self) -> int | None:
        return self.target_w

    @property
    def input_height(self) -> int | None:
        return self.target_h

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the session, returning ``{output_name: array}``."""
        names = [o["name"] for o in self.outputs]
        return dict(zip(names, self.session.run(names, feed)))
```

- [ ] **Step 2: Import smoke-test + geometry check on a real model**

Run (from repo root):

```bash
python -c "
import sys; sys.path.insert(0, 'tools')
from model.base_onnx import ONNXModel
m = ONNXModel('weights/da3_anyview_n3_644x490_giant-large-1.1.onnx', device='cpu')
print('ANYVIEW geom:', m.num_views, m.target_h, m.target_w)
assert (m.num_views, m.target_h, m.target_w) == (3, 490, 644), (m.num_views, m.target_h, m.target_w)
mm = ONNXModel('weights/da3_metric_644x490_giant-large-1.1.onnx', device='cpu')
print('METRIC geom:', mm.num_views, mm.target_h, mm.target_w)
assert mm.num_views is None and mm.target_h == 490 and mm.target_w == 644, (mm.num_views, mm.target_h, mm.target_w)
print('OK')
"
```

Expected: prints both geometries and `OK` (no assertion error). Any-view N=3, 490×644; metric num_views=None, 490×644.

- [ ] **Step 3: Commit**

```bash
git add tools/model/base_onnx.py
git commit -m "feat(tools): add ONNXModel base session wrapper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `base_da3.py` — `BaseDA3Model` mixin

**Files:**
- Create: `tools/model/base_da3.py`

**Interfaces:**
- Consumes: `self.target_h`, `self.target_w` (provided by `ONNXModel` at runtime); `alignment.align_anyview_with_metric`, `alignment.align_to_input_ext_scale`.
- Produces (mixin methods, all rely on `self.target_h`/`self.target_w`):
  - `preprocess_views(imgs: list[str|np.ndarray], intrs: np.ndarray, target_h: int|None=None, target_w: int|None=None) -> tuple[np.ndarray, np.ndarray, list[dict]]` → `(img_batch (1,N,3,H,W), intrs_adj (N,3,3), metas)`
  - `_preprocess_one(img, K, target_h, target_w) -> tuple[np.ndarray, np.ndarray, dict]`
  - `@staticmethod normalize_extrinsics(extrs: np.ndarray) -> np.ndarray` → `(N,4,4)`
  - `@staticmethod map_anyview_keys(raw: dict) -> dict` → `{depth, depth_conf, extrinsics, intrinsics}`
  - `@staticmethod extract_metric(raw: dict) -> tuple[np.ndarray, np.ndarray]` → `(depth, sky)`
  - `align_with_metric(av: dict, metric_depths: np.ndarray, metric_skys: np.ndarray) -> dict[str, np.ndarray]` (inputs `(1,N,H,W)`; returns numpy `{depth,depth_conf,extrinsics,intrinsics}` squeezed to `(N,…)`)
  - `align_to_input(result: dict, input_extrinsics: np.ndarray, input_intrinsics: np.ndarray) -> dict`

- [ ] **Step 1: Write the module**

Create `tools/model/base_da3.py`:

```python
"""Backend-agnostic Depth Anything 3 pre/post-processing mixin.

Holds the preprocessing, extrinsics normalization, output-name mapping, and
alignment orchestration shared by every DA3 inference wrapper.  It is a pure
mixin: it defines no ``__init__`` and relies on ``self.target_h`` / ``self.target_w``
supplied by the backend base (``ONNXModel`` now; ``TRTModel`` later).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from model.alignment import align_anyview_with_metric, align_to_input_ext_scale


class BaseDA3Model:
    """Shared DA3 preprocessing + alignment.  Requires ``self.target_h/target_w``."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ---- preprocessing (exact-resize style) --------------------------------

    def preprocess_views(
        self,
        imgs: list,
        intrs: np.ndarray,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Resize/normalize *N* views to ``(1, N, 3, H, W)`` with scaled intrinsics.

        ``target_h``/``target_w`` default to ``self.target_h``/``self.target_w``;
        pass explicit values to preprocess for a differently-sized model (e.g. the
        metric branch inside the nested pipeline).
        """
        th = target_h if target_h is not None else self.target_h
        tw = target_w if target_w is not None else self.target_w
        n = len(imgs)
        proc = np.zeros((n, 3, th, tw), dtype=np.float32)
        intrs_out = np.zeros((n, 3, 3), dtype=np.float32)
        metas: list[dict] = []
        for i in range(n):
            proc[i], intrs_out[i], meta = self._preprocess_one(imgs[i], intrs[i], th, tw)
            metas.append(meta)
        return proc[None], intrs_out, metas  # add B=1

    def _preprocess_one(
        self, img, K: np.ndarray, target_h: int, target_w: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Preprocess a single view (path or BGR array)."""
        if isinstance(img, str):
            bgr = cv2.imread(img, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Could not load image: {img}")
        elif isinstance(img, np.ndarray):
            bgr = img.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_f = rgb_r.astype(np.float32) / 255.0
        chw = ((img_f - self._MEAN) / self._STD).transpose(2, 0, 1).astype(np.float32)

        sx, sy = target_w / orig_w, target_h / orig_h
        K_adj = K.copy().astype(np.float32)
        K_adj[0, 0] *= sx
        K_adj[0, 2] *= sx
        K_adj[1, 1] *= sy
        K_adj[1, 2] *= sy

        meta = {"orig_h": orig_h, "orig_w": orig_w, "scale_x": sx, "scale_y": sy}
        return chw, K_adj, meta

    # ---- extrinsics normalization ------------------------------------------

    @staticmethod
    def normalize_extrinsics(extrs: np.ndarray) -> np.ndarray:
        """First camera to origin, median camera distance = 1 (clamped 1e-1).

        Numpy replica of ``DepthAnything3._normalize_extrinsics``.  ``extrs`` is
        ``(N, 4, 4)`` world-to-camera.
        """
        ex_t = extrs.copy()
        transform = np.linalg.inv(ex_t[0])
        ex_t_norm = ex_t @ transform
        c2ws = np.linalg.inv(ex_t_norm)
        dists = np.linalg.norm(c2ws[..., :3, 3], axis=-1)
        median_dist = max(float(np.median(dists)), 1e-1)
        ex_t_norm[..., :3, 3] /= median_dist
        return ex_t_norm

    # ---- output-name resolvers ---------------------------------------------

    @staticmethod
    def map_anyview_keys(raw: dict) -> dict:
        """Normalise any-view output keys to ``depth/depth_conf/extrinsics/intrinsics``."""
        out: dict[str, np.ndarray] = {}
        for name, val in raw.items():
            low = name.lower()
            if "depth_conf" in low or "conf" in low:
                out["depth_conf"] = val
            elif "pred_extrinsics" in low:
                out["extrinsics"] = val
            elif "pred_intrinsics" in low:
                out["intrinsics"] = val
            elif "depth" in low:
                out["depth"] = val
            else:
                out[name] = val
        return out

    @staticmethod
    def extract_metric(raw: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract ``(depth, sky)`` from raw metric outputs."""
        depth = sky = None
        for name, val in raw.items():
            low = name.lower()
            arr = val.squeeze().astype(np.float32)
            if "sky" in low:
                sky = arr
            elif "depth" in low:
                depth = arr
            elif depth is None:
                depth = arr
        if sky is None:
            sky = np.zeros_like(depth)
        return depth, sky

    # ---- alignment orchestration -------------------------------------------

    def align_with_metric(
        self, av: dict, metric_depths: np.ndarray, metric_skys: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run ``align_anyview_with_metric`` and squeeze the batch dim.

        ``av`` holds numpy any-view outputs (batched, ``(1,N,…)``).
        ``metric_depths``/``metric_skys`` are ``(1, N, H, W)``.  Returns numpy
        ``{depth, depth_conf, extrinsics, intrinsics}`` squeezed to ``(N, …)``.
        """
        aligned = align_anyview_with_metric(
            anyview_depth=torch.from_numpy(av["depth"]),
            anyview_conf=torch.from_numpy(av["depth_conf"]),
            anyview_extrinsics=torch.from_numpy(av["extrinsics"]),
            anyview_intrinsics=torch.from_numpy(av["intrinsics"]),
            metric_depth=torch.from_numpy(metric_depths),
            metric_sky=torch.from_numpy(metric_skys),
        )
        out: dict[str, np.ndarray] = {}
        for k in ("depth", "depth_conf", "extrinsics", "intrinsics"):
            val = aligned[k].float().cpu().numpy()
            if val.ndim >= 4 and val.shape[0] == 1:
                val = val.squeeze(0)
            out[k] = val
        return out

    def align_to_input(
        self, result: dict, input_extrinsics: np.ndarray, input_intrinsics: np.ndarray,
    ) -> dict:
        """Umeyama-align the prediction to the input camera poses (in place-ish).

        Mirrors ``DepthAnything3.inference(align_to_input_ext_scale=True)``.
        """
        aligned = align_to_input_ext_scale(
            pred_depth=result["depth"],
            pred_extrinsics=result["extrinsics"],
            input_extrinsics=input_extrinsics,
            input_intrinsics=input_intrinsics,
            align_scale=True,
        )
        result = dict(result)
        result["depth"] = aligned["depth"]
        result["extrinsics"] = aligned["extrinsics"]
        result["intrinsics"] = aligned["intrinsics"]
        return result
```

- [ ] **Step 2: Unit smoke-test the pure helpers**

Run:

```bash
python -c "
import sys; sys.path.insert(0, 'tools')
import numpy as np
from model.base_da3 import BaseDA3Model

# normalize_extrinsics: first cam -> identity rotation at origin, median dist 1
rng = np.random.default_rng(0)
def rand_ext(n):
    E=[]
    for _ in range(n):
        A=rng.standard_normal((3,3)); Q,_=np.linalg.qr(A)
        if np.linalg.det(Q)<0: Q[:,0]*=-1
        M=np.eye(4); M[:3,:3]=Q; M[:3,3]=rng.standard_normal(3); E.append(M)
    return np.stack(E)
ext = rand_ext(3)
norm = BaseDA3Model.normalize_extrinsics(ext)
c2w0 = np.linalg.inv(norm[0])
assert np.allclose(c2w0[:3,3], 0, atol=1e-6), 'first cam not at origin'
print('normalize_extrinsics OK')

# map_anyview_keys
raw = {'depth':1,'depth_conf':2,'pred_extrinsics':3,'pred_intrinsics':4}
m = BaseDA3Model.map_anyview_keys(raw)
assert set(m) == {'depth','depth_conf','extrinsics','intrinsics'}, m
print('map_anyview_keys OK')

# extract_metric
d,s = BaseDA3Model.extract_metric({'depth':np.ones((1,1,4,5)),'sky':np.zeros((1,1,4,5))})
assert d.shape==(4,5) and s.shape==(4,5)
print('extract_metric OK')
print('ALL OK')
"
```

Expected: `normalize_extrinsics OK`, `map_anyview_keys OK`, `extract_metric OK`, `ALL OK`.

- [ ] **Step 3: Commit**

```bash
git add tools/model/base_da3.py
git commit -m "feat(tools): add BaseDA3Model preprocessing/alignment mixin

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: concrete session models — `DA3AnyViewONNX`, `DA3MetricONNX`

**Files:**
- Create: `tools/model/da3anyview_onnx.py`
- Create: `tools/model/da3metric_onnx.py`

**Interfaces:**
- Consumes: `ONNXModel` (Task 1), `BaseDA3Model` (Task 2).
- Produces:
  - `class DA3AnyViewONNX(ONNXModel, BaseDA3Model)` with `infer(imgs, extrs, intrs, *, normalize_extrinsics=False) -> dict` (`{depth,depth_conf,extrinsics,intrinsics}`, batched `(1,N,…)` values as returned by the model).
  - `class DA3MetricONNX(ONNXModel, BaseDA3Model)` with `infer_view(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]` where `img` is a preprocessed `(3, H, W)` CHW array → `(depth, sky)`.

- [ ] **Step 1: Write `da3anyview_onnx.py`**

```python
"""Any-view Depth Anything 3 ONNX wrapper.

Multi-view depth + confidence + predicted camera parameters.  Built on
``ONNXModel`` (session) + ``BaseDA3Model`` (pre/post).
"""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_onnx import ONNXModel


class DA3AnyViewONNX(ONNXModel, BaseDA3Model):
    """Any-view ONNX inference: images (+ extrinsics/intrinsics) → depth bundle."""

    def infer(
        self,
        imgs: list,
        extrs: np.ndarray,
        intrs: np.ndarray,
        *,
        normalize_extrinsics: bool = False,
    ) -> dict:
        """Run the any-view model.

        ``imgs`` are *N* paths or BGR arrays; ``extrs`` ``(N,4,4)``; ``intrs``
        ``(N,3,3)`` at original resolution.  Returns the mapped output dict with
        batched values (``(1,N,…)``).  Extrinsics are normalized here only when
        ``normalize_extrinsics=True`` (the ONNX graph does not normalize).
        """
        img_batch, intrs_adj, _ = self.preprocess_views(imgs, intrs)
        ext = self.normalize_extrinsics(extrs) if normalize_extrinsics else extrs
        raw = self.run({
            "image": img_batch.astype(np.float32),
            "extrinsics": ext[None].astype(np.float32),
            "intrinsics": intrs_adj[None].astype(np.float32),
        })
        return self.map_anyview_keys(raw)
```

- [ ] **Step 2: Write `da3metric_onnx.py`**

```python
"""Metric Depth Anything 3 ONNX wrapper (single-image, raw depth + sky)."""

from __future__ import annotations

import numpy as np

from model.base_da3 import BaseDA3Model
from model.base_onnx import ONNXModel


class DA3MetricONNX(ONNXModel, BaseDA3Model):
    """Metric ONNX inference on a single already-preprocessed view."""

    def infer_view(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``img`` is a preprocessed ``(3, H, W)`` CHW array → ``(depth, sky)``."""
        raw = self.run({"image": img[None].astype(np.float32)})
        return self.extract_metric(raw)
```

- [ ] **Step 3: Smoke-test both against real models + data**

Run:

```bash
python -c "
import sys; sys.path.insert(0, 'tools')
import numpy as np
from astribot_dataloader import load_images_cam_params
from model.da3anyview_onnx import DA3AnyViewONNX
from model.da3metric_onnx import DA3MetricONNX

images, exts, ixts = load_images_cam_params('set1', 0)   # 3 views
av = DA3AnyViewONNX('weights/da3_anyview_n3_644x490_giant-large-1.1.onnx', device='cuda')
out = av.infer(images, exts, ixts, normalize_extrinsics=True)
print('anyview keys:', sorted(out))
assert set(out) >= {'depth','depth_conf','extrinsics','intrinsics'}, out
print('anyview depth shape:', out['depth'].shape)

# metric: preprocess one view via the mixin at the metric model's size, then infer
m = DA3MetricONNX('weights/da3_metric_644x490_giant-large-1.1.onnx', device='cuda')
mv = av.preprocess_views(images[:1], ixts[:1], target_h=m.target_h, target_w=m.target_w)[0][0,0]
d, s = m.infer_view(mv)
print('metric depth/sky shapes:', d.shape, s.shape)
assert d.shape == s.shape
print('OK')
"
```

Expected: prints any-view keys `['depth','depth_conf','extrinsics','intrinsics']`, a depth shape, metric depth/sky shapes equal, then `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/model/da3anyview_onnx.py tools/model/da3metric_onnx.py
git commit -m "feat(tools): add DA3AnyViewONNX and DA3MetricONNX session models

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `da3nested_onnx.py` — `DA3NestedONNX`

**Files:**
- Create: `tools/model/da3nested_onnx.py`

**Interfaces:**
- Consumes: `DA3AnyViewONNX`, `DA3MetricONNX` (Task 3), `BaseDA3Model` (Task 2).
- Produces:
  - `class DA3NestedONNX(BaseDA3Model)` with `__init__(anyview_path: str, metric_path: str, device: str = "cuda")` (attrs `self.av`, `self.metric`, `self.target_h`, `self.target_w`) and `infer(imgs, extrs, intrs, *, align_input_ext_scale: bool = True) -> dict` returning numpy `{depth (N,H,W), depth_conf (N,H,W), extrinsics (N,3,4), intrinsics (N,3,3)}`.

- [ ] **Step 1: Write the module**

```python
"""Nested Depth Anything 3 ONNX pipeline (any-view + metric + alignment).

Composes ``DA3AnyViewONNX`` and ``DA3MetricONNX`` and aligns their outputs to
reproduce ``NestedDepthAnything3Net`` — the ONNX sibling of the TRT
``DA3NestedTRT``.  Output depth is left **unmasked** (PyTorch does not
confidence-mask depth; only sky regions are set to max depth, handled inside
``align_anyview_with_metric``).
"""

from __future__ import annotations

import cv2
import numpy as np

from model.base_da3 import BaseDA3Model
from model.da3anyview_onnx import DA3AnyViewONNX
from model.da3metric_onnx import DA3MetricONNX


class DA3NestedONNX(BaseDA3Model):
    """Any-view + metric ONNX pipeline replicating the nested PyTorch model."""

    def __init__(self, anyview_path: str, metric_path: str, device: str = "cuda") -> None:
        self.av = DA3AnyViewONNX(anyview_path, device)
        self.metric = DA3MetricONNX(metric_path, device)
        # BaseDA3Model methods (normalize_extrinsics, align_*) need target size.
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        print(
            f"[NESTED] anyview N={self.av.num_views} @ {self.av.target_h}x{self.av.target_w}, "
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
        """Run the full nested pipeline; returns numpy depth/conf/extrinsics/intrinsics."""
        n = len(imgs)
        if self.av.num_views is not None and n != self.av.num_views:
            raise ValueError(
                f"Got {n} views but the any-view ONNX model expects "
                f"{self.av.num_views}. Pass exactly {self.av.num_views} views."
            )

        # 1. Any-view: preprocess + normalize + run + map
        img_batch, intrs_adj, _ = self.av.preprocess_views(imgs, intrs)
        extrs_norm = self.normalize_extrinsics(extrs)
        av = self.av.map_anyview_keys(self.av.run({
            "image": img_batch.astype(np.float32),
            "extrinsics": extrs_norm[None].astype(np.float32),
            "intrinsics": intrs_adj[None].astype(np.float32),
        }))

        # 2. Metric branch (one view at a time)
        metric_depths, metric_skys = self._run_metric_branch(img_batch)

        # 3. Align any-view depth to metric (sky handling inside)
        result = self.align_with_metric(av, metric_depths, metric_skys)

        # 4. Optional Umeyama align to input poses (PyTorch inference default)
        if align_input_ext_scale:
            result = self.align_to_input(result, extrs, intrs_adj)

        return result

    def _run_metric_branch(self, img_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-view metric inference on the already-preprocessed any-view batch.

        ``img_batch`` is ``(1, N, 3, H_av, W_av)``.  Each view is resized to the
        metric model's H/W when it differs, run, then depth/sky are resized back
        to the any-view H/W.  Returns ``(1, N, H_av, W_av)`` depth and sky.
        """
        n = img_batch.shape[1]
        h, w = self.av.target_h, self.av.target_w
        mh, mw = self.metric.target_h, self.metric.target_w
        need_resize = (mh, mw) != (h, w)

        depths = np.zeros((1, n, h, w), dtype=np.float32)
        skys = np.zeros((1, n, h, w), dtype=np.float32)
        for i in range(n):
            view = img_batch[0, i]  # (3, H_av, W_av)
            if need_resize:
                view = cv2.resize(
                    view.transpose(1, 2, 0), (mw, mh), interpolation=cv2.INTER_LINEAR,
                ).transpose(2, 0, 1)
            d, s = self.metric.infer_view(view)
            if need_resize:
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
                s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
            depths[0, i] = d
            skys[0, i] = s
        return depths, skys
```

- [ ] **Step 2: Smoke-test the full pipeline against real models + data**

Run:

```bash
python -c "
import sys; sys.path.insert(0, 'tools')
from astribot_dataloader import load_images_cam_params
from model.da3nested_onnx import DA3NestedONNX

images, exts, ixts = load_images_cam_params('set1', 0)
nested = DA3NestedONNX(
    'weights/da3_anyview_n3_644x490_giant-large-1.1.onnx',
    'weights/da3_metric_644x490_giant-large-1.1.onnx',
    device='cuda',
)
r = nested.infer(images, exts, ixts, align_input_ext_scale=True)
for k in ('depth','depth_conf','extrinsics','intrinsics'):
    print(k, r[k].shape, r[k].dtype)
assert r['depth'].shape[0] == 3
print('OK')
"
```

Expected: prints the four output shapes (`depth`/`depth_conf` `(3,490,644)`, `extrinsics` `(3,3,4)`, `intrinsics` `(3,3,3)`) and `OK`.

- [ ] **Step 3: View-count guard test**

Run (expects a clear ValueError, not an ORT shape error):

```bash
python -c "
import sys; sys.path.insert(0, 'tools')
from astribot_dataloader import load_images_cam_params
from model.da3nested_onnx import DA3NestedONNX
images, exts, ixts = load_images_cam_params('set0', 0)   # 2 views vs N=3 model
nested = DA3NestedONNX(
    'weights/da3_anyview_n3_644x490_giant-large-1.1.onnx',
    'weights/da3_metric_644x490_giant-large-1.1.onnx', device='cuda')
try:
    nested.infer(images, exts, ixts)
    print('FAIL: no error raised')
except ValueError as e:
    print('OK guard:', e)
"
```

Expected: `OK guard: Got 2 views but the any-view ONNX model expects 3. ...`

- [ ] **Step 4: Commit**

```bash
git add tools/model/da3nested_onnx.py
git commit -m "feat(tools): add DA3NestedONNX pipeline (anyview + metric + align)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: migrate `infer_onnx_nested.py` onto `DA3NestedONNX`

**Files:**
- Modify: `tools/infer_onnx_nested.py`

**Interfaces:**
- Consumes: `DA3NestedONNX` (Task 4), `astribot_dataloader` (`CAMERA_SETS`, `count_frames`, `load_images_cam_params`).
- Produces: unchanged CLI + `result.npz` output (`depth`, `depth_conf`, `extrinsics`, `intrinsics`).

- [ ] **Step 1: Capture a golden baseline BEFORE editing**

Run the current script and stash its output for later comparison:

```bash
python tools/infer_onnx_nested.py --camera-set set1 --frame 0 \
    --export-dir /tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad/nested_baseline
ls /tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad/nested_baseline/result.npz
```

Expected: `result.npz` written under the scratchpad `nested_baseline/` dir.

- [ ] **Step 2: Rewrite the script body**

Replace the entire contents of `tools/infer_onnx_nested.py` with the thin driver below. This removes `NestedONNXInference`, `preprocess_views`, `_normalize_extrinsics_numpy`, `_map_av_keys`, `_extract_metric`, and the `_MEAN`/`_STD` constants (all now in the base/model classes), keeping the CLI, data loop, and saving identical.

```python
#!/usr/bin/env python3
"""
Run nested DepthAnything3 inference via split ONNX models on the Astribot dataset.

Combines an any-view ONNX model with a metric ONNX model (``DA3NestedONNX``),
aligns their outputs, and saves results matching ``infer_pytorch.py``.

Usage:
    # Single frame
    python tools/infer_onnx_nested.py --camera-set set1 --frame 0

    # All frames
    python tools/infer_onnx_nested.py --camera-set set1 --all-frames

    # Custom ONNX paths
    python tools/infer_onnx_nested.py --camera-set set1 --frame 0 \\
        --onnx-anyview weights/da3_anyview_n3_644x490.onnx \\
        --onnx-metric  weights/da3_metric_644x490.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure tools/ is on sys.path for astribot_dataloader and model.* imports
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))

from astribot_dataloader import CAMERA_SETS, count_frames, load_images_cam_params  # noqa: E402
from model.da3nested_onnx import DA3NestedONNX  # noqa: E402

DEFAULT_ANYVIEW_ONNX = "weights/da3_anyview_n3_644x490_giant-large-1.1.onnx"
DEFAULT_METRIC_ONNX = "weights/da3_metric_644x490_giant-large-1.1.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested DepthAnything3 ONNX inference on Astribot stereo dataset",
    )
    parser.add_argument(
        "--camera-set", choices=["set1", "set2"], default="set1",
        help="set1 = head_rgbd+stereo_left+stereo_right (3 views), "
             "set2 = set1+torso_rgbd (4 views).",
    )
    parser.add_argument(
        "--frame", type=int, default=None,
        help="Single frame index (0-based).  Use --all-frames to process all.",
    )
    parser.add_argument(
        "--all-frames", action="store_true",
        help="Process all frames common to the selected cameras.",
    )
    parser.add_argument(
        "--onnx-anyview", type=str, default=DEFAULT_ANYVIEW_ONNX,
        help="Path to any-view ONNX model.",
    )
    parser.add_argument(
        "--onnx-metric", type=str, default=DEFAULT_METRIC_ONNX,
        help="Path to metric ONNX model.",
    )
    parser.add_argument(
        "--export-dir", default="output_onnx",
        help="Directory to save results (NPZ per frame).",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="ONNX Runtime device.",
    )
    parser.add_argument(
        "--no-align-input-ext-scale", dest="align_input_ext_scale",
        action="store_false",
        help="Disable Umeyama alignment of the prediction to the input camera "
             "poses (on by default, matching DepthAnything3.inference).",
    )
    parser.set_defaults(align_input_ext_scale=True)
    args = parser.parse_args()

    # --- Determine frame range ---
    camera_set = CAMERA_SETS[args.camera_set]
    if args.all_frames:
        frame_counts = {key: count_frames(key) for key in camera_set}
        num_frames = min(frame_counts.values())
        if num_frames == 0:
            raise RuntimeError("No frames found.")
        print(f"Frames per camera: {frame_counts}")
        print(f"Processing all {num_frames} common frames.")
        frame_indices = list(range(num_frames))
    elif args.frame is not None:
        frame_indices = [args.frame]
    else:
        parser.error("Specify --frame N or --all-frames.")

    # --- Load nested ONNX pipeline ---
    pipeline = DA3NestedONNX(args.onnx_anyview, args.onnx_metric, device=args.device)

    # --- Per-frame inference ---
    for frame_idx in frame_indices:
        print(f"\n--- Frame {frame_idx} ---")
        images, exts, ixts = load_images_cam_params(args.camera_set, frame_idx)
        print(f"  Views: {len(images)}")
        for p in images:
            print(f"    {p}")

        result = pipeline.infer(
            images, exts, ixts,
            align_input_ext_scale=args.align_input_ext_scale,
        )

        print(f"  depth shape:      {result['depth'].shape}")
        print(f"  depth_conf shape: {result['depth_conf'].shape}")
        print(f"  extrinsics out:   {result['extrinsics'].shape}")
        print(f"  intrinsics out:   {result['intrinsics'].shape}")

        # --- Save ---
        out_dir = Path(args.export_dir)
        if len(frame_indices) > 1:
            out_dir = out_dir / f"frame_{frame_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_dir / "result.npz",
            depth=result["depth"].astype(np.float32),
            depth_conf=result["depth_conf"].astype(np.float32),
            extrinsics=result["extrinsics"].astype(np.float32),
            intrinsics=result["intrinsics"].astype(np.float32),
        )
        print(f"  Saved: {out_dir / 'result.npz'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the migrated script**

```bash
python tools/infer_onnx_nested.py --camera-set set1 --frame 0 \
    --export-dir /tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad/nested_new
```

Expected: runs to `Done.` and writes `nested_new/result.npz`.

- [ ] **Step 4: Golden-output diff (baseline vs migrated)**

```bash
python -c "
import numpy as np
base = '/tmp/claude-1001/-home-chuong-workspace-depth-models-Depth-Anything-3/f7d776ca-2d2c-4061-8bfd-3adfb9391b67/scratchpad'
a = np.load(base + '/nested_baseline/result.npz')
b = np.load(base + '/nested_new/result.npz')
for k in ['depth','depth_conf','extrinsics','intrinsics']:
    md = float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))))
    print(f'{k:12s} max_abs_diff={md:.3e}  shape={a[k].shape}')
    assert md < 1e-4, f'{k} diverged: {md}'
print('GOLDEN MATCH')
"
```

Expected: each key `max_abs_diff` < 1e-4 and `GOLDEN MATCH`. (The `align_to_input` Umeyama step uses `random_state=42`, so results are deterministic.)

- [ ] **Step 5: Lint the changed file**

```bash
black --line-length 99 tools/infer_onnx_nested.py tools/model/base_onnx.py tools/model/base_da3.py tools/model/da3anyview_onnx.py tools/model/da3metric_onnx.py tools/model/da3nested_onnx.py
flake8 --max-line-length 100 tools/infer_onnx_nested.py tools/model/base_onnx.py tools/model/base_da3.py tools/model/da3anyview_onnx.py tools/model/da3metric_onnx.py tools/model/da3nested_onnx.py
```

Expected: black reports "unchanged" or reformats cleanly; flake8 prints nothing.

- [ ] **Step 6: Commit**

```bash
git add tools/infer_onnx_nested.py
git commit -m "refactor(tools): migrate infer_onnx_nested onto DA3NestedONNX

Removes the embedded NestedONNXInference pipeline and its duplicated
preprocessing / extrinsics-norm / key-mapping helpers in favour of the new
DA3NestedONNX class. Output result.npz verified byte-identical (max_abs_diff
< 1e-4) against the pre-refactor baseline on set1/frame 0.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Import style:** scripts are run from the repo root with `tools/` on `sys.path` — always `from model.X import Y`, never `from tools.model...`. The `# noqa: E402` on the post-`sys.path.insert` imports is intentional.
- **Do not add confidence masking** to any ONNX output. PyTorch does not mask depth by confidence; only sky regions are set to max depth (already inside `align_anyview_with_metric`).
- **Determinism:** the golden diff relies on `align_to_input_ext_scale(..., random_state=42)` and small (<100k) tensors avoiding the sampling branch. If a diff shows only the depth *scale* differing after `align_to_input`, confirm `random_state` is threaded (it is, inside `alignment.py`).
- **`weights/*.onnx.data`:** the models use external weights; ORT resolves the sidecar `.data` automatically as long as the `.onnx` and `.data` stay co-located.
```
