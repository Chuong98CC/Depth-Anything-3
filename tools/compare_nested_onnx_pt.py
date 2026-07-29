#!/usr/bin/env python3
"""
Compare nested PyTorch model vs ONNX any-view + ONNX metric + alignment.

Verifies that the split-ONNX pipeline (two independently exported models combined
via ``align_anyview_with_metric``) reproduces the end-to-end PyTorch nested model
output exactly.

Usage:
    python tools/compare_nested_onnx_pt.py \\
        --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \\
        --onnx-anyview weights/da3_anyview_n3_644x490_giant-large-1.1.onnx \\
        --onnx-metric  weights/da3_metric_644x490_giant-large-1.1.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from depth_anything_3.api import DepthAnything3
from model.alignment import align_anyview_with_metric


# ---------------------------------------------------------------------------
# Data loading (mirrors astribot_dataloader)
# ---------------------------------------------------------------------------

def _load_example_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load Astribot set1 / frame 0 images, extrinsics, intrinsics."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from astribot_dataloader import CAMERA_SETS, load_calib, load_camera_data  # noqa: PLC0415

    calib = load_calib(
        "/home/chuong/workspace/demo_data/astribot_camera_calib_params/"
        "astribot_calibration_full.json",
        target_res=(640, 480),
    )
    camera_set = CAMERA_SETS["set1"]
    images, exts, ixts, _, _ = load_camera_data(
        calib, camera_set, frame_idx=0, sensor_type="color",
    )
    return images, exts, ixts


# ---------------------------------------------------------------------------
# Common preprocessing (exact resize, same as any-view ONNX / TRT)
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_for_onnx(
    image_paths: list[str],
    intrs: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess N images to (1, N, 3, H, W) with scaled intrinsics."""
    import cv2  # noqa: PLC0415

    N = len(image_paths)
    proc = np.zeros((N, 3, target_h, target_w), dtype=np.float32)
    intrs_out = np.zeros((N, 3, 3), dtype=np.float32)

    for i, path in enumerate(image_paths):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_f = rgb_r.astype(np.float32) / 255.0
        proc[i] = ((img_f - _MEAN) / _STD).transpose(2, 0, 1)

        # Scale intrinsics
        K = intrs[i].copy().astype(np.float32)
        sx, sy = target_w / orig_w, target_h / orig_h
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        intrs_out[i] = K

    return proc[None], intrs_out  # add B=1


# ---------------------------------------------------------------------------
# PyTorch nested forward
# ---------------------------------------------------------------------------

def _normalize_extrinsics_torch(ex_t: torch.Tensor) -> torch.Tensor:
    """Replica of ``DepthAnything3._normalize_extrinsics``."""
    from depth_anything_3.utils.geometry import affine_inverse  # noqa: PLC0415

    transform = affine_inverse(ex_t[:, :1])
    ex_t_norm = ex_t @ transform
    c2ws = affine_inverse(ex_t_norm)
    dists = c2ws[..., :3, 3].norm(dim=-1)
    median_dist = torch.median(dists).clamp(min=1e-1)
    ex_t_norm[..., :3, 3] /= median_dist
    return ex_t_norm


def run_pytorch_nested(
    api_model: DepthAnything3,
    imgs_t: torch.Tensor,
    extrs_t: torch.Tensor,
    intrs_t: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Run the full PyTorch nested model forward + alignment."""
    ex_t_norm = _normalize_extrinsics_torch(extrs_t.clone())

    with torch.no_grad():
        raw = api_model.model(
            imgs_t,
            extrinsics=ex_t_norm,
            intrinsics=intrs_t,
            export_feat_layers=[],
            infer_gs=False,
        )
    return {k: v.float().cpu().numpy() for k, v in raw.items() if isinstance(v, torch.Tensor)}


# ---------------------------------------------------------------------------
# ONNX split pipeline (any-view + metric + alignment)
# ---------------------------------------------------------------------------

def run_onnx_split(
    onnx_anyview_path: str,
    onnx_metric_path: str,
    img_batch: np.ndarray,
    extrs: np.ndarray,
    intrs: np.ndarray,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Run any-view ONNX + metric ONNX, then align."""
    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    # --- Any-view ONNX ---
    sess_av = ort.InferenceSession(onnx_anyview_path, providers=providers)
    av_inputs = {
        "image": img_batch.astype(np.float32),
        "extrinsics": extrs[None].astype(np.float32),
        "intrinsics": intrs[None].astype(np.float32),
    }
    av_out_names = [o.name for o in sess_av.get_outputs()]
    av_raw = dict(zip(av_out_names, sess_av.run(av_out_names, av_inputs)))

    # Map any-view output names
    av = _map_onnx_keys(av_raw)

    # --- Metric ONNX (one image at a time, metric model is single-view) ---
    sess_m = ort.InferenceSession(onnx_metric_path, providers=providers)
    N, _, H, W = img_batch.shape[1:]
    metric_depths = np.zeros((1, N, 1, H, W), dtype=np.float32)
    metric_skys = np.zeros((1, N, 1, H, W), dtype=np.float32)

    # Metric ONNX input shape: (1, 3, H_m, W_m) — read from model
    m_shape = sess_m.get_inputs()[0].shape  # e.g. [1, 3, 490, 644]
    m_h = m_shape[2] if isinstance(m_shape[2], int) else H
    m_w = m_shape[3] if isinstance(m_shape[3], int) else W

    import cv2  # noqa: PLC0415

    for i in range(N):
        # Extract single-view preprocessed image, resize to metric dims
        view_img = img_batch[0, i]  # (3, H_av, W_av)
        if (H, W) != (m_h, m_w):
            view_img = cv2.resize(
                view_img.transpose(1, 2, 0), (m_w, m_h), interpolation=cv2.INTER_LINEAR,
            )
            view_img = view_img.transpose(2, 0, 1)  # back to CHW

        m_inputs = {"image": view_img[None].astype(np.float32)}  # (1, 3, H_m, W_m)
        m_out_names = [o.name for o in sess_m.get_outputs()]
        m_raw = dict(zip(m_out_names, sess_m.run(m_out_names, m_inputs)))

        # Extract depth/sky — metric model outputs (B, 1, H, W) per view
        m_depth, m_sky = _extract_metric_onnx(m_raw, i)

        if (m_h, m_w) != (H, W):
            m_depth = cv2.resize(m_depth, (W, H), interpolation=cv2.INTER_LINEAR)
            m_sky = cv2.resize(m_sky, (W, H), interpolation=cv2.INTER_LINEAR)

        metric_depths[0, i, 0] = m_depth
        metric_skys[0, i, 0] = m_sky

    # Metric model outputs shape: (B, N, 1, H, W) → squeeze channel → (B, N, H, W)
    metric_depths = metric_depths.squeeze(2)
    metric_skys = metric_skys.squeeze(2)

    # --- Alignment ---
    aligned = align_anyview_with_metric(
        anyview_depth=torch.from_numpy(av["depth"]),
        anyview_conf=torch.from_numpy(av["depth_conf"]),
        anyview_extrinsics=torch.from_numpy(av["extrinsics"]),
        anyview_intrinsics=torch.from_numpy(av["intrinsics"]),
        metric_depth=torch.from_numpy(metric_depths),
        metric_sky=torch.from_numpy(metric_skys),
    )

    return {k: v.float().cpu().numpy() for k, v in aligned.items()}


# ---------------------------------------------------------------------------
# ONNX output name mapping
# ---------------------------------------------------------------------------

def _map_onnx_keys(raw: dict) -> dict:
    """Normalise ONNX output keys to canonical names."""
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


def _extract_metric_onnx(raw: dict, view_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (depth, sky) for a specific view from metric ONNX outputs."""
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


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare(pt: dict, onx: dict, label: str) -> None:
    print(f"\n{'='*72}")
    print(f"[COMPARE] {label}")
    print(f"{'-'*72}")

    for key in ["depth", "depth_conf", "extrinsics", "intrinsics"]:
        if key not in pt or key not in onx:
            print(f"  SKIP {key}: missing from one side")
            continue
        p, o = pt[key], onx[key]
        if p.shape != o.shape:
            print(f"  WARN {key}: shape PT={list(p.shape)} ONNX={list(o.shape)}")
            # Crop to overlapping region
            mins = tuple(min(a, b) for a, b in zip(p.shape, o.shape))
            p = p[tuple(slice(0, s) for s in mins)]
            o = o[tuple(slice(0, s) for s in mins)]

        abs_diff = np.abs(p.astype(np.float64) - o.astype(np.float64))
        rel_err = abs_diff / (np.abs(p.astype(np.float64)) + 1e-8)
        valid = np.isfinite(p) & np.isfinite(o)

        print(
            f"  {key:15s} "
            f"max_abs={float(abs_diff[valid].max()):.4e}  "
            f"mean_abs={float(abs_diff[valid].mean()):.4e}  "
            f"max_rel={float(rel_err[valid].max()):.4e}  "
            f"mean_rel={float(rel_err[valid].mean()):.4e}  "
            f"shape={list(p.shape)}"
        )

    # Allclose
    print(f"{'-'*72}")
    for key, atol in [("depth", 1e-3), ("depth_conf", 1e-3)]:
        if key in pt and key in onx:
            ok = np.allclose(pt[key], onx[key], atol=atol)
            print(f"[COMPARE] {key:15s} allclose(atol={atol:.0e}): {ok}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch nested model vs split ONNX pipeline",
    )
    parser.add_argument("--model-dir", required=True, type=str,
                        help="PyTorch nested model checkpoint (HuggingFace ID or local dir).")
    parser.add_argument("--onnx-anyview", required=True, type=str,
                        help="Path to any-view ONNX model.")
    parser.add_argument("--onnx-metric", required=True, type=str,
                        help="Path to metric ONNX model.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for PyTorch / ONNX Runtime.")
    parser.add_argument("--height", type=int, default=490,
                        help="Target height (must match ONNX inputs).")
    parser.add_argument("--width", type=int, default=644,
                        help="Target width (must match ONNX inputs).")
    args = parser.parse_args()

    target_h, target_w = args.height, args.width
    dev = torch.device(args.device)

    # 1. Load example data --------------------------------------------------
    print("[DATA] Loading Astribot set1 / frame 0 ...")
    images, exts_np, ixts_np = _load_example_data()
    N = len(images)
    print(f"[DATA] {N} views loaded")

    # 2. Preprocess (exact resize, both models use identical input) ---------
    img_batch, ixts_adj = _preprocess_for_onnx(images, ixts_np, target_h, target_w)
    print(f"[PREP] Image batch: {list(img_batch.shape)}")

    # 3. PyTorch nested model -----------------------------------------------
    print(f"\n[PYTORCH] Loading nested model from {args.model_dir} ...")
    api_model = DepthAnything3.from_pretrained(args.model_dir)
    api_model = api_model.to(dev)
    api_model.eval()

    imgs_t = torch.from_numpy(img_batch).to(dev).float()
    exts_t = torch.from_numpy(exts_np[None]).to(dev).float()   # add B=1
    intrs_t = torch.from_numpy(ixts_adj[None]).to(dev).float()

    print("[PYTORCH] Running nested forward ...")
    pt_out = run_pytorch_nested(api_model, imgs_t, exts_t, intrs_t)

    # 4. ONNX split pipeline ------------------------------------------------
    print("\n[ONNX] Running any-view + metric + alignment ...")
    exts_norm_np = _normalize_extrinsics_numpy(exts_np)
    onx_out = run_onnx_split(
        args.onnx_anyview, args.onnx_metric,
        img_batch, exts_norm_np, ixts_adj,
        device=args.device,
    )

    # 5. Compare ------------------------------------------------------------
    _compare(pt_out, onx_out, "PyTorch nested  vs  ONNX anyview+metric+align")


def _normalize_extrinsics_numpy(extrs: np.ndarray) -> np.ndarray:
    """Numpy replica for ONNX path (PyTorch path normalises in-torch)."""
    ex_t = extrs.copy()
    transform = np.linalg.inv(ex_t[0])
    ex_t_norm = ex_t @ transform
    c2ws = np.linalg.inv(ex_t_norm)
    dists = np.linalg.norm(c2ws[..., :3, 3], axis=-1)
    median_dist = max(float(np.median(dists)), 1e-1)
    ex_t_norm[..., :3, 3] /= median_dist
    return ex_t_norm


if __name__ == "__main__":
    main()
