from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- local imports (same directory) ----
from storm import TinyTransformerHAR, TinyTransformerHARConfig
from train import (
    NpzSequenceDataset,
    evaluate,
    _set_seed,
    _confusion_matrix,
    _macro_f1_from_cm,
    FocalLoss,
)


# =====================================================================
# Constants
# =====================================================================

UNIFIED_LABELS = [
    "walking", "running", "upstairs", "downstairs",
    "sitting", "standing", "lying", "other",
]
INDIVIDUAL_DATASETS = ["uci", "motionsense", "pamap2"]

SUBJECT_OFFSETS = {"uci": 1000, "motionsense": 2000, "pamap2": 3000}

_BASE_TRAIN_ARGS: Dict[str, Any] = {
    # ---- training basics ----
    "batch": 256,
    "lr": 6e-4,
    "weight_decay": 8e-4,
    "seed": 0,
    "num_workers": 4,
    "grad_clip": 1.0,
    "scheduler": "onecycle",
    "warmup_epochs": 15,
    "early_stop": 15,
    "ema_decay": 0.999,

    # ---- loss & class balancing ----
    "loss": "ce",
    "focal_gamma": 2.0,
    "label_smoothing": 0.05,
    "class_weight": "none",
    "sampler": "weighted",
    "metric": "val_quant_macro_f1",

    # ---- augmentation ----
    "jitter": 0.005,
    "scale": 0.08,
    "time_mask": 0.10,
    "time_warp": 0.03,
    "p_drop_gyro": 0.30,
    "p_drop_acc": 0.03,
    "p_drop_axis": 0.15,
    "mixup_alpha": 0.0,
    "cutmix_alpha": 0.0,
    "mix_prob": 0.5,

    # ---- regularization ----
    "drop_path": 0.0,
    "feat_dropout": 0.0,
    "sam": True,
    "sam_rho": 0.05,
    "rdrop_alpha": 0.0,

    # ---- quantization ----
    "amp": True,
    "iqat": True,
    "eval_quant": True,
    "qat": True,
    "iqat_percentile": 99.8,
    "iqat_scale_jitter": 0.1,
    "qat_momentum": 0.93,
    "qat_lr_mult": 0.1,

    # ---- deploy simulation ----
    "deploy_sim": "periodic",
    "deploy_sim_every": 12,
    "deploy_sim_last_epochs": 3,

    # ---- self-distillation ----
    "self_distill_epochs": 0,
    "self_distill_temp": 3.0,
    "self_distill_alpha": 0.5,
    "self_distill_lr_mult": 0.3,

    # ---- test-time augmentation ----
    "tta": 0,
    "tta_jitter": 0.005,
    "tta_scale": 0.03,

    # ---- architecture ----
    "in_ch": 6,
    "d_model": 32,
    "nhead": 2,
    "depth": 2,
    "ffn_mult": 2,
    "num_classes": 8,
    "attn_window": 64,
    "int_ln": True,
    "int_ln_lut_size": 256,
}
_SINGLE_DS_COMMON: Dict[str, Any] = {
    "qat": False,
    "eval_quant": False,
    "deploy_sim": "off",
    "metric": "val_macro_f1",
    "warmup_epochs": 5,
    "early_stop": 20,
    "label_smoothing": 0.05,
    "p_drop_gyro": 0.10,
    "p_drop_acc": 0.0,
    "p_drop_axis": 0.05,
    "jitter": 0.008,
    "scale": 0.10,
    "time_warp": 0.04,
    "ema_decay": 0.999,
}

_SINGLE_DS_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "uci": {
        # ~7 k windows – large enough for batch=128, mild dropout OK
        "batch": 128,
    },
    "motionsense": {
        # ~3 k windows – batch=64 gives ~47 steps/epoch
        "batch": 64,
        "p_drop_gyro": 0.08,
        "p_drop_axis": 0.05,
    },
    "pamap2": {
        # ~700 windows after 'other' removal – needs smallest batch
        "batch": 32,
        "p_drop_gyro": 0.05,
        "p_drop_axis": 0.03,
        "early_stop": 25,
        "weight_decay": 5e-3,
        "label_smoothing": 0.03,
    },
}

_BASELINE_COMMON: Dict[str, Any] = {
    "qat": False,
    "iqat": False,
    "eval_quant": False,
    "deploy_sim": "off",
    "int_ln": False,
    "amp": True,
    "metric": "val_macro_f1",
    "warmup_epochs": 10,
    "early_stop": 20,
    "label_smoothing": 0.05,
    "p_drop_gyro": 0.0,
    "p_drop_acc": 0.0,
    "p_drop_axis": 0.0,
    "jitter": 0.005,
    "scale": 0.08,
    "time_mask": 0.10,
    "time_warp": 0.03,
    "sam": True,
    "ema_decay": 0.999,
}

BASELINE_MODELS = ["cnn1d", "lstm", "transformer"]


# =====================================================================
# Helpers
# =====================================================================

def _ts() -> str:
    return time.strftime("%H:%M:%S")

_STORE_FALSE_FLAGS = {"amp", "iqat", "eval_quant", "qat", "int_ln"}
_STORE_TRUE_FLAGS = {"sam"}
_SKIP_KEYS = {"epochs", "device"}


def _run_train_py(
    train_py: Path,
    *,
    train_npz: Path,
    val_npz: Path,
    test_npz: Optional[Path],
    out_pt: Path,
    epochs: int,
    device: str,
    extra_args: Dict[str, Any],
) -> dict:
    args = dict(_BASE_TRAIN_ARGS)
    args.update(extra_args)

    cmd = [
        sys.executable, str(train_py),
        "--train", str(train_npz),
        "--val",   str(val_npz),
        "--out",   str(out_pt),
        "--epochs", str(epochs),
        "--device", device,
    ]
    if test_npz is not None and test_npz.exists():
        cmd += ["--test", str(test_npz)]

    for k, v in args.items():
        if k in _SKIP_KEYS:
            continue
        flag = f"--{k.replace('_', '-')}"

        if k in _STORE_FALSE_FLAGS:
            if not v:
                cmd.append(flag)
            continue

        if k in _STORE_TRUE_FLAGS:
            if v:
                cmd.append(flag)
            continue

        cmd += [flag, str(v)]

    print(f"  [{_ts()}] running: {' '.join(cmd[:6])} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] train.py returned {result.returncode}")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return {"error": result.stderr[-2000:], "returncode": result.returncode, "time_s": dt}

    metrics_path = out_pt.with_suffix(".metrics.json")
    if not metrics_path.exists():
        return {"error": "metrics file not found", "time_s": dt}

    metrics = json.loads(metrics_path.read_text())
    metrics["time_s"] = dt
    return metrics


def _load_model(ckpt_path: Path, device: str) -> Tuple[TinyTransformerHAR, TinyTransformerHARConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", ckpt.get("config", {}))
    if isinstance(cfg_dict, TinyTransformerHARConfig):
        cfg = cfg_dict
    else:
        cfg = TinyTransformerHARConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in TinyTransformerHARConfig.__dataclass_fields__
        })
    model = TinyTransformerHAR(cfg)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def _eval_with_transform(
    model: nn.Module,
    test_npz: Path,
    device: str,
    num_classes: int,
    transform_fn=None,
    *,
    batch_size: int = 512,
    seed: int = 0,
) -> Dict[str, Any]:
    ds = NpzSequenceDataset(test_npz, train=False, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    losses: List[float] = []
    ys: List[int] = []
    ps: List[int] = []
    criterion = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if transform_fn is not None:
            xb = transform_fn(xb).to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        pred = logits.argmax(dim=-1)
        ys.extend(yb.cpu().tolist())
        ps.extend(pred.cpu().tolist())

    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(ps, dtype=np.int64)
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_f1 = _macro_f1_from_cm(cm)
    per_class = (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).astype(np.float64)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_acc": per_class.tolist(),
        "confusion_matrix": cm.tolist(),
        "n": int(y_true.size),
    }


# =====================================================================
# Part A – per-dataset split helpers
# =====================================================================

def _ensure_single_dataset_splits(
    dataset_root: Path,
    ds_name: str,
    create_dataset_py: Path,
    *,
    seed: int = 42,
) -> Path:
    out_dir = dataset_root / f"{ds_name}_single" / "unified"
    if (out_dir / "train.npz").exists() and (out_dir / "test.npz").exists():
        return out_dir

    print(f"  [{_ts()}] Creating single-dataset splits for '{ds_name}' ...")
    cmd = [
        sys.executable, str(create_dataset_py),
        "--out-root", str(dataset_root / f"{ds_name}_single"),
        "--datasets", ds_name,
        "--seed", str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] create_dataset.py for {ds_name} failed: {result.stderr[-1000:]}")
    return out_dir


def _filter_other_from_dir(ds_dir: Path) -> Tuple[Path, List[str], int]:
    meta_file = ds_dir / "meta.json"
    if not meta_file.exists():
        meta_file = ds_dir.parent / "meta.json"
    if not meta_file.exists():
        return ds_dir, UNIFIED_LABELS, len(UNIFIED_LABELS)

    meta = json.loads(meta_file.read_text())
    labels: List[str] = meta.get("labels", UNIFIED_LABELS)

    other_idx: Optional[int] = None
    for i, name in enumerate(labels):
        if name.lower() == "other":
            other_idx = i
            break
    if other_idx is None:
        return ds_dir, labels, len(labels)

    new_labels = [l for i, l in enumerate(labels) if i != other_idx]
    old_to_new: Dict[int, int] = {}
    new_id = 0
    for i in range(len(labels)):
        if i == other_idx:
            continue
        old_to_new[i] = new_id
        new_id += 1

    out_dir = ds_dir.parent / "no_other"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_needed = [s for s in ("train", "val", "test") if (ds_dir / f"{s}.npz").exists()]
    if all((out_dir / f"{s}.npz").exists() for s in splits_needed):
        print(f"    'other' already filtered → {out_dir}")
        return out_dir, new_labels, len(new_labels)

    print(f"    Filtering 'other' class (idx={other_idx}) from splits ...")
    for split in splits_needed:
        src = ds_dir / f"{split}.npz"
        dst = out_dir / f"{split}.npz"
        data = np.load(src, allow_pickle=False)
        X = (data["X"] if "X" in data.files else data["x"]).astype(np.float32)
        y = (data["y"] if "y" in data.files else data["Y"]).astype(np.int64)
        subj = (data["subj"].astype(np.int64)
                if "subj" in data.files
                else np.zeros(X.shape[0], dtype=np.int64))

        mask = y != other_idx
        X_f = X[mask]
        y_f = np.vectorize(lambda t: old_to_new[int(t)])(y[mask]).astype(np.int64)
        s_f = subj[mask]

        meta_str = json.dumps({"labels": new_labels, "filtered": "other_removed"})
        np.savez_compressed(dst, X=X_f, y=y_f, subj=s_f, meta=meta_str)

        n_removed = int(mask.size - mask.sum())
        print(f"      [{split}] {int(mask.sum())}/{mask.size} kept "
              f"({n_removed} 'other' removed, {len(new_labels)} classes)")

    return out_dir, new_labels, len(new_labels)


# =====================================================================
# Part B – channel-masked evaluation helpers
# =====================================================================

def _make_zero_channels_transform(channels_to_zero: List[int]):
    """Return a transform that zeros the given channel indices (dim=-1)."""
    def transform(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for ch in channels_to_zero:
            if ch < x.shape[-1]:
                x[..., ch] = 0.0
        return x
    return transform


def _make_noise_transform(channels: List[int], sigma: float, seed: int = 42):
    """Add Gaussian noise to specified channels."""
    gen = torch.Generator()
    gen.manual_seed(seed)

    def transform(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for ch in channels:
            if ch < x.shape[-1]:
                noise = torch.randn(x[..., ch].shape, generator=gen).to(x.device) * sigma
                x[..., ch] = x[..., ch] + noise
        return x
    return transform


def _make_intermittent_dropout_transform(
    channels: List[int],
    drop_frac: float = 0.30,
    seed: int = 42,
):
    gen = torch.Generator()
    gen.manual_seed(seed)

    def transform(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        B, T, C = x.shape
        mask = torch.rand(B, T, generator=gen).to(x.device) < drop_frac
        for ch in channels:
            if ch < C:
                x[..., ch] = x[..., ch].masked_fill(mask, 0.0)
        return x
    return transform


def _make_stuck_value_transform(channels: List[int], seed: int = 42):
    rng = np.random.RandomState(seed)

    def transform(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        B, T, C = x.shape
        for b in range(B):
            t_freeze = rng.randint(T // 4, 3 * T // 4)
            for ch in channels:
                if ch < C:
                    x[b, t_freeze:, ch] = x[b, t_freeze, ch]
        return x
    return transform


def _make_scale_drift_transform(channels: List[int], drift_factor: float = 2.0, seed: int = 42):
    def transform(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        B, T, C = x.shape
        ramp = torch.linspace(1.0, drift_factor, T, device=x.device).unsqueeze(0).unsqueeze(-1)
        for ch in channels:
            if ch < C:
                x[..., ch:ch + 1] = x[..., ch:ch + 1] * ramp
        return x
    return transform


# =====================================================================
# Part C – sensor-failure test definitions
# =====================================================================

def build_failure_tests(seed: int = 42) -> List[Dict[str, Any]]:
    """Return a list of failure-test definitions."""
    ACC_CH  = [0, 1, 2]
    GYRO_CH = [3, 4, 5]
    ALL_CH  = ACC_CH + GYRO_CH

    tests: List[Dict[str, Any]] = []

    # --- Baseline (no failure) ---
    tests.append({
        "name": "baseline_no_failure",
        "description": "No failure – baseline reference",
        "transform": None,
    })

    # --- Total sensor failures ---
    tests.append({
        "name": "gyro_total_failure",
        "description": "Gyroscope total failure (all gyro channels zeroed)",
        "transform": _make_zero_channels_transform(GYRO_CH),
    })
    tests.append({
        "name": "acc_total_failure",
        "description": "Accelerometer total failure (all acc channels zeroed)",
        "transform": _make_zero_channels_transform(ACC_CH),
    })
    tests.append({
        "name": "all_sensors_failure",
        "description": "Total failure – all channels zeroed (sanity: should ≈ random)",
        "transform": _make_zero_channels_transform(ALL_CH),
    })

    # --- Single-axis failures ---
    axis_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for ch_idx, name in enumerate(axis_names):
        tests.append({
            "name": f"single_axis_failure_{name}",
            "description": f"Single axis failure: {name} zeroed",
            "transform": _make_zero_channels_transform([ch_idx]),
        })

    # --- Additive noise (sensor degradation) ---
    for sigma in [0.5, 1.0, 2.0]:
        tests.append({
            "name": f"gyro_noise_sigma{sigma}",
            "description": f"Additive Gaussian noise on gyro (σ={sigma})",
            "transform": _make_noise_transform(GYRO_CH, sigma=sigma, seed=seed),
        })
        tests.append({
            "name": f"acc_noise_sigma{sigma}",
            "description": f"Additive Gaussian noise on acc (σ={sigma})",
            "transform": _make_noise_transform(ACC_CH, sigma=sigma, seed=seed),
        })
        tests.append({
            "name": f"all_noise_sigma{sigma}",
            "description": f"Additive Gaussian noise on all channels (σ={sigma})",
            "transform": _make_noise_transform(ALL_CH, sigma=sigma, seed=seed),
        })

    # --- Intermittent dropout (connection loss) ---
    for frac in [0.10, 0.30, 0.50]:
        pct = int(frac * 100)
        tests.append({
            "name": f"gyro_intermittent_{pct}pct",
            "description": f"Intermittent gyro dropout ({pct}% timesteps zeroed)",
            "transform": _make_intermittent_dropout_transform(GYRO_CH, drop_frac=frac, seed=seed),
        })
        tests.append({
            "name": f"acc_intermittent_{pct}pct",
            "description": f"Intermittent acc dropout ({pct}% timesteps zeroed)",
            "transform": _make_intermittent_dropout_transform(ACC_CH, drop_frac=frac, seed=seed),
        })

    # --- Stuck-at / frozen sensor ---
    tests.append({
        "name": "gyro_stuck_value",
        "description": "Gyroscope stuck at last good value (~50% of window frozen)",
        "transform": _make_stuck_value_transform(GYRO_CH, seed=seed),
    })
    tests.append({
        "name": "acc_stuck_value",
        "description": "Accelerometer stuck at last good value (~50% of window frozen)",
        "transform": _make_stuck_value_transform(ACC_CH, seed=seed),
    })

    # --- Gain / scale drift ---
    for drift in [1.5, 2.0, 3.0]:
        tests.append({
            "name": f"acc_scale_drift_{drift}x",
            "description": f"Acc gain drift: linear ramp from 1.0× to {drift}× over window",
            "transform": _make_scale_drift_transform(ACC_CH, drift_factor=drift, seed=seed),
        })
        tests.append({
            "name": f"gyro_scale_drift_{drift}x",
            "description": f"Gyro gain drift: linear ramp from 1.0× to {drift}× over window",
            "transform": _make_scale_drift_transform(GYRO_CH, drift_factor=drift, seed=seed),
        })

    return tests


# =====================================================================
# Report formatting
# =====================================================================

def _print_section(title: str, char: str = "="):
    w = 80
    print()
    print(char * w)
    print(f"  {title}")
    print(char * w)


def _print_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str, int]]):
    """Print a simple ASCII table.  columns: [(key, header, width), ...]"""
    header = "".join(f"{hdr:<{w}}" for _, hdr, w in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = ""
        for key, _hdr, w in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            line += f"{str(val):<{w}}"
        print(line)


def print_report(report: Dict[str, Any]):

    # --- Part A ---
    if report.get("individual_datasets"):
        _print_section("A) Individual Datasets (no harmonization)")
        rows = []
        for ds_name, info in report["individual_datasets"].items():
            if "error" in info:
                rows.append({"dataset": ds_name, "acc": "ERROR", "macro_f1": "ERROR", "n": ""})
                continue
            ft = info.get("final_test") or info.get("final_val", {})
            rows.append({
                "dataset": ds_name,
                "acc": ft.get("acc", 0.0),
                "macro_f1": ft.get("macro_f1", 0.0),
                "n": ft.get("n", "?"),
                "time_s": f"{info.get('time_s', 0):.0f}",
            })
        _print_table(rows, [
            ("dataset",  "Dataset",  20),
            ("acc",      "Accuracy", 14),
            ("macro_f1", "Macro-F1", 14),
            ("n",        "N_test",   10),
            ("time_s",   "Time(s)",  10),
        ])

    # --- Part B ---
    if report.get("sensor_subsets"):
        _print_section("B) Unified Dataset – Sensor Subsets")
        rows = []
        for cfg_name, info in report["sensor_subsets"].items():
            if "error" in info:
                rows.append({"config": cfg_name, "acc": "ERROR", "macro_f1": "ERROR"})
                continue
            rows.append({
                "config":   cfg_name,
                "acc":      info.get("acc", 0.0),
                "macro_f1": info.get("macro_f1", 0.0),
                "n":        info.get("n", "?"),
            })
        _print_table(rows, [
            ("config",   "Configuration", 30),
            ("acc",      "Accuracy",      14),
            ("macro_f1", "Macro-F1",      14),
            ("n",        "N_test",        10),
        ])

    # --- Part C ---
    if report.get("failure_tests"):
        _print_section("C) Sensor Failure Robustness Tests")
        rows = []
        baseline_f1: Optional[float] = None
        for test_name, info in report["failure_tests"].items():
            if "error" in info:
                rows.append({"test": test_name, "acc": "ERROR", "macro_f1": "ERROR"})
                continue
            f1 = info.get("macro_f1", 0.0)
            if test_name == "baseline_no_failure":
                baseline_f1 = f1
            delta = ""
            if baseline_f1 is not None and test_name != "baseline_no_failure":
                delta = f"{f1 - baseline_f1:+.4f}"
            rows.append({
                "test":     test_name,
                "acc":      info.get("acc", 0.0),
                "macro_f1": f1,
                "delta_f1": delta,
                "n":        info.get("n", "?"),
            })
        _print_table(rows, [
            ("test",     "Test",     36),
            ("acc",      "Accuracy", 12),
            ("macro_f1", "Macro-F1", 12),
            ("delta_f1", "ΔF1",      12),
            ("n",        "N",         8),
        ])

    # --- Part D ---
    if report.get("baselines"):
        _print_section("D) Baseline Models (FP32, ~20 k params)")
        rows = []
        for model_name, info in report["baselines"].items():
            if "error" in info:
                rows.append({"model": model_name, "acc": "ERROR", "macro_f1": "ERROR"})
                continue
            ft = info.get("final_test") or info.get("final_val", {})
            rows.append({
                "model":    model_name,
                "acc":      ft.get("acc", 0.0),
                "macro_f1": ft.get("macro_f1", 0.0),
                "n":        ft.get("n", "?"),
                "time_s":   f"{info.get('time_s', 0):.0f}",
            })
        _print_table(rows, [
            ("model",    "Model",    20),
            ("acc",      "Accuracy", 14),
            ("macro_f1", "Macro-F1", 14),
            ("n",        "N_test",   10),
            ("time_s",   "Time(s)",  10),
        ])


def _maybe_plot_report(report: Dict[str, Any], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot")
        return

    names: List[str] = []
    f1s:   List[float] = []

    if report.get("individual_datasets"):
        for ds_name, info in report["individual_datasets"].items():
            ft = info.get("final_test") or info.get("final_val", {})
            names.append(f"[A] {ds_name}")
            f1s.append(ft.get("macro_f1", 0.0))

    if report.get("sensor_subsets"):
        for cfg_name, info in report["sensor_subsets"].items():
            names.append(f"[B] {cfg_name}")
            f1s.append(info.get("macro_f1", 0.0))

    if report.get("failure_tests"):
        key_tests = [
            "baseline_no_failure",
            "gyro_total_failure", "acc_total_failure", "all_sensors_failure",
            "gyro_noise_sigma1.0", "acc_noise_sigma1.0",
            "gyro_intermittent_30pct", "acc_intermittent_30pct",
            "gyro_stuck_value", "acc_stuck_value",
            "acc_scale_drift_2.0x", "gyro_scale_drift_2.0x",
        ]
        for t_name in key_tests:
            if t_name in report["failure_tests"]:
                info = report["failure_tests"][t_name]
                names.append(f"[C] {t_name}")
                f1s.append(info.get("macro_f1", 0.0))

    if report.get("baselines"):
        for model_name, info in report["baselines"].items():
            ft = info.get("final_test") or info.get("final_val", {})
            names.append(f"[D] {model_name}")
            f1s.append(ft.get("macro_f1", 0.0))

    if not names:
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    colors = []
    for n in names:
        if n.startswith("[A]"):
            colors.append("#4C72B0")
        elif n.startswith("[B]"):
            colors.append("#55A868")
        elif n.startswith("[D]"):
            colors.append("#8172B2")
        else:
            colors.append("#C44E52")

    bars = ax.barh(y_pos, f1s, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Benchmark: Dataset Configurations & Sensor Failure Robustness")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# =====================================================================
# Utility helpers
# =====================================================================

def _extract_test_or_val(metrics: dict) -> dict:
    """Extract test (or val) metrics from a train.py output dict."""
    if "error" in metrics:
        return metrics
    ft = metrics.get("final_test")
    if ft is not None:
        return ft
    fv = metrics.get("final_val")
    if fv is not None:
        return fv
    return {"error": "no final_test or final_val in metrics"}


def _print_individual_summary(ds_name: str, metrics: dict):
    if "error" in metrics:
        print(f"  {ds_name}: ERROR – {metrics.get('error', '?')[:200]}")
        return
    ft = metrics.get("final_test") or metrics.get("final_val", {})
    print(f"  {ds_name}: acc={ft.get('acc', 0):.4f}  "
          f"macro_f1={ft.get('macro_f1', 0):.4f}  "
          f"N={ft.get('n', '?')}  "
          f"time={metrics.get('time_s', 0):.0f}s")


def _make_json_serializable(obj):
    """Recursively convert numpy types and other non-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Benchmark dataset configurations and sensor-failure robustness for TinyTransformerHAR")
    ap.add_argument("--dataset-root", type=Path, required=True, help="Root containing unified/ (and optionally raw data for single-dataset creation)")
    ap.add_argument("--workdir", type=Path, required=True, help="Working directory for checkpoints and outputs")
    ap.add_argument("--epochs", type=int, default=40, help="Training epochs per experiment (lower = faster benchmark)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="Override batch size (0 = use per-experiment defaults)")
    ap.add_argument("--skip-training", action="store_true", help="Skip Part A & B training; run only failure tests (requires --unified-ckpt)")
    ap.add_argument("--unified-ckpt", type=Path, default=None, help="Pre-trained unified full checkpoint for failure tests without re-training")
    ap.add_argument("--skip-individual", action="store_true", help="Skip Part A (individual datasets)")
    ap.add_argument("--skip-subsets", action="store_true", help="Skip Part B (sensor subsets)")
    ap.add_argument("--skip-failures", action="store_true", help="Skip Part C (failure tests)")
    ap.add_argument("--skip-baselines", action="store_true", help="Skip Part D (baseline model comparisons)")
    ap.add_argument("--num-workers", type=int, default=4)

    args = ap.parse_args()

    _set_seed(args.seed)
    args.workdir.mkdir(parents=True, exist_ok=True)

    train_py          = Path(__file__).resolve().parent / "train.py"
    create_dataset_py = Path(__file__).resolve().parent / "create_dataset.py"

    dataset_root = args.dataset_root.resolve()
    unified_dir  = dataset_root / "unified"

    if not unified_dir.exists():
        print(f"[ERROR] Unified dataset not found at {unified_dir}")
        print("Run create_dataset.py first, or point --dataset-root to the right location.")
        sys.exit(1)

    train_npz = unified_dir / "train.npz"
    val_npz   = unified_dir / "val.npz"
    test_npz  = unified_dir / "test.npz"

    meta_path    = unified_dir / "meta.json"
    unified_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    num_classes  = len(unified_meta.get("labels", UNIFIED_LABELS))

    report: Dict[str, Any] = {
        "timestamp":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root":        str(dataset_root),
        "epochs":              args.epochs,
        "device":              args.device,
        "seed":                args.seed,
        "unified_meta_labels": unified_meta.get("labels", UNIFIED_LABELS),
        "individual_datasets": {},
        "sensor_subsets":      {},
        "failure_tests":       {},
        "baselines":           {},
    }

    # =================================================================
    # Part A: Individual datasets
    # =================================================================
    if not args.skip_training and not args.skip_individual:
        _print_section("A) Training on individual datasets")
        for ds_name in INDIVIDUAL_DATASETS:
            print(f"\n--- {ds_name} ---")
            ds_dir = _ensure_single_dataset_splits(
                dataset_root, ds_name, create_dataset_py, seed=args.seed + 1
            )
            ds_dir, ds_labels, ds_num_classes = _filter_other_from_dir(ds_dir)

            ds_train = ds_dir / "train.npz"
            ds_val   = ds_dir / "val.npz"
            ds_test  = ds_dir / "test.npz"

            if not ds_train.exists():
                print(f"  [SKIP] train split not found for {ds_name}")
                report["individual_datasets"][ds_name] = {"error": "train split not found"}
                continue

            out_dir = args.workdir / f"individual_{ds_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_pt = out_dir / "best.pt"

            extra: Dict[str, Any] = {"num_classes": ds_num_classes}
            extra.update(_SINGLE_DS_COMMON)
            extra.update(_SINGLE_DS_OVERRIDES.get(ds_name, {}))
            if args.batch > 0:
                extra["batch"] = args.batch

            metrics = _run_train_py(
                train_py,
                train_npz=ds_train,
                val_npz=ds_val,
                test_npz=ds_test if ds_test.exists() else None,
                out_pt=out_pt,
                epochs=args.epochs,
                device=args.device,
                extra_args=extra,
            )
            report["individual_datasets"][ds_name] = metrics
            _print_individual_summary(ds_name, metrics)

    # =================================================================
    # Part B: Sensor subsets on unified dataset
    # =================================================================
    unified_ckpt_path = args.unified_ckpt

    if not args.skip_training and not args.skip_subsets:
        _print_section("B) Unified dataset – sensor subset evaluation")

        # B1: Full model (acc + gyro, 6 ch)
        print("\n--- unified_full (acc + gyro, 6ch) ---")
        full_dir = args.workdir / "unified_full"
        full_dir.mkdir(parents=True, exist_ok=True)
        full_pt = full_dir / "best.pt"

        extra_full: Dict[str, Any] = {"in_ch": 6, "num_classes": num_classes}
        if args.batch > 0:
            extra_full["batch"] = args.batch

        full_metrics = _run_train_py(
            train_py,
            train_npz=train_npz,
            val_npz=val_npz,
            test_npz=test_npz,
            out_pt=full_pt,
            epochs=args.epochs,
            device=args.device,
            extra_args=extra_full,
        )
        report["sensor_subsets"]["unified_full_6ch"] = _extract_test_or_val(full_metrics)
        _res = report["sensor_subsets"]["unified_full_6ch"]
        print(f"  → acc={_res.get('acc', 0):.4f}  "
              f"macro_f1={_res.get('macro_f1', 0):.4f}  N={_res.get('n', '?')}")

        if full_pt.exists():
            unified_ckpt_path = full_pt

        # B2: Acc-only model trained with gyro always dropped (p_drop_gyro=1.0)
        print("\n--- unified_acc_only (6ch model, gyro zeroed during training & eval) ---")
        acc_dir = args.workdir / "unified_acc_only"
        acc_dir.mkdir(parents=True, exist_ok=True)
        acc_pt = acc_dir / "best.pt"

        extra_acc: Dict[str, Any] = {
            "in_ch": 6,
            "num_classes": num_classes,
            "p_drop_gyro": 1.0,   # always zero gyro during training
            "p_drop_acc": 0.0,
        }
        if args.batch > 0:
            extra_acc["batch"] = args.batch

        acc_metrics = _run_train_py(
            train_py,
            train_npz=train_npz,
            val_npz=val_npz,
            test_npz=test_npz,
            out_pt=acc_pt,
            epochs=args.epochs,
            device=args.device,
            extra_args=extra_acc,
        )
        if acc_pt.exists():
            model_acc, _ = _load_model(acc_pt, args.device)
            gyro_zero    = _make_zero_channels_transform([3, 4, 5])
            eval_acc_only = _eval_with_transform(
                model_acc, test_npz, args.device, num_classes,
                transform_fn=gyro_zero, seed=args.seed,
            )
            report["sensor_subsets"]["unified_acc_only_3ch"] = eval_acc_only
            print(f"  → acc={eval_acc_only['acc']:.4f}  "
                  f"macro_f1={eval_acc_only['macro_f1']:.4f}  N={eval_acc_only['n']}")
        else:
            report["sensor_subsets"]["unified_acc_only_3ch"] = _extract_test_or_val(acc_metrics)
            _res = report["sensor_subsets"]["unified_acc_only_3ch"]
            print(f"  → acc={_res.get('acc', 0):.4f}  "
                  f"macro_f1={_res.get('macro_f1', 0):.4f}  N={_res.get('n', '?')}")

        # B3: Full model evaluated with gyro zeroed at inference only
        if unified_ckpt_path and unified_ckpt_path.exists():
            print("\n--- unified_full model (eval with gyro zeroed at inference) ---")
            model_full, _ = _load_model(unified_ckpt_path, args.device)
            eval_full_acc_only = _eval_with_transform(
                model_full, test_npz, args.device, num_classes,
                transform_fn=_make_zero_channels_transform([3, 4, 5]),
                seed=args.seed,
            )
            report["sensor_subsets"]["unified_full_eval_acc_only"] = eval_full_acc_only
            print(f"  → acc={eval_full_acc_only['acc']:.4f}  "
                  f"macro_f1={eval_full_acc_only['macro_f1']:.4f}  N={eval_full_acc_only['n']}")

    elif args.skip_training and unified_ckpt_path and unified_ckpt_path.exists() \
            and not args.skip_subsets:
        _print_section("B) Sensor subset evaluation (using existing checkpoint)")
        model, cfg = _load_model(unified_ckpt_path, args.device)

        report["sensor_subsets"]["unified_full_6ch"] = _eval_with_transform(
            model, test_npz, args.device, num_classes,
            transform_fn=None, seed=args.seed,
        )
        _res = report["sensor_subsets"]["unified_full_6ch"]
        print(f"  unified_full_6ch: acc={_res['acc']:.4f}  "
              f"macro_f1={_res['macro_f1']:.4f}  N={_res['n']}")

        report["sensor_subsets"]["unified_full_eval_acc_only"] = _eval_with_transform(
            model, test_npz, args.device, num_classes,
            transform_fn=_make_zero_channels_transform([3, 4, 5]),
            seed=args.seed,
        )
        _res = report["sensor_subsets"]["unified_full_eval_acc_only"]
        print(f"  unified_full_eval_acc_only: acc={_res['acc']:.4f}  "
              f"macro_f1={_res['macro_f1']:.4f}  N={_res['n']}")

    # =================================================================
    # Part C: Sensor failure robustness tests
    # =================================================================
    if not args.skip_failures:
        _ckpt = unified_ckpt_path or (args.workdir / "unified_full" / "best.pt")
        if _ckpt and _ckpt.exists():
            _print_section("C) Sensor Failure Robustness Tests")
            model, cfg = _load_model(_ckpt, args.device)
            failure_tests = build_failure_tests(seed=args.seed)

            for i, ftest in enumerate(failure_tests):
                name = ftest["name"]
                desc = ftest["description"]
                print(f"  [{i + 1}/{len(failure_tests)}] {name}: {desc}")
                result = _eval_with_transform(
                    model, test_npz, args.device, num_classes,
                    transform_fn=ftest["transform"],
                    seed=args.seed,
                )
                report["failure_tests"][name] = result
                print(f"    acc={result['acc']:.4f}  macro_f1={result['macro_f1']:.4f}")
        else:
            print(f"[WARN] No unified checkpoint found at {_ckpt}, skipping failure tests.")
            print("  Provide --unified-ckpt or run without --skip-training.")

    # =================================================================
    # Part D: Baseline models on unified dataset (FP32)
    # =================================================================
    if not args.skip_training and not args.skip_baselines:
        _print_section("D) Baseline models on unified dataset (FP32, ~20 k params)")

        for model_name in BASELINE_MODELS:
            print(f"\n--- {model_name} ---")
            bl_dir = args.workdir / f"baseline_{model_name}"
            bl_dir.mkdir(parents=True, exist_ok=True)
            bl_pt = bl_dir / "best.pt"

            extra_bl: Dict[str, Any] = {
                "in_ch": 6,
                "num_classes": num_classes,
                "model": model_name,
            }
            extra_bl.update(_BASELINE_COMMON)
            if args.batch > 0:
                extra_bl["batch"] = args.batch

            metrics = _run_train_py(
                train_py,
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                out_pt=bl_pt,
                epochs=args.epochs,
                device=args.device,
                extra_args=extra_bl,
            )
            report["baselines"][model_name] = metrics
            _print_individual_summary(model_name, metrics)

    # =================================================================
    # Final report
    # =================================================================
    _print_section("FINAL REPORT")
    print_report(report)

    report_path = args.workdir / "benchmark_report.json"
    report_path.write_text(json.dumps(_make_json_serializable(report), indent=2) + "\n")
    print(f"\nJSON report saved: {report_path}")

    _maybe_plot_report(report, args.workdir / "benchmark_report.png")
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()