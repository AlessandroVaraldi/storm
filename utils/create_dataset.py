import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm
import requests

# ----------------------------------------------------------------------
# URLs and constants
# ----------------------------------------------------------------------

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
PAMAP_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"

# MotionSense GitHub repo zip (public, contains data/ folder)
MOTIONSENSE_ZIP_URL = "https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip"

REQ_TIMEOUT = 30

# Unified label set (semantic space)
UNIFIED_LABELS = [
    "walking", "running", "upstairs", "downstairs",
    "sitting", "standing", "lying", "other"
]
UNIFIED_L2ID = {k: i for i, k in enumerate(UNIFIED_LABELS)}

# Extended label set: all named activities (no "other" bucket)
ALL_LABELS = [
    "walking", "running", "upstairs", "downstairs",
    "sitting", "standing", "lying",
    "cycling", "nordic_walking", "vacuum_cleaning", "ironing", "rope_jumping",
]
ALL_L2ID = {k: i for i, k in enumerate(ALL_LABELS)}

# Sentinel for "drop this sample" (transient / truly unknown)
_DROP = -1


# ----------------------------------------------------------------------
# Label mapping helpers
# ----------------------------------------------------------------------

def uci_label_to_unified(label_id: int, *, all_classes: bool = False) -> int:
    """UCI HAR activities:
    1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS,
    4 SITTING, 5 STANDING, 6 LAYING.
    """
    name = {
        1: "walking",
        2: "upstairs",
        3: "downstairs",
        4: "sitting",
        5: "standing",
        6: "lying",
    }.get(int(label_id))
    if name is None:
        return _DROP if all_classes else UNIFIED_L2ID["other"]
    l2id = ALL_L2ID if all_classes else UNIFIED_L2ID
    return l2id[name]


def motionsense_label_to_unified(act: str, *, all_classes: bool = False) -> int:
    """MotionSense activity codes in A_DeviceMotion_data folder:
    dws downstairs, ups upstairs, wlk walking, jog jogging, sit sitting, std standing.
    """
    act = act.strip().lower()
    name = {
        "wlk": "walking",
        "jog": "running",
        "ups": "upstairs",
        "dws": "downstairs",
        "sit": "sitting",
        "std": "standing",
    }.get(act)
    if name is None:
        return _DROP if all_classes else UNIFIED_L2ID["other"]
    l2id = ALL_L2ID if all_classes else UNIFIED_L2ID
    return l2id[name]


def pamap_label_to_unified(code: int, *, all_classes: bool = False) -> int:
    """PAMAP2 codes (subset):
    1 lying, 2 sitting, 3 standing, 4 walking, 5 running,
    12 ascending stairs, 13 descending stairs.
    Extended (all_classes): 6 cycling, 7 Nordic walking,
    16 vacuum cleaning, 17 ironing, 24 rope jumping.
    """
    _COMMON = {
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "running",
        12: "upstairs",
        13: "downstairs",
    }
    _EXTRA = {
        6: "cycling",
        7: "nordic_walking",
        16: "vacuum_cleaning",
        17: "ironing",
        24: "rope_jumping",
    }
    code = int(code)
    name = _COMMON.get(code)
    if name is not None:
        l2id = ALL_L2ID if all_classes else UNIFIED_L2ID
        return l2id[name]
    if all_classes:
        name = _EXTRA.get(code)
        return ALL_L2ID[name] if name is not None else _DROP
    return UNIFIED_L2ID["other"]


# ----------------------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------------------

def http_get(url: str, stream: bool = False):
    return requests.get(url, timeout=REQ_TIMEOUT, stream=stream)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_py(o):
    """Convert numpy types to plain Python types for JSON."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [to_py(x) for x in o]
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    return o


def save_npz(path: Path, X: np.ndarray, y: np.ndarray, subj: np.ndarray, meta: dict):
    meta_str = json.dumps(to_py(meta))
    np.savez_compressed(path, X=X, y=y, subj=subj, meta=meta_str)


def resample_to(fs_in: float, fs_out: float, x: np.ndarray) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return x
    g = math.gcd(int(round(fs_in * 100)), int(round(fs_out * 100)))
    up = int(round(fs_out * 100)) // g
    down = int(round(fs_in * 100)) // g
    return resample_poly(x, up, down, axis=0)


def window_stream(x: np.ndarray,
                  y: np.ndarray,
                  subj: np.ndarray,
                  win: int,
                  stride: int,
                  min_purity: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xw, Yw, Sw = [], [], []
    T = int(x.shape[0])
    if T < win:
        return (np.zeros((0, win, x.shape[1]), dtype=x.dtype),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=int))

    for s in range(0, T - win + 1, stride):
        e = s + win
        seg_y_raw = y[s:e]
        # Filter out _DROP sentinel (-1) before majority vote
        valid = seg_y_raw[seg_y_raw >= 0]
        if valid.size == 0:
            continue
        counts = np.bincount(valid)
        seg_y = counts.argmax()  # majority label
        purity = counts[seg_y] / win
        if purity < min_purity:
            continue  # discard ambiguous window
        seg_s = int(subj[s]) if subj.size else -1
        Xw.append(x[s:e])
        Yw.append(seg_y)
        Sw.append(seg_s)

    if not Xw:
        return (np.zeros((0, win, x.shape[1]), dtype=x.dtype),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=int))

    return (np.stack(Xw),
            np.array(Yw, dtype=int),
            np.array(Sw, dtype=int))


def balance_classes(X: np.ndarray,
                    y: np.ndarray,
                    subj: np.ndarray,
                    max_per_class: int,
                    rng: np.random.Generator):
    if max_per_class <= 0:
        return X, y, subj
    idxs = []
    for c in np.unique(y):
        idc = np.where(y == c)[0]
        rng.shuffle(idc)
        idc = idc[:max_per_class]
        idxs.append(idc)
    if not idxs:
        return X, y, subj
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return X[idx], y[idx], subj[idx]


def cap_other_class(
    X: np.ndarray,
    y: np.ndarray,
    subj: np.ndarray,
    other_label: int,
    max_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_ratio <= 0:
        return X, y, subj

    non_other_mask = y != other_label
    other_mask = y == other_label
    n_other = int(other_mask.sum())
    if n_other == 0:
        return X, y, subj

    non_other_counts = np.bincount(y[non_other_mask])
    if non_other_counts.size == 0 or non_other_counts.max() == 0:
        return X, y, subj

    cap = int(round(max_ratio * float(non_other_counts.max())))
    if n_other <= cap:
        return X, y, subj

    other_idxs = np.where(other_mask)[0]
    rng.shuffle(other_idxs)
    keep_other = other_idxs[:cap]
    non_other_idxs = np.where(non_other_mask)[0]
    idx = np.concatenate([non_other_idxs, keep_other])
    idx.sort()

    print(f"  [cap_other] reduced 'other' from {n_other} to {cap} windows "
          f"(max_ratio={max_ratio:.2f}, largest non-other class={int(non_other_counts.max())})")
    return X[idx], y[idx], subj[idx]


def _class_counts_for_split(y: np.ndarray, mask: np.ndarray, n_classes: int) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros((n_classes,), dtype=int)
    return np.bincount(y[mask], minlength=n_classes).astype(int)


def split_by_subject_min_class(subj: np.ndarray,
                               y: np.ndarray,
                               train_frac: float,
                               val_frac: float,
                               seed: int,
                               min_class_rel: float,
                               min_class_abs: int,
                               n_tries: int = 500):
    rng = np.random.default_rng(seed)
    uniq = np.unique(subj)
    n_sub = len(uniq)

    n_tr = int(round(train_frac * n_sub))
    n_va = int(round(val_frac * n_sub))
    n_te = n_sub - n_tr - n_va
    if n_tr <= 0 or n_va < 0 or n_te <= 0:
        raise ValueError("Invalid split fractions lead to empty split(s).")

    n_classes = int(np.max(y)) + 1
    total_per_class = np.bincount(y, minlength=n_classes).astype(int)

    exp_tr = total_per_class * train_frac
    exp_va = total_per_class * val_frac
    exp_te = total_per_class * (1.0 - train_frac - val_frac)

    def req(exp):
        r = np.ceil(min_class_rel * exp).astype(int)
        if min_class_abs > 0:
            r = np.maximum(r, min_class_abs)
        return r

    req_tr = req(exp_tr)
    req_va = req(exp_va)
    req_te = req(exp_te)

    best = None  # (violation_score, tr_sub, va_sub, te_sub)

    for _ in range(max(1, n_tries)):
        perm = uniq.copy()
        rng.shuffle(perm)

        tr_sub = set(perm[:n_tr])
        va_sub = set(perm[n_tr:n_tr + n_va])
        te_sub = set(perm[n_tr + n_va:])

        tr_mask = np.isin(subj, list(tr_sub))
        va_mask = np.isin(subj, list(va_sub))
        te_mask = np.isin(subj, list(te_sub))

        c_tr = _class_counts_for_split(y, tr_mask, n_classes)
        c_va = _class_counts_for_split(y, va_mask, n_classes)
        c_te = _class_counts_for_split(y, te_mask, n_classes)

        v_tr = np.maximum(0, req_tr - c_tr).sum()
        v_va = np.maximum(0, req_va - c_va).sum()
        v_te = np.maximum(0, req_te - c_te).sum()
        violation = int(v_tr + v_va + v_te)

        if violation == 0:
            meta = {
                "train_subjects": sorted(int(s) for s in tr_sub),
                "val_subjects": sorted(int(s) for s in va_sub),
                "test_subjects": sorted(int(s) for s in te_sub),
            }
            return tr_mask, va_mask, te_mask, meta

        if best is None or violation < best[0]:
            best = (violation, tr_sub, va_sub, te_sub)

    violation, tr_sub, va_sub, te_sub = best
    tr_mask = np.isin(subj, list(tr_sub))
    va_mask = np.isin(subj, list(va_sub))
    te_mask = np.isin(subj, list(te_sub))
    meta = {
        "train_subjects": sorted(int(s) for s in tr_sub),
        "val_subjects": sorted(int(s) for s in va_sub),
        "test_subjects": sorted(int(s) for s in te_sub),
        "split_warning": {
            "message": "Could not satisfy min per-class constraints with subject-disjoint split. "
                       "Returned best-effort split.",
            "violation_score": int(violation),
            "min_class_rel": float(min_class_rel),
            "min_class_abs": int(min_class_abs),
            "tries": int(n_tries),
        }
    }
    return tr_mask, va_mask, te_mask, meta


def split_by_subject_stratified_source(
        subj: np.ndarray,
        y: np.ndarray,
        train_frac: float,
        val_frac: float,
        seed: int,
        source_divisor: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    test_frac = max(0.0, 1.0 - train_frac - val_frac)

    uniq = np.unique(subj)
    # group by source
    groups: Dict[int, List[int]] = {}
    for s in uniq:
        src = int(s) // source_divisor
        groups.setdefault(src, []).append(int(s))

    tr_subs, va_subs, te_subs = [], [], []
    for src in sorted(groups):
        subs = np.array(groups[src])
        rng.shuffle(subs)
        n = len(subs)
        n_tr = max(1, int(round(train_frac * n)))
        n_va = max(0, int(round(val_frac * n)))
        # Ensure at least 1 test subject if there are ≥3 subjects
        n_te = n - n_tr - n_va
        if n_te <= 0 and n >= 3:
            n_va = max(0, n_va - 1)
            n_te = n - n_tr - n_va
        if n_te < 0:
            n_te = 0
            n_va = n - n_tr
        tr_subs.extend(subs[:n_tr].tolist())
        va_subs.extend(subs[n_tr:n_tr + n_va].tolist())
        te_subs.extend(subs[n_tr + n_va:].tolist())

    tr_mask = np.isin(subj, tr_subs)
    va_mask = np.isin(subj, va_subs)
    te_mask = np.isin(subj, te_subs)

    meta = {
        "train_subjects": sorted(tr_subs),
        "val_subjects": sorted(va_subs),
        "test_subjects": sorted(te_subs),
        "split_strategy": "stratified_by_source",
        "source_divisor": source_divisor,
    }
    return tr_mask, va_mask, te_mask, meta


def remap_labels_contiguous(y: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str], Dict[int, int]]:
    present = sorted(np.unique(y).tolist())
    mapping = {old: i for i, old in enumerate(present)}
    y_new = np.vectorize(lambda t: mapping[int(t)])(y)
    labels_new = [labels[i] for i in present]
    return y_new, labels_new, mapping


def compute_class_dist(y: np.ndarray, label_names: List[str]) -> Dict[str, int]:
    uniq, cnt = np.unique(y, return_counts=True)
    return {label_names[int(k)]: int(v) for k, v in zip(uniq, cnt)}


def compute_basic_stats(X: np.ndarray,
                        y: np.ndarray,
                        subj: np.ndarray,
                        label_names: List[str],
                        win_len_s: float,
                        name: str) -> Dict[str, Any]:
    if X.size == 0:
        return {
            "name": name,
            "N_windows": 0,
            "T": 0,
            "C": 0,
            "num_subjects": 0,
            "class_distribution": {},
            "approx_total_duration_s": 0.0,
        }
    N, T, C = X.shape
    duration_s = float(N * win_len_s)
    return {
        "name": name,
        "N_windows": int(N),
        "T": int(T),
        "C": int(C),
        "num_subjects": int(np.unique(subj).size),
        "class_distribution": compute_class_dist(y, label_names),
        "approx_total_duration_s": duration_s,
    }


# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------

def load_uci_har(root: Path,
                 fs_target: float,
                 win_len_s: float,
                 *,
                 all_classes: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    ensure_dir(root)
    zip_path = root / "UCI_HAR_Dataset.zip"
    data_dir = root / "UCI HAR Dataset"

    if not data_dir.exists():
        if not zip_path.exists():
            print("[UCI] downloading archive...")
            r = http_get(UCI_URL, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for c in r.iter_content(1024 * 256):
                    if c:
                        f.write(c)
        print("[UCI] extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

    def read_split(split: str):
        base = data_dir / split / "Inertial Signals"

        def r(name): return np.loadtxt(base / name)

        ax = r(f"total_acc_x_{split}.txt")
        ay = r(f"total_acc_y_{split}.txt")
        az = r(f"total_acc_z_{split}.txt")
        gx = r(f"body_gyro_x_{split}.txt")
        gy = r(f"body_gyro_y_{split}.txt")
        gz = r(f"body_gyro_z_{split}.txt")
        X = np.stack([ax, ay, az, gx, gy, gz], axis=-1)  # [N, 128, 6]
        y = np.loadtxt(data_dir / split / f"y_{split}.txt").astype(int)
        subj = np.loadtxt(data_dir / split / f"subject_{split}.txt").astype(int)
        return X, y, subj

    Xt, yt, st = read_split("train")
    Xv, yv, sv = read_split("test")
    X = np.concatenate([Xt, Xv], axis=0)
    y_raw = np.concatenate([yt, yv], axis=0)
    subj = np.concatenate([st, sv], axis=0)

    y = np.vectorize(lambda lid: uci_label_to_unified(lid, all_classes=all_classes))(y_raw)

    fs_in = 50.0
    T_in = X.shape[1]
    _ = T_in / fs_in  # duration (not used explicitly)

    T_target = int(round(win_len_s * fs_target))
    if T_target <= 0:
        raise ValueError("Target window length * fs must be > 0")

    X_res = []
    for i in range(X.shape[0]):
        x_win = X[i]  # [T_in, 6]
        x_res = resample_to(fs_in, fs_target, x_win)
        if x_res.shape[0] >= T_target:
            x_res = x_res[:T_target]
        else:
            pad = np.zeros((T_target - x_res.shape[0], x_res.shape[1]), dtype=x_res.dtype)
            x_res = np.vstack([x_res, pad])
        X_res.append(x_res)

    X = np.stack(X_res, axis=0)  # [N, T_target, 6]

    # Drop samples marked _DROP (only possible when all_classes=True)
    keep = y != _DROP
    if not keep.all():
        X, y, subj = X[keep], y[keep], subj[keep]

    label_names = ALL_LABELS if all_classes else UNIFIED_LABELS
    stats = compute_basic_stats(X, y, subj, label_names, win_len_s, name="uci")
    print(f"[UCI] windows: {stats['N_windows']}, T={stats['T']}, C={stats['C']}")
    print(f"[UCI] subjects: {stats['num_subjects']}")
    print(f"[UCI] class_dist: {stats['class_distribution']}")
    return X, y, subj, stats


def load_motionsense(root: Path,
                     fs_target: float,
                     win_len_s: float,
                     stride_s: float,
                     min_purity: float = 0.0,
                     *,
                     all_classes: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    ensure_dir(root)
    cache_dir = root / "_cache"
    ensure_dir(cache_dir)

    repo_zip = cache_dir / "motion-sense-master.zip"
    repo_dir = cache_dir / "motion-sense-master"

    # If repo_dir doesn't exist, download+extract repo zip
    if not repo_dir.exists():
        if not repo_zip.exists():
            print("[MotionSense] downloading GitHub repo zip...")
            r = http_get(MOTIONSENSE_ZIP_URL, stream=True)
            r.raise_for_status()
            with open(repo_zip, "wb") as f:
                for c in r.iter_content(1024 * 256):
                    if c:
                        f.write(c)
        print("[MotionSense] extracting repo zip...")
        with zipfile.ZipFile(repo_zip, "r") as zf:
            zf.extractall(cache_dir)

    data_dir = repo_dir / "data"
    if not data_dir.exists():
        # fallback: locate extracted folder
        cands = list(cache_dir.glob("motion-sense-*"))
        for c in cands:
            if (c / "data").exists():
                repo_dir = c
                data_dir = c / "data"
                break

    a_zip_path = data_dir / "A_DeviceMotion_data.zip"
    if not a_zip_path.exists():
        print(f"[MotionSense] Missing {a_zip_path}")
        win = int(round(win_len_s * fs_target))
        return (np.zeros((0, win, 6)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int),
                compute_basic_stats(np.zeros((0, win, 6)), np.zeros((0,), dtype=int),
                                    np.zeros((0,), dtype=int), UNIFIED_LABELS, win_len_s, "motionsense"))

    # Activities and trial codes (repo convention)
    ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
    TRIAL_CODES = {
        "dws": [1, 2, 11],
        "ups": [3, 4, 12],
        "wlk": [7, 8, 15],
        "jog": [9, 16],
        "std": [6, 14],
        "sit": [5, 13],
    }

    # Subjects are 1..24
    subjects = list(range(1, 25))

    # MotionSense nominal sampling rate
    fs_in = 50.0

    win = int(round(win_len_s * fs_target))
    stride = int(round(stride_s * fs_target))

    Xs, Ys, Ss = [], [], []

    # Column-name candidates (some dumps differ slightly)
    acc_candidates = [
        ("userAcceleration.x", "userAcceleration.y", "userAcceleration.z"),
        ("userAccelerationX", "userAccelerationY", "userAccelerationZ"),
        ("user_acceleration.x", "user_acceleration.y", "user_acceleration.z"),
    ]
    gyr_candidates = [
        ("rotationRate.x", "rotationRate.y", "rotationRate.z"),
        ("rotationRateX", "rotationRateY", "rotationRateZ"),
        ("rotation_rate.x", "rotation_rate.y", "rotation_rate.z"),
        ("gyro.x", "gyro.y", "gyro.z"),
    ]

    def pick_cols(cols, dfcols):
        for triplet in cols:
            if all(c in dfcols for c in triplet):
                return list(triplet)
        return None

    # Read from the zip directly
    with zipfile.ZipFile(a_zip_path, "r") as zf:
        # Build a quick set for membership checks
        names = set(zf.namelist())

        # We expect paths like: A_DeviceMotion_data/<act>_<trial>/sub_<sid>.csv
        # Sometimes zip includes a leading folder like "A_DeviceMotion_data/..."
        # We'll generate both patterns and check which exists.
        total_jobs = sum(len(TRIAL_CODES[a]) * len(subjects) for a in ACT_LABELS)
        pbar = tqdm(total=total_jobs, desc="[MotionSense] reading A_DeviceMotion_data.zip")

        for act in ACT_LABELS:
            y_id = motionsense_label_to_unified(act, all_classes=all_classes)
            for tr in TRIAL_CODES[act]:
                for sid in subjects:
                    pbar.update(1)

                    rel1 = f"A_DeviceMotion_data/{act}_{tr}/sub_{sid}.csv"
                    rel2 = f"data/A_DeviceMotion_data/{act}_{tr}/sub_{sid}.csv"
                    rel3 = f"motion-sense-master/data/A_DeviceMotion_data/{act}_{tr}/sub_{sid}.csv"

                    member = None
                    for rel in (rel1, rel2, rel3):
                        if rel in names:
                            member = rel
                            break
                    if member is None:
                        # try a slower fallback: search by suffix
                        suffix = f"/{act}_{tr}/sub_{sid}.csv"
                        cand = [n for n in names if n.endswith(suffix)]
                        if cand:
                            member = cand[0]
                        else:
                            continue

                    try:
                        with zf.open(member) as f:
                            df = pd.read_csv(f)
                    except Exception:
                        continue

                    # Drop unnamed index col if present
                    if "Unnamed: 0" in df.columns:
                        df = df.drop(columns=["Unnamed: 0"])

                    acc_cols = pick_cols(acc_candidates, df.columns)
                    gyr_cols = pick_cols(gyr_candidates, df.columns)
                    if acc_cols is None or gyr_cols is None:
                        continue

                    x = df[acc_cols + gyr_cols].astype(float).values  # [T,6]

                    if np.isnan(x).any():
                        x = pd.DataFrame(x).interpolate(limit_direction="both").ffill().bfill().values

                    # Resample if needed
                    x = resample_to(fs_in, fs_target, x)

                    T = x.shape[0]
                    y_stream = np.full((T,), y_id, dtype=int)
                    s_stream = np.full((T,), sid, dtype=int)

                    Xw, Yw, Sw = window_stream(x, y_stream, s_stream, win, stride, min_purity)
                    if Xw.shape[0] == 0:
                        continue

                    Xs.append(Xw)
                    Ys.append(Yw)
                    Ss.append(Sw)

        pbar.close()

    if not Xs:
        print("[MotionSense] no windows produced.")
        return (np.zeros((0, win, 6)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int),
                compute_basic_stats(np.zeros((0, win, 6)), np.zeros((0,), dtype=int),
                                    np.zeros((0,), dtype=int), UNIFIED_LABELS, win_len_s, "motionsense"))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    subj = np.concatenate(Ss, axis=0)

    # Drop samples marked _DROP
    keep = y != _DROP
    if not keep.all():
        X, y, subj = X[keep], y[keep], subj[keep]

    label_names = ALL_LABELS if all_classes else UNIFIED_LABELS
    stats = compute_basic_stats(X, y, subj, label_names, win_len_s, name="motionsense")
    print(f"[MotionSense] windows: {stats['N_windows']}, T={stats['T']}, C={stats['C']}")
    print(f"[MotionSense] subjects: {stats['num_subjects']}")
    print(f"[MotionSense] class_dist: {stats['class_distribution']}")
    return X, y, subj, stats


def load_pamap2(root: Path,
                fs_target: float,
                win_len_s: float,
                stride_s: float,
                min_purity: float = 0.0,
                *,
                all_classes: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    ensure_dir(root)
    zip_path = root / "PAMAP2_Dataset.zip"
    data_dir = root / "PAMAP2_Dataset"

    if not data_dir.exists():
        if not zip_path.exists():
            print("[PAMAP2] downloading zip...")
            r = http_get(PAMAP_ZIP_URL, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for c in r.iter_content(1024 * 256):
                    if c:
                        f.write(c)
        print("[PAMAP2] extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

    proto_dir = data_dir / "Protocol"
    dat_files = sorted(proto_dir.glob("subject*.dat"))
    if not dat_files:
        print(f"[PAMAP2] no subject*.dat files found in {proto_dir}")
        return (np.zeros((0, 1, 6)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int),
                compute_basic_stats(np.zeros((0, 1, 6)), np.zeros((0,), dtype=int),
                                    np.zeros((0,), dtype=int), UNIFIED_LABELS, win_len_s, "pamap2"))

    Xs, Ys, Ss = [], [], []
    win = int(round(win_len_s * fs_target))
    stride = int(round(stride_s * fs_target))

    for dat in dat_files:
        sid = int(dat.stem.replace("subject", ""))
        try:
            df = pd.read_csv(dat, sep=r"\s+", header=None, engine="python")
        except Exception:
            continue

        total_cols = df.shape[1]
        t = df.iloc[:, 0].astype(float).values
        act = df.iloc[:, 1].astype(float).values

        if t.size < 10:
            continue

        act_series = pd.Series(act).ffill().bfill()
        act_codes = act_series.round().astype(int).values

        # IMU hand columns in PAMAP2 protocol files
        def safe(cols, total): return [c for c in cols if 0 <= c < total]

        acc_cols = safe([4, 5, 6], total_cols)     # hand acc16g
        gyr_cols = safe([10, 11, 12], total_cols)  # hand gyro
        if len(acc_cols) < 3 or len(gyr_cols) < 3:
            continue

        acc = df.iloc[:, acc_cols].astype(float)
        gyr = df.iloc[:, gyr_cols].astype(float)

        def clean(m: pd.DataFrame) -> np.ndarray:
            m = m.copy()
            m = m.interpolate(limit_direction="both").ffill().bfill()
            return m.values

        acc = clean(acc)
        gyr = clean(gyr)

        t = t - t[0]
        if t[-1] <= 0:
            continue
        dur = float(t[-1])
        T = int(round(dur * fs_target))
        if T <= 1:
            continue
        t_uniform = np.linspace(0.0, dur, T)

        acc_interp = np.vstack([
            np.interp(t_uniform, t, acc[:, i]) for i in range(3)
        ]).T
        gyr_interp = np.vstack([
            np.interp(t_uniform, t, gyr[:, i]) for i in range(3)
        ]).T

        idx_nn = np.searchsorted(t, np.clip(t_uniform, t[0], t[-1]))
        idx_nn = np.clip(idx_nn, 0, len(act_codes) - 1)
        y_stream = np.array([pamap_label_to_unified(int(act_codes[i]), all_classes=all_classes) for i in idx_nn], dtype=int)
        s_stream = np.full((T,), sid, dtype=int)

        X_stream = np.concatenate([acc_interp, gyr_interp], axis=1)  # [T,6]
        Xw, Yw, Sw = window_stream(X_stream, y_stream, s_stream, win, stride, min_purity)
        if Xw.shape[0] == 0:
            continue

        Xs.append(Xw)
        Ys.append(Yw)
        Ss.append(Sw)

    if not Xs:
        print("[PAMAP2] no windows produced.")
        return (np.zeros((0, win, 6)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int),
                compute_basic_stats(np.zeros((0, win, 6)), np.zeros((0,), dtype=int),
                                    np.zeros((0,), dtype=int), UNIFIED_LABELS, win_len_s, "pamap2"))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    subj = np.concatenate(Ss, axis=0)

    # Drop samples marked _DROP (transient / unknown activities in all_classes mode)
    keep = y != _DROP
    if not keep.all():
        print(f"  [PAMAP2] dropping {(~keep).sum()} windows with unknown activity codes")
        X, y, subj = X[keep], y[keep], subj[keep]

    label_names = ALL_LABELS if all_classes else UNIFIED_LABELS
    stats = compute_basic_stats(X, y, subj, label_names, win_len_s, name="pamap2")
    print(f"[PAMAP2] windows: {stats['N_windows']}, T={stats['T']}, C={stats['C']}")
    print(f"[PAMAP2] subjects: {stats['num_subjects']}")
    print(f"[PAMAP2] class_dist: {stats['class_distribution']}")
    return X, y, subj, stats


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, required=True, help="Output root directory (data + unified splits).")
    p.add_argument("--datasets", type=str, default="uci,motionsense,pamap2", help="Comma-separated list of datasets: uci,motionsense,pamap2")
    p.add_argument("--fs", type=float, default=50.0, help="Target sampling rate (Hz) for the unified dataset.")
    p.add_argument("--win-len", type=float, default=2.56, help="Window length in seconds.")
    p.add_argument("--win-stride", type=float, default=1.28, help="Window stride in seconds (overlap when < win-len).")
    p.add_argument("--train-frac", type=float, default=0.80, help="Train fraction (subject-wise).")
    p.add_argument("--val-frac", type=float, default=0.10, help="Validation fraction (subject-wise).")
    p.add_argument("--max-per-class", type=int, default=0, help="Optional cap per class (0 = unlimited).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffles/splits.")
    p.add_argument("--dry-run", action="store_true", help="Run everything but skip writing files.")
    p.add_argument("--min-class-rel", type=float, default=0.5, help="Minimum per-class presence per split, relative to expected (0..1).")
    p.add_argument("--min-class-abs", type=int, default=0, help="Optional absolute minimum windows per class per split (0 disables).")
    p.add_argument("--split-tries", type=int, default=500, help="How many random subject assignments to try to satisfy class constraints.")
    p.add_argument("--min-purity", type=float, default=0.0, help="Minimum purity for a window: fraction of samples with the majority label.")
    p.add_argument("--max-other-ratio", type=float, default=0.0, help="Cap 'other' class to at most this ratio × the largest non-other class.")
    p.add_argument("--drop-other", action="store_true", default=False, help="Remove all samples with the 'other' label entirely.")
    p.add_argument("--all-classes", action="store_true", default=False, help="Use all named activity classes (e.g. cycling, ironing from PAMAP2).")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = args.out_root.expanduser().resolve()
    ensure_dir(out_root)
    unified_dir = out_root / "unified"
    ensure_dir(unified_dir)

    rng = np.random.default_rng(args.seed)
    ds_list = [s.strip().lower() for s in args.datasets.split(",") if s.strip()]

    fs_target = float(args.fs)
    win_len_s = float(args.win_len)
    stride_s = float(args.win_stride)
    T_target = int(round(fs_target * win_len_s))
    if T_target <= 0:
        raise ValueError("win_len * fs must be > 0")

    use_all = bool(args.all_classes)
    drop_other = bool(args.drop_other) or use_all

    if use_all:
        print("Mode: --all-classes (expanded label set, no 'other' bucket)")
    elif drop_other:
        print("Mode: --drop-other (unified labels, 'other' class removed)")

    print(f"Target fs = {fs_target:.2f} Hz, window_len = {win_len_s:.3f} s, T = {T_target} samples")

    X_all, y_all, s_all = [], [], []
    dataset_stats: Dict[str, Any] = {}
    meta_sources: Dict[str, int] = {}

    # Offsets to keep subject IDs disjoint across datasets
    SUBJECT_OFFSETS = {"uci": 1000, "motionsense": 2000, "pamap2": 3000}

    # ---------------- UCI HAR ----------------
    if "uci" in ds_list:
        X, y, subj, stats = load_uci_har(out_root / "uci", fs_target, win_len_s,
                                          all_classes=use_all)
        X_all.append(X)
        y_all.append(y)
        s_all.append(subj + SUBJECT_OFFSETS["uci"])
        dataset_stats["uci"] = stats
        meta_sources["uci"] = stats["N_windows"]

    # ---------------- MotionSense ------------
    if "motionsense" in ds_list:
        X, y, subj, stats = load_motionsense(out_root / "motionsense", fs_target, win_len_s, stride_s,
                                              min_purity=args.min_purity,
                                              all_classes=use_all)
        X_all.append(X)
        y_all.append(y)
        s_all.append(subj + SUBJECT_OFFSETS["motionsense"])
        dataset_stats["motionsense"] = stats
        meta_sources["motionsense"] = stats["N_windows"]

    # ---------------- PAMAP2 -----------------
    if "pamap2" in ds_list:
        X, y, subj, stats = load_pamap2(out_root / "pamap2", fs_target, win_len_s, stride_s,
                                         min_purity=args.min_purity,
                                         all_classes=use_all)
        X_all.append(X)
        y_all.append(y)
        s_all.append(subj + SUBJECT_OFFSETS["pamap2"])
        dataset_stats["pamap2"] = stats
        meta_sources["pamap2"] = stats["N_windows"]

    if not X_all:
        print("No datasets loaded. Exiting.")
        return

    # Concatenate all sources
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subj = np.concatenate(s_all, axis=0)

    # Select the label space in use
    active_labels = ALL_LABELS if use_all else UNIFIED_LABELS

    # Global stats BEFORE label remapping / balancing
    global_stats_pre = {
        "N_windows": int(X.shape[0]),
        "T": int(X.shape[1]),
        "C": int(X.shape[2]),
        "total_subjects": int(np.unique(subj).size),
        "class_distribution_unified_ids": compute_class_dist(y, active_labels),
        "approx_total_duration_s": float(X.shape[0] * win_len_s),
    }

    print("\n[GLOBAL] Pre-processing statistics")
    print(f"  Total windows: {global_stats_pre['N_windows']}")
    print(f"  Total subjects: {global_stats_pre['total_subjects']}")
    print(f"  Class distribution (unified ids): {global_stats_pre['class_distribution_unified_ids']}")

    # Drop 'other' class entirely (--drop-other or implied by --all-classes with unified labels)
    if drop_other and not use_all:
        other_id = UNIFIED_L2ID["other"]
        mask = y != other_id
        n_dropped = int((~mask).sum())
        if n_dropped > 0:
            X, y, subj = X[mask], y[mask], subj[mask]
            print(f"  [drop-other] removed {n_dropped} 'other' windows")

    # Cap 'other' class before remapping (only relevant when other exists)
    if args.max_other_ratio > 0 and not drop_other:
        other_id = UNIFIED_L2ID["other"]
        X, y, subj = cap_other_class(X, y, subj, other_id, args.max_other_ratio, rng)

    # Remap labels to contiguous 0..K-1 for the final dataset
    y, labels_remapped, label_id_mapping = remap_labels_contiguous(y, active_labels)

    # Optional balancing (after remap) - if enabled, do it BEFORE splitting
    if args.max_per_class > 0:
        X, y, subj = balance_classes(X, y, subj, args.max_per_class, rng)

    # Subject-wise split, stratified by dataset source to keep class
    # distributions (especially the 'other' bucket) proportional across
    # train / val / test.
    tr_mask, va_mask, te_mask, subj_meta = split_by_subject_stratified_source(
        subj=subj, y=y,
        train_frac=args.train_frac, val_frac=args.val_frac,
        seed=args.seed,
    )

    # Standardization (fit on FINAL train split only)
    mu = X[tr_mask].reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X[tr_mask].reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X = (X - mu) / sd

    # Slice splits
    Xtr, ytr, str_ = X[tr_mask], y[tr_mask], subj[tr_mask]
    Xva, yva, sva = X[va_mask], y[va_mask], subj[va_mask]
    Xte, yte, ste = X[te_mask], y[te_mask], subj[te_mask]

    # Split-level stats (post-processing)
    split_stats = {
        "train": compute_basic_stats(Xtr, ytr, str_, labels_remapped, win_len_s, "train"),
        "val": compute_basic_stats(Xva, yva, sva, labels_remapped, win_len_s, "val"),
        "test": compute_basic_stats(Xte, yte, ste, labels_remapped, win_len_s, "test"),
    }

    # Global stats AFTER processing
    global_stats_post = {
        "N_windows": int(X.shape[0]),
        "T": int(X.shape[1]),
        "C": int(X.shape[2]),
        "total_subjects": int(np.unique(subj).size),
        "class_distribution": compute_class_dist(y, labels_remapped),
        "approx_total_duration_s": float(X.shape[0] * win_len_s),
    }

    print("\n[GLOBAL] Post-processing statistics")
    print(f"  Total windows: {global_stats_post['N_windows']}")
    print(f"  Total subjects: {global_stats_post['total_subjects']}")
    print(f"  T={global_stats_post['T']}, C={global_stats_post['C']}")
    print(f"  Class distribution (all splits): {global_stats_post['class_distribution']}")

    print("\n[SPLITS] Statistics")
    for split_name, st in split_stats.items():
        print(f"  [{split_name}] N={st['N_windows']}, subjects={st['num_subjects']}, "
              f"T={st['T']}, C={st['C']}")
        print(f"          class_dist = {st['class_distribution']}")

    # Build meta dictionary
    meta = {
        "labels": labels_remapped,                    # active labels (after dropping unused)
        "unified_label_space": UNIFIED_LABELS,        # full label set
        "fs_target_Hz": fs_target,
        "win_len_s": win_len_s,
        "win_stride_s": stride_s,
        "T": int(X.shape[1]),
        "C": int(X.shape[2]),
        "standardization": {
            "mean": mu.tolist(),
            "std": sd.tolist(),
            "fit_on": "train_split"
        },
        "label_id_mapping_from_unified_space": label_id_mapping,  # old_id -> new_id
        "min_purity": float(args.min_purity),
        "max_other_ratio": float(args.max_other_ratio),
        "drop_other": drop_other,
        "all_classes": use_all,
        "sources": meta_sources,                      # per dataset: number of windows
        "dataset_stats": dataset_stats,               # per dataset stats (pre-concat)
        "global_stats_pre": global_stats_pre,         # before remap/balance
        "global_stats_post": global_stats_post,       # after all processing
        "split_stats": split_stats,                   # train/val/test stats
    }
    meta.update(subj_meta)

    if args.dry_run:
        print("\n[dry-run] Skipping file writes.")
        print(f"[dry-run] train_N={len(Xtr)}, val_N={len(Xva)}, test_N={len(Xte)}")
        return

    # Write NPZ files + meta.json
    save_npz(unified_dir / "train.npz", Xtr, ytr, str_, meta)
    save_npz(unified_dir / "val.npz", Xva, yva, sva, meta)
    save_npz(unified_dir / "test.npz", Xte, yte, ste, meta)
    with open(unified_dir / "meta.json", "w") as f:
        json.dump(to_py(meta), f, indent=2)

    print(f"\n[OK] Saved unified dataset under: {unified_dir}")


if __name__ == "__main__":
    main()