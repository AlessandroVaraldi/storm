from __future__ import annotations

import argparse, json, math, os, random, time, logging, warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from storm import STORM, STORMConfig
from torch_ema import ExponentialMovingAverage

from utils.deploy_sim import DeploySimOps, replace_linear_conv_with_quant, set_weight_quant_enabled

# ---- quiet logs and stable threading ----
for name in ("torch._inductor", "torch._dynamo", "torchinductor", "torch.compiler"):
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"TensorFloat32 tensor cores.*", category=UserWarning)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

def _fake_quant_dequant_symm_int8(x: torch.Tensor, s: float) -> torch.Tensor:
    if not math.isfinite(float(s)) or float(s) <= 0.0:
        return x
    s_t = torch.tensor(float(s), device=x.device, dtype=torch.float32)
    x_fp32 = x.to(torch.float32)
    q = torch.round(x_fp32 / s_t).clamp(-127.0, 127.0)
    x_hat = q * s_t
    return x_hat.to(dtype=x.dtype)


def _fake_quant_dequant_ste(x: torch.Tensor, s: float) -> torch.Tensor:
    x_hat = _fake_quant_dequant_symm_int8(x, s)
    return x + (x_hat - x).detach()


# ----------------------------------------------------------------------
# SAM (Sharpness-Aware Minimization)
# ----------------------------------------------------------------------

class SAM:
    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.optimizer = optimizer
        self.rho = rho
        self.state: Dict[torch.Tensor, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        self.state.clear()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p] = p.data.clone()
                e_w = p.grad * scale
                p.data.add_(e_w)
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in self.state:
                    p.data.copy_(self.state[p])
        self.state.clear()
        self.optimizer.step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = None
        norms = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if shared_device is None:
                        shared_device = p.grad.device
                    norms.append(p.grad.detach().norm(2.0).to(shared_device))
        if not norms:
            return torch.tensor(0.0)
        return torch.stack(norms).norm(2.0)


# ----------------------------------------------------------------------
# R-Drop (Regularized Dropout)
# ----------------------------------------------------------------------

def _rdrop_kl_loss(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    kl_pq = F.kl_div(q, p_prob, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(p, q_prob, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pq + kl_qp)


# ----------------------------------------------------------------------
# Self-Distillation helpers
# ----------------------------------------------------------------------

@torch.no_grad()
def _generate_soft_labels(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    temperature: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = F.softmax(logits / temperature, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(yb.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


class DistillationLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.base_criterion = base_criterion
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        hard_target: torch.Tensor,
        soft_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce_loss = self.base_criterion(logits, hard_target)
        if soft_target is None or self.alpha <= 0.0:
            return ce_loss
        T = self.temperature
        student_log_soft = F.log_softmax(logits / T, dim=-1)
        # soft_target is already a probability distribution
        kd_loss = F.kl_div(student_log_soft, soft_target, reduction="batchmean") * (T * T)
        return (1.0 - self.alpha) * ce_loss + self.alpha * kd_loss


class NpzSequenceDatasetWithSoftLabels(Dataset):
    def __init__(
        self,
        npz_path: Path,
        *,
        soft_probs: Optional[np.ndarray] = None,
        max_samples: int = 0,
        train: bool = False,
        jitter: float = 0.01,
        scale: float = 0.05,
        time_mask: float = 0.10,
        time_warp: float = 0.05,
        p_drop_gyro: float = 0.30,
        p_drop_acc: float = 0.05,
        p_drop_axis: float = 0.10,
        seed: int = 0,
    ):
        data = np.load(npz_path, allow_pickle=False)
        x = _npz_get_first(data, ["X", "x"]).astype(np.float32)
        y = _npz_get_first(data, ["y", "Y"]).astype(np.int64)
        subj = data["subj"].astype(np.int64) if ("subj" in data.files) else None

        if x.ndim != 3:
            raise ValueError("X/x must have shape [N,T,Cin]")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must have shape [N] and match X/x")

        if max_samples and x.shape[0] > max_samples:
            x = x[:max_samples]
            y = y[:max_samples]
            if subj is not None:
                subj = subj[:max_samples]
            if soft_probs is not None:
                soft_probs = soft_probs[:max_samples]

        self.x = x
        self.y = y
        self.subj = subj
        self.soft_probs = soft_probs  # [N, C] or None
        self.train = bool(train)
        self.jitter = float(jitter)
        self.scale = float(scale)
        self.time_mask = float(time_mask)
        self.time_warp = float(time_warp)
        self.p_drop_gyro = float(p_drop_gyro)
        self.p_drop_acc = float(p_drop_acc)
        self.p_drop_axis = float(p_drop_axis)
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx])
        y = torch.tensor(self.y[idx])
        if self.train:
            x = self._augment(x)
        if self.soft_probs is not None:
            soft = torch.from_numpy(self.soft_probs[idx])
            return x, y, soft
        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        T, C = x.shape
        out = x.clone()
        if C >= 6 and (self.p_drop_gyro > 0) and (self.rng.random() < self.p_drop_gyro):
            out[:, 3:6] = 0.0
        if (self.p_drop_acc > 0) and (self.rng.random() < self.p_drop_acc) and C >= 3:
            out[:, 0:3] = 0.0
        if (self.p_drop_axis > 0) and (self.rng.random() < self.p_drop_axis):
            k = self.rng.randrange(0, C)
            out[:, k] = 0.0
        if self.scale > 0:
            s = float((1.0 + torch.randn((), dtype=out.dtype) * self.scale).clamp(0.7, 1.3).item())
            out = out * s
        if self.jitter > 0:
            out = out + torch.randn_like(out) * self.jitter
        if self.time_warp > 0 and T > 8:
            alpha = float(torch.empty(()).uniform_(1.0 - self.time_warp, 1.0 + self.time_warp).item())
            newT = int(max(4, round(T * alpha)))
            t_src = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
            t_dst = np.linspace(0.0, 1.0, num=newT, dtype=np.float32)
            out_np = out.detach().cpu().numpy()
            warped = np.vstack([np.interp(t_dst, t_src, out_np[:, c]).astype(np.float32) for c in range(C)]).T
            out = torch.from_numpy(warped).to(dtype=out.dtype)
            if out.shape[0] >= T:
                st = (out.shape[0] - T) // 2
                out = out[st:st + T]
            else:
                pad = torch.zeros((T - out.shape[0], C), dtype=out.dtype)
                out = torch.cat([out, pad], dim=0)
        if self.time_mask > 0 and T > 12:
            mT = int(T * self.time_mask)
            st = self.rng.randint(0, max(0, T - mT))
            out[st:st + mT] = 0.0
        return out


# ----------------------------------------------------------------------
# Mixup / CutMix helpers
# ----------------------------------------------------------------------

def _mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup: linearly blend pairs of examples and their labels."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5 for stability
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[idx]
    return mixed, y, y[idx], lam


def _cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix for time series: replace a temporal segment with another sample's."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    B, T, C = x.shape
    cut_len = int(T * (1.0 - lam))
    if cut_len <= 0:
        return x, y, y, 1.0
    cut_start = np.random.randint(0, max(1, T - cut_len))
    idx = torch.randperm(B, device=x.device)
    mixed = x.clone()
    mixed[:, cut_start:cut_start + cut_len, :] = x[idx, cut_start:cut_start + cut_len, :]
    lam_actual = 1.0 - cut_len / T
    return mixed, y, y[idx], lam_actual


def _mixed_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for mixup/cutmix: weighted sum of two CE/focal losses."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def _symmetric_scale_from_npz(
    npz_path: Path,
    *,
    percentile: float,
    max_abs_elems: int,
    seed: int,
) -> float:
    data = np.load(npz_path, allow_pickle=False)
    x = _npz_get_first(data, ["X", "x"]).astype(np.float32)
    x_abs = np.abs(x.reshape(-1))
    if x_abs.size == 0:
        return 1.0 / 127.0

    if max_abs_elems and x_abs.size > max_abs_elems:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(x_abs.size, size=int(max_abs_elems), replace=False)
        x_abs = x_abs[idx]

    p = float(np.percentile(x_abs, float(percentile)))
    if not math.isfinite(p) or p <= 0.0:
        return 1.0 / 127.0
    return p / 127.0


class _EMAAbsMaxObserver:
    def __init__(self, momentum: float, eps: float = 1e-12):
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.ema_maxabs: Optional[float] = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> float:
        cur = float(x.detach().to(torch.float32).abs().amax().cpu().item())
        if not math.isfinite(cur) or cur <= 0.0:
            cur = 0.0
        if self.ema_maxabs is None:
            self.ema_maxabs = max(cur, self.eps)
        else:
            self.ema_maxabs = max(self.eps, self.momentum * self.ema_maxabs + (1.0 - self.momentum) * cur)
        return float(self.ema_maxabs)


class _QATHookManager:
    def __init__(self, *, momentum: float):
        self.momentum = float(momentum)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.observers: dict[str, _EMAAbsMaxObserver] = {}

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        self.observers.clear()

    def attach_default(self, model: nn.Module) -> None:
        # Keep this list aligned with the model structure, but conservative about logits.
        module_names: list[str] = []
        for name, m in model.named_modules():
            if name == "":
                continue
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.LayerNorm)):
                # Avoid quantizing classifier output logits during training.
                if name.endswith("classifier"):
                    continue
                module_names.append(name)
            elif type(m).__name__ in ("TransformerBlock", "AttnPool"):
                # These return tuples; the hook will quantize the first element.
                module_names.append(name)

        for name in module_names:
            m = dict(model.named_modules())[name]
            obs = _EMAAbsMaxObserver(momentum=self.momentum)
            self.observers[name] = obs

            def _make_hook(obs_ref: _EMAAbsMaxObserver):
                def _hook(_module: nn.Module, _inp, out):
                    if torch.is_tensor(out):
                        ema = obs_ref.update(out)
                        s = ema / 127.0
                        return _fake_quant_dequant_ste(out, s)
                    if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                        y0 = out[0]
                        ema = obs_ref.update(y0)
                        s = ema / 127.0
                        y0q = _fake_quant_dequant_ste(y0, s)
                        return (y0q,) + tuple(out[1:])
                    return out

                return _hook

            self.handles.append(m.register_forward_hook(_make_hook(obs)))


def _npz_get_first(data: "np.lib.npyio.NpzFile", keys: list[str]) -> np.ndarray:
    for k in keys:
        if k in data:
            return data[k]
    raise KeyError(f"None of keys present: {keys}. Found: {list(data.keys())}")


class NpzSequenceDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        *,
        max_samples: int = 0,
        train: bool = False,
        # ---- augment knobs (match first script defaults) ----
        jitter: float = 0.01,
        scale: float = 0.05,
        time_mask: float = 0.10,
        time_warp: float = 0.05,
        # ---- missing-sensor robustness (no extra channels, keep compatibility) ----
        p_drop_gyro: float = 0.30,
        p_drop_acc: float = 0.05,
        p_drop_axis: float = 0.10,
        seed: int = 0,
    ):
        data = np.load(npz_path, allow_pickle=False)
        # create_dataset.py uses X/y/subj/meta; keep backward compat with x/y
        x = _npz_get_first(data, ["X", "x"]).astype(np.float32)  # [N,T,Cin]
        y = _npz_get_first(data, ["y", "Y"]).astype(np.int64)    # [N]
        subj = data["subj"].astype(np.int64) if ("subj" in data.files) else None

        if x.ndim != 3:
            raise ValueError("X/x must have shape [N,T,Cin]")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must have shape [N] and match X/x")

        if max_samples and x.shape[0] > max_samples:
            x = x[:max_samples]
            y = y[:max_samples]
            if subj is not None:
                subj = subj[:max_samples]

        self.x = x
        self.y = y
        self.subj = subj
        self.train = bool(train)
        self.jitter = float(jitter)
        self.scale = float(scale)
        self.time_mask = float(time_mask)
        self.time_warp = float(time_warp)
        self.p_drop_gyro = float(p_drop_gyro)
        self.p_drop_acc = float(p_drop_acc)
        self.p_drop_axis = float(p_drop_axis)
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx])  # [T,C]
        y = torch.tensor(self.y[idx])

        # "true missing sensors" based on dataset id encoded in subj offsets:
        # uci~1000s, motionsense~2000s, pamap2~3000s.

        if self.train:
            x = self._augment(x)

        return x, y

    def _dataset_id(self, idx: int) -> Optional[int]:
        if self.subj is None:
            return None
        sid = int(self.subj[idx])
        return sid // 1000

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Match first script logic, but keep shape [T,C] and stay deterministic per-worker.
        T, C = x.shape
        out = x.clone()

        # ---- simulate missing sensors (group + axis dropout), no extra channels ----
        if C >= 6 and (self.p_drop_gyro > 0) and (self.rng.random() < self.p_drop_gyro):
            out[:, 3:6] = 0.0
        if (self.p_drop_acc > 0) and (self.rng.random() < self.p_drop_acc) and C >= 3:
            out[:, 0:3] = 0.0
        if (self.p_drop_axis > 0) and (self.rng.random() < self.p_drop_axis):
            k = self.rng.randrange(0, C)
            out[:, k] = 0.0

        # ---- amplitude augs ----
        if self.scale > 0:
            s = float((1.0 + torch.randn((), dtype=out.dtype) * self.scale).clamp(0.7, 1.3).item())
            out = out * s
        if self.jitter > 0:
            out = out + torch.randn_like(out) * self.jitter

        # ---- time warp (linear resample) ----
        if self.time_warp > 0 and T > 8:
            alpha = float(torch.empty(()).uniform_(1.0 - self.time_warp, 1.0 + self.time_warp).item())
            newT = int(max(4, round(T * alpha)))
            t_src = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
            t_dst = np.linspace(0.0, 1.0, num=newT, dtype=np.float32)
            out_np = out.detach().cpu().numpy()
            warped = np.vstack([np.interp(t_dst, t_src, out_np[:, c]).astype(np.float32) for c in range(C)]).T
            out = torch.from_numpy(warped).to(dtype=out.dtype)
            # center crop/pad to original T
            if out.shape[0] >= T:
                st = (out.shape[0] - T) // 2
                out = out[st:st + T]
            else:
                pad = torch.zeros((T - out.shape[0], C), dtype=out.dtype)
                out = torch.cat([out, pad], dim=0)

        # ---- time mask ----
        if self.time_mask > 0 and T > 12:
            mT = int(T * self.time_mask)
            st = self.rng.randint(0, max(0, T - mT))
            out[st:st + mT] = 0.0

        return out


def _resolve_split_path(p: Optional[Path], split_name: str) -> Optional[Path]:
    if p is None:
        return None
    if p.is_dir():
        candidate = p / f"{split_name}.npz"
        return candidate
    return p


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


@torch.no_grad()
def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


@torch.no_grad()
def _macro_f1_from_cm(cm: np.ndarray) -> float:
    # cm: [C,C]
    C = cm.shape[0]
    f1s: list[float] = []
    for c in range(C):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    criterion: nn.Module,
    *,
    input_fakequant_s: Optional[float] = None,
) -> Dict[str, object]:
    model.eval()
    losses: list[float] = []
    ys: list[int] = []
    ps: list[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if input_fakequant_s is not None:
            xb = _fake_quant_dequant_ste(xb, float(input_fakequant_s))
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(float(loss.detach().cpu().item()))
        pred = torch.argmax(logits, dim=-1)
        ys.extend(yb.detach().cpu().numpy().astype(np.int64).tolist())
        ps.extend(pred.detach().cpu().numpy().astype(np.int64).tolist())

    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(ps, dtype=np.int64)
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_f1 = _macro_f1_from_cm(cm)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    per_class_acc = (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).astype(np.float64)

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
        "n": int(y_true.size),
    }


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    criterion: nn.Module,
    *,
    n_aug: int = 5,
    jitter: float = 0.005,
    scale: float = 0.03,
    input_fakequant_s: Optional[float] = None,
) -> Dict[str, object]:
    """Evaluate with Test-Time Augmentation: average logits over original + augmented inputs."""
    model.eval()
    losses: list[float] = []
    ys: list[int] = []
    ps: list[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        all_logits: list[torch.Tensor] = []

        # Original (clean) forward pass
        xb_in = xb
        if input_fakequant_s is not None:
            xb_in = _fake_quant_dequant_ste(xb_in, float(input_fakequant_s))
        all_logits.append(model(xb_in))

        # Augmented forward passes
        for _ in range(n_aug):
            xb_aug = xb.clone()
            # Additive jitter
            if jitter > 0:
                xb_aug = xb_aug + torch.randn_like(xb_aug) * jitter
            # Multiplicative scale
            if scale > 0:
                s = float(torch.empty(()).uniform_(1.0 - scale, 1.0 + scale).item())
                xb_aug = xb_aug * s
            if input_fakequant_s is not None:
                xb_aug = _fake_quant_dequant_ste(xb_aug, float(input_fakequant_s))
            all_logits.append(model(xb_aug))

        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        loss = criterion(avg_logits, yb)
        losses.append(float(loss.detach().cpu().item()))
        pred = torch.argmax(avg_logits, dim=-1)
        ys.extend(yb.detach().cpu().numpy().astype(np.int64).tolist())
        ps.extend(pred.detach().cpu().numpy().astype(np.int64).tolist())

    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(ps, dtype=np.int64)
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_f1 = _macro_f1_from_cm(cm)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    per_class_acc = (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).astype(np.float64)

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
        "n": int(y_true.size),
    }


class FocalLoss(nn.Module):
    def __init__(
        self,
        *,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"expected logits [N,C], got {tuple(logits.shape)}")
        if target.ndim != 1:
            raise ValueError(f"expected target [N], got {tuple(target.shape)}")

        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]
        pt = log_pt.exp()

        nll = -log_pt
        if self.label_smoothing > 0.0:
            smooth = -log_probs.mean(dim=-1)
            nll = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth

        focal = (1.0 - pt).clamp(min=0.0).pow(self.gamma)
        loss = focal * nll

        if self.weight is not None:
            alpha = self.weight.gather(0, target)
            loss = loss * alpha

        return loss.mean()


def _maybe_plot(history: Dict[str, list], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] plotting disabled (matplotlib missing?): {e}")
        return

    epochs = history.get("epoch", [])
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax0, ax1, ax2 = axes

    ax0.plot(epochs, history.get("train_loss", []), label="train")
    ax0.plot(epochs, history.get("val_loss", []), label="val")
    ax0.set_title("Loss")
    ax0.set_xlabel("epoch")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(epochs, history.get("train_acc", []), label="train")
    ax1.plot(epochs, history.get("val_acc", []), label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history.get("lr"):
        ax2.plot(epochs, history.get("lr", []))
    ax2.set_title("LR")
    ax2.set_xlabel("epoch")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()

    # ── data I/O ──
    ap.add_argument("--data", type=Path, default=None, help="single NPZ, legacy (X/y or x/y)")
    ap.add_argument("--train", type=Path, default=None, help="train split NPZ")
    ap.add_argument("--val", type=Path, default=None, help="val split NPZ (omit → 10%% of train)")
    ap.add_argument("--test", type=Path, default=None, help="test split NPZ")
    ap.add_argument("--out", type=Path, required=True, help="output checkpoint .pt")
    ap.add_argument("--max-train", type=int, default=0, help="cap train samples (0=all)")
    ap.add_argument("--max-val", type=int, default=0, help="cap val samples (0=all)")

    # ── model architecture ──
    ap.add_argument("--model", type=str, default="storm", choices=["storm"], help="model architecture (default: storm)")
    ap.add_argument("--in-ch", type=int, default=6, help="input channels")
    ap.add_argument("--d-model", type=int, default=32, help="embedding dim")
    ap.add_argument("--nhead", type=int, default=2, help="attention heads")
    ap.add_argument("--depth", type=int, default=2, help="transformer blocks")
    ap.add_argument("--ffn-mult", type=int, default=2, help="FFN hidden multiplier")
    ap.add_argument("--num-classes", type=int, default=8, help="output classes")
    ap.add_argument("--attn-window", type=int, default=64, help="attention window size")
    ap.add_argument("--int-ln", action="store_false", help="use IntegerLayerNorm (LUT rsqrt)")
    ap.add_argument("--int-ln-lut-size", type=int, default=256, help="rsqrt LUT entries")

    # ── training basics ──
    ap.add_argument("--epochs", type=int, default=200, help="total training epochs")
    ap.add_argument("--batch", type=int, default=256, help="batch size")
    ap.add_argument("--lr", type=float, default=6e-4, help="learning rate (max_lr for onecycle)")
    ap.add_argument("--weight-decay", type=float, default=8e-4, help="AdamW weight decay")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="gradient clipping norm")
    ap.add_argument("--scheduler", type=str, default="onecycle", choices=["none", "cosine", "onecycle"], help="LR schedule")
    ap.add_argument("--warmup-epochs", type=int, default=15, help="linear warmup epochs (cosine)")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    ap.add_argument("--num-workers", type=int, default=4, help="dataloader workers")
    ap.add_argument("--amp", action="store_false", help="mixed-precision (CUDA only)")
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (0=off)")

    # ── loss & class balancing ──
    ap.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"], help="loss function")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="focal loss gamma")
    ap.add_argument("--label-smoothing", type=float, default=0.05, help="label smoothing")
    ap.add_argument("--class-weight", type=str, default="none", choices=["none", "balanced"], help="class weights for loss")
    ap.add_argument("--sampler", type=str, default="weighted", choices=["none", "weighted"], help="WeightedRandomSampler")
    ap.add_argument("--metric", type=str, default="val_quant_acc", choices=["val_loss", "val_acc", "val_macro_f1", "val_quant_loss", "val_quant_acc", "val_quant_macro_f1"], help="best-ckpt metric")
    ap.add_argument("--early-stop", type=int, default=15, help="patience epochs (0=off)")

    # ── data augmentation ──
    ap.add_argument("--jitter", type=float, default=0.005, help="additive noise std")
    ap.add_argument("--scale", type=float, default=0.08, help="multiplicative scale std")
    ap.add_argument("--time-mask", type=float, default=0.10, help="time-mask fraction")
    ap.add_argument("--time-warp", type=float, default=0.03, help="time-warp range")
    ap.add_argument("--p-drop-gyro", type=float, default=0.30, help="prob zero-out gyro [3:6]")
    ap.add_argument("--p-drop-acc", type=float, default=0.03, help="prob zero-out acc [0:3]")
    ap.add_argument("--p-drop-axis", type=float, default=0.15, help="prob zero-out one axis")
    ap.add_argument("--mixup-alpha", type=float, default=0.0, help="mixup Beta alpha (0=off)")
    ap.add_argument("--cutmix-alpha", type=float, default=0.0, help="cutmix Beta alpha (0=off)")
    ap.add_argument("--mix-prob", type=float, default=0.5, help="per-batch mix probability")

    # ── regularization ──
    ap.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth rate (0=off)")
    ap.add_argument("--feat-dropout", type=float, default=0.0, help="dropout before classifier (0=off)")
    ap.add_argument("--sam", action="store_true", default=False, help="enable SAM (flatter minima)")
    ap.add_argument("--sam-rho", type=float, default=0.05, help="SAM perturbation radius")
    ap.add_argument("--rdrop-alpha", type=float, default=0.0, help="R-Drop KL weight (0=off)")

    # ── quantization (IQAT / QAT) ──
    ap.add_argument("--iqat", action="store_false", help="input fake-quant int8")
    ap.add_argument("--iqat-percentile", type=float, default=99.8, help="percentile for input scale")
    ap.add_argument("--iqat-max-elems", type=int, default=2_000_000, help="max elements for percentile")
    ap.add_argument("--iqat-s-input", type=float, default=0.0, help="override input scale (0=auto)")
    ap.add_argument("--iqat-scale-jitter", type=float, default=0.1, help="train-only scale jitter ±frac")
    ap.add_argument("--qat", action="store_false", help="fake-quant activations via EMA hooks")
    ap.add_argument("--qat-momentum", type=float, default=0.93, help="QAT EMA observer momentum")
    ap.add_argument("--qat-lr-mult", type=float, default=0.1, help="LR multiplier for QAT stage (0=full)")
    ap.add_argument("--eval-quant", action="store_false", help="report quantized val/test metrics")

    # ── deploy simulation ──
    ap.add_argument("--deploy-sim", type=str, default="periodic", choices=["off", "last_epochs", "periodic", "always"], help="deploy-sim schedule")
    ap.add_argument("--deploy-sim-last-epochs", type=int, default=3, help="epochs for last_epochs mode")
    ap.add_argument("--deploy-sim-every", type=int, default=12, help="batch period for periodic mode")

    # ── self-distillation ──
    ap.add_argument("--self-distill-epochs", type=int, default=0, help="extra distill epochs (0=off)")
    ap.add_argument("--self-distill-temp", type=float, default=3.0, help="soft-label temperature")
    ap.add_argument("--self-distill-alpha", type=float, default=0.5, help="soft vs hard weight")
    ap.add_argument("--self-distill-lr-mult", type=float, default=0.3, help="LR multiplier for distill")

    # ── test-time augmentation ──
    ap.add_argument("--tta", type=int, default=0, help="TTA augmentations (0=off)")
    ap.add_argument("--tta-jitter", type=float, default=0.005, help="TTA noise std")
    ap.add_argument("--tta-scale", type=float, default=0.03, help="TTA scale range")

    args = ap.parse_args()

    _set_seed(args.seed)

    if args.data is None and args.train is None:
        raise SystemExit("Provide --train (recommended) or --data (legacy)")

    if args.train is None:
        args.train = args.data

    tr_path = _resolve_split_path(args.train, "train")
    va_path = _resolve_split_path(args.val, "val") if args.val is not None else None
    te_path = _resolve_split_path(args.test, "test") if args.test is not None else None

    train_ds = NpzSequenceDataset(
        tr_path,
        max_samples=args.max_train,
        train=True,
        jitter=float(args.jitter),
        scale=float(args.scale),
        time_mask=float(args.time_mask),
        time_warp=float(args.time_warp),
        p_drop_gyro=float(args.p_drop_gyro),
        p_drop_acc=float(args.p_drop_acc),
        p_drop_axis=float(args.p_drop_axis),
        seed=int(args.seed),
    )
    if va_path is not None:
        val_ds = NpzSequenceDataset(va_path, max_samples=args.max_val, train=False, seed=int(args.seed))
    else:
        n = len(train_ds)
        n_val = max(1, int(0.1 * n))
        n_train = n - n_val
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

    pin = (args.device.startswith("cuda") and torch.cuda.is_available())

    # Print class stats (helps diagnose imbalance)
    def _y_from_ds(ds: Dataset) -> np.ndarray:
        if isinstance(ds, torch.utils.data.Subset):
            return ds.dataset.y[np.asarray(ds.indices, dtype=np.int64)]
        return ds.y

    y_train_all = _y_from_ds(train_ds)
    y_val_all = _y_from_ds(val_ds)
    tr_counts = np.bincount(y_train_all.astype(np.int64), minlength=args.num_classes)
    va_counts = np.bincount(y_val_all.astype(np.int64), minlength=args.num_classes)
    print(f"train: N={len(y_train_all)} class_counts={tr_counts.tolist()}")
    print(f"val:   N={len(y_val_all)} class_counts={va_counts.tolist()}")

    sampler = None
    shuffle = True
    if args.sampler == "weighted":
        # per-sample weight = 1 / count[label]
        denom = np.maximum(tr_counts, 1).astype(np.float64)
        w_per_class = 1.0 / denom
        sample_w = w_per_class[y_train_all.astype(np.int64)]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).double(),
            num_samples=len(sample_w),
            replacement=True,
        )
        shuffle = False
        print("train sampler: WeightedRandomSampler(enabled)")

    # Avoid zero-length epochs: if drop_last=True and N < batch_size, DataLoader yields 0 batches.
    # That can trigger LR-scheduler warnings (scheduler.step called before any optimizer.step).
    drop_last_train = True
    if len(train_ds) < int(args.batch):
        drop_last_train = False
        print(f"note: len(train)={len(train_ds)} < batch={int(args.batch)} => setting drop_last=False")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last_train,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=bool(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=bool(args.num_workers > 0),
    )

    is_baseline = args.model != "storm"

    if is_baseline:
        model = build_baseline(args.model, in_ch=args.in_ch, num_classes=args.num_classes).to(args.device)
        cfg = model.cfg
        print(f"model: {args.model} baseline ({sum(p.numel() for p in model.parameters()):,d} params)")
    else:
        cfg = STORMConfig(
            in_ch=args.in_ch,
            d_model=args.d_model,
            nhead=args.nhead,
            depth=args.depth,
            ffn_mult=args.ffn_mult,
            num_classes=args.num_classes,
            attention_window=args.attn_window,
            int_layernorm=bool(args.int_ln),
            int_ln_lut_size=int(args.int_ln_lut_size),
            drop_path_rate=float(args.drop_path),
            feat_dropout=float(args.feat_dropout),
        )
        model = STORM(cfg).to(args.device)
        if args.int_ln:
            print(f"int-ln: enabled (lut_size={args.int_ln_lut_size})")
        if cfg.drop_path_rate > 0:
            print(f"drop-path: rate={cfg.drop_path_rate:.3f} (linearly increasing per block)")
        if cfg.feat_dropout > 0:
            print(f"feat-dropout: rate={cfg.feat_dropout:.3f}")

    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        print(f"mix: mixup_alpha={args.mixup_alpha:.2f}, cutmix_alpha={args.cutmix_alpha:.2f}, prob={args.mix_prob:.2f}")
    if args.sam:
        print(f"sam: enabled (rho={args.sam_rho:.4f})")
    if args.rdrop_alpha > 0:
        print(f"r-drop: alpha={args.rdrop_alpha:.2f}")
    if args.self_distill_epochs > 0:
        print(f"self-distill: epochs={args.self_distill_epochs}, temp={args.self_distill_temp:.1f}, "
              f"alpha={args.self_distill_alpha:.2f}, lr_mult={args.self_distill_lr_mult:.2f}")

    # ---- deployment-sim (optional, STORM only) ----
    deploy_sim_enabled = (not is_baseline) and str(args.deploy_sim) != "off"
    deploy_ops: Optional[DeploySimOps] = None
    if deploy_sim_enabled:
        replace_linear_conv_with_quant(model)
        model.to(args.device)
        set_weight_quant_enabled(model, False)
        deploy_ops = DeploySimOps(device=torch.device(args.device))

    def _deploy_on_for_epoch_batch(epoch_idx0: int, batch_idx0: int, *, stage_epochs: int) -> bool:
        if not deploy_sim_enabled:
            return False
        mode = str(args.deploy_sim)
        if mode == "always":
            return True
        if mode == "last_epochs":
            last_n = max(0, int(args.deploy_sim_last_epochs))
            return epoch_idx0 >= max(0, int(stage_epochs) - last_n)
        if mode == "periodic":
            k = max(1, int(args.deploy_sim_every))
            return (batch_idx0 % k) == 0
        return False

    def _set_deploy_sim(enabled: bool) -> None:
        if not deploy_sim_enabled:
            return
        set_weight_quant_enabled(model, bool(enabled))
        if bool(enabled) and deploy_ops is not None:
            model.set_ops(deploy_ops)
        else:
            from storm import DefaultOps

            model.set_ops(DefaultOps())

    # Input scale for IQAT (also reused for quantized-input eval + export)
    q_s_input: Optional[float] = None
    if args.iqat or args.eval_quant:
        if args.iqat_s_input and float(args.iqat_s_input) > 0.0:
            q_s_input = float(args.iqat_s_input)
        else:
            q_s_input = _symmetric_scale_from_npz(
                tr_path,
                percentile=float(args.iqat_percentile),
                max_abs_elems=int(args.iqat_max_elems),
                seed=int(args.seed),
            )
        msg = f"quant: s_input={q_s_input:.8e} (percentile={args.iqat_percentile}, iqat={bool(args.iqat)}, qat={bool(args.qat)})"
        if args.iqat and float(args.iqat_scale_jitter) > 0.0:
            msg += f"; iqat_scale_jitter=±{100.0*float(args.iqat_scale_jitter):.1f}% (train-only)"
        print(msg)

    qat_mgr: Optional[_QATHookManager] = _QATHookManager(momentum=float(args.qat_momentum)) if bool(args.qat) else None

    # Optimizer LR:
    # - onecycle: start from a low base LR and let OneCycle drive to max_lr=args.lr
    # - otherwise: use args.lr directly
    # AdamW param groups: decay vs no_decay (match first script stability)
    def _build_param_groups(m: nn.Module, base_lr: float, wd: float):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            is_norm = ("norm" in n.lower()) or ("bn" in n.lower()) or ("bias" in n.lower())
            (no_decay if is_norm else decay).append(p)
        return [
            {"params": decay, "lr": base_lr, "weight_decay": wd},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]

    def _make_opt_and_scheduler(*, stage_epochs: int, lr_override: float = 0.0,
                                  enable_sam: bool = True,
                                  loader_for_steps: Optional[DataLoader] = None,
                                  ) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler], Optional[SAM]]:
        if int(stage_epochs) <= 0:
            raise ValueError("stage_epochs must be > 0")

        effective_lr = float(lr_override) if lr_override > 0 else float(args.lr)

        if args.scheduler == "onecycle":
            base_lr = effective_lr / 25.0
            opt = torch.optim.AdamW(_build_param_groups(model, base_lr, float(args.weight_decay)))
        else:
            opt = torch.optim.AdamW(_build_param_groups(model, effective_lr, float(args.weight_decay)))

        _loader = loader_for_steps if loader_for_steps is not None else train_loader
        steps_per_epoch = max(1, len(_loader))
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max(1, int(stage_epochs) - max(int(args.warmup_epochs), 0)),
            )
        elif args.scheduler == "onecycle":
            pct_start = min(0.3, max(0.0, float(args.warmup_epochs) / max(int(stage_epochs), 1)))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=effective_lr,
                epochs=int(stage_epochs),
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start,
                anneal_strategy="cos",
            )
        sam_wrapper: Optional[SAM] = None
        if args.sam and enable_sam:
            sam_wrapper = SAM(opt, rho=float(args.sam_rho))
        return opt, scheduler, sam_wrapper

    # Loss weights
    weight_t: Optional[torch.Tensor] = None
    if args.class_weight == "balanced":
        counts = tr_counts.astype(np.float64)
        inv = 1.0 / np.maximum(counts, 1.0)
        # Normalize to mean=1 so the loss scale stays comparable and stable.
        # Common formula: w_c ∝ 1 / count_c with average weight ~= 1.
        w = inv * (float(inv.size) / np.maximum(inv.sum(), 1e-12))
        weight_t = torch.from_numpy(w.astype(np.float32))

    weight_t_dev = weight_t.to(args.device) if weight_t is not None else None
    if str(args.loss) == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight_t_dev, label_smoothing=float(args.label_smoothing))
    elif str(args.loss) == "focal":
        criterion = FocalLoss(weight=weight_t_dev, gamma=float(args.focal_gamma), label_smoothing=float(args.label_smoothing))
    else:
        raise SystemExit(f"unknown --loss {args.loss}")

    history: Dict[str, list] = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out
    last_path = args.out.with_name(args.out.stem + ".last.pt")
    preqat_best_path = args.out.with_name(args.out.stem + ".preqat.pt")
    preqat_last_path = args.out.with_name(args.out.stem + ".preqat.last.pt")
    metrics_path = args.out.with_suffix(".metrics.json")
    plot_path = args.out.with_suffix(".curves.png")

    def _metric_is_higher_better(metric: str) -> bool:
        return metric != "val_loss" and metric != "val_quant_loss"

    def _metric_value(
        *,
        metric: str,
        val_loss: float,
        val_acc: float,
        val_macro_f1: float,
        val_q_loss: Optional[float],
        val_q_acc: Optional[float],
        val_q_macro_f1: Optional[float],
    ) -> float:
        if metric == "val_loss":
            return float(val_loss)
        if metric == "val_acc":
            return float(val_acc)
        if metric == "val_macro_f1":
            return float(val_macro_f1)
        if metric.startswith("val_quant_"):
            if (val_q_loss is None) or (val_q_acc is None) or (val_q_macro_f1 is None):
                raise SystemExit("--metric val_quant_* requires --eval-quant and a valid s_input")
            if metric == "val_quant_loss":
                return float(val_q_loss)
            if metric == "val_quant_acc":
                return float(val_q_acc)
            if metric == "val_quant_macro_f1":
                return float(val_q_macro_f1)
        raise SystemExit(f"unknown --metric {metric}")

    def _maybe_load_checkpoint(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict) and ("state_dict" in ckpt):
                return ckpt
        except Exception as e:
            print(f"[warn] could not load checkpoint {path}: {e}")
        return None

    def _load_state_for_stage(path: Path) -> None:
        ckpt = _maybe_load_checkpoint(path)
        if ckpt is None:
            raise SystemExit(f"could not load checkpoint {path}")
        # For *continuing training*, prefer raw (non-EMA) weights if present.
        sd = ckpt.get("state_dict_raw", None)
        if sd is None:
            sd = ckpt["state_dict"]
        model.load_state_dict(sd, strict=True)
        model.to(args.device)

    def _run_stage(
        *,
        stage_name: str,
        stage_epochs: int,
        global_epoch_offset: int,
        enable_qat: bool,
        stage_best_path: Path,
        stage_last_path: Path,
        lr_override: float = 0.0,
        custom_criterion: Optional[nn.Module] = None,
        custom_train_loader: Optional[DataLoader] = None,
        enable_sam: bool = True,
        enable_rdrop: bool = True,
    ) -> dict:
        if int(stage_epochs) <= 0:
            return {
                "stage": stage_name,
                "epochs_ran": 0,
                "best_path": str(stage_best_path),
                "last_path": str(stage_last_path),
                "best_saved": False,
                "best_metric": None,
            }

        stage_train_loader = custom_train_loader if custom_train_loader is not None else train_loader
        opt, scheduler, sam_wrapper = _make_opt_and_scheduler(
            stage_epochs=int(stage_epochs), lr_override=lr_override,
            enable_sam=enable_sam, loader_for_steps=stage_train_loader,
        )
        stage_criterion = custom_criterion if custom_criterion is not None else criterion
        stage_rdrop_alpha = float(args.rdrop_alpha) if enable_rdrop else 0.0

        # SAM is incompatible with AMP (double forward/backward with mixed precision is tricky)
        stage_use_amp = bool(
            args.amp and args.device.startswith("cuda") and torch.cuda.is_available()
            and (not enable_qat) and (sam_wrapper is None)
        )
        scaler = torch.amp.GradScaler(enabled=stage_use_amp)

        ema: Optional[ExponentialMovingAverage] = None
        if float(args.ema_decay) > 0.0:
            ema = ExponentialMovingAverage(model.parameters(), decay=float(args.ema_decay))

        best_saved = False
        best_metric: Optional[float] = None
        bad_epochs = 0
        improved_dir_higher = _metric_is_higher_better(str(args.metric))

        stage_history: Dict[str, list] = {
            "epoch": [],
            "stage": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_macro_f1": [],
            "lr": [],
        }

        for epoch in range(int(stage_epochs)):
            model.train()
            train_losses: list[float] = []
            tr_correct = 0
            tr_seen = 0
            amp_skipped_steps = 0
            nonfinite_loss_batches = 0

            # Warmup: linearly ramp LR for first warmup_epochs (cosine scheduler starts after)
            if args.scheduler == "cosine" and int(args.warmup_epochs) > 0 and epoch < int(args.warmup_epochs):
                warm_frac = float(epoch + 1) / float(max(1, int(args.warmup_epochs)))
                for pg in opt.param_groups:
                    pg["lr"] = float(args.lr) * warm_frac

            for batch_idx, batch_data in enumerate(stage_train_loader):
                # Support both (x, y) and (x, y, soft_target) tuples
                if len(batch_data) == 3:
                    xb, yb, soft_tb = batch_data
                    soft_tb = soft_tb.to(args.device, non_blocking=True)
                else:
                    xb, yb = batch_data
                    soft_tb = None
                xb = xb.to(args.device, non_blocking=True)
                yb = yb.to(args.device, non_blocking=True)
                if args.iqat and q_s_input is not None:
                    s_eff = float(q_s_input)
                    jitter = float(args.iqat_scale_jitter)
                    if jitter > 0.0:
                        lo = max(1e-12, 1.0 - jitter)
                        hi = 1.0 + jitter
                        r = float(torch.empty((), device=xb.device).uniform_(lo, hi).item())
                        s_eff *= r
                    xb = _fake_quant_dequant_ste(xb, s_eff)

                # ---- Mixup / CutMix ----
                yb_orig = yb  # keep original targets for accuracy tracking
                mix_active = False
                lam_mix = 1.0
                ya_mix = yb
                yb_mix = yb
                if (args.mixup_alpha > 0 or args.cutmix_alpha > 0) and random.random() < args.mix_prob:
                    mix_active = True
                    if args.cutmix_alpha > 0 and (args.mixup_alpha <= 0 or random.random() > 0.5):
                        xb, ya_mix, yb_mix, lam_mix = _cutmix_data(xb, yb, args.cutmix_alpha)
                    else:
                        xb, ya_mix, yb_mix, lam_mix = _mixup_data(xb, yb, args.mixup_alpha)

                _set_deploy_sim(_deploy_on_for_epoch_batch(epoch, int(batch_idx), stage_epochs=int(stage_epochs)))
                opt.zero_grad(set_to_none=True)

                # ---- Forward + loss (potentially with R-Drop and distillation) ----
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=stage_use_amp):
                    logits = model(xb)

                    # Compute base task loss
                    if isinstance(stage_criterion, DistillationLoss) and soft_tb is not None:
                        task_loss = stage_criterion(logits, yb, soft_tb)
                    elif mix_active:
                        task_loss = _mixed_criterion(stage_criterion, logits, ya_mix, yb_mix, lam_mix)
                    else:
                        task_loss = stage_criterion(logits, yb)

                    loss = task_loss

                    # ---- R-Drop: second forward with different dropout, KL consistency ----
                    if stage_rdrop_alpha > 0 and model.training:
                        logits2 = model(xb)
                        if isinstance(stage_criterion, DistillationLoss) and soft_tb is not None:
                            task_loss2 = stage_criterion(logits2, yb, soft_tb)
                        elif mix_active:
                            task_loss2 = _mixed_criterion(stage_criterion, logits2, ya_mix, yb_mix, lam_mix)
                        else:
                            task_loss2 = stage_criterion(logits2, yb)
                        kl_loss = _rdrop_kl_loss(logits, logits2)
                        loss = 0.5 * (task_loss + task_loss2) + stage_rdrop_alpha * kl_loss

                if not torch.isfinite(loss.detach()):
                    nonfinite_loss_batches += 1
                    if nonfinite_loss_batches <= 3:
                        print(
                            f"[warn] non-finite loss at stage={stage_name} epoch={global_epoch_offset + epoch + 1} "
                            f"batch={batch_idx} => skipping optimizer/scheduler step"
                        )
                    opt.zero_grad(set_to_none=True)
                    continue

                # ---- SAM or standard optimizer step ----
                if sam_wrapper is not None:
                    # SAM: first backward + ascent step
                    loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                    sam_wrapper.first_step()
                    # SAM: second forward at perturbed weights
                    logits_sam = model(xb)
                    if isinstance(stage_criterion, DistillationLoss) and soft_tb is not None:
                        loss2 = stage_criterion(logits_sam, yb, soft_tb)
                    elif mix_active:
                        loss2 = _mixed_criterion(stage_criterion, logits_sam, ya_mix, yb_mix, lam_mix)
                    else:
                        loss2 = stage_criterion(logits_sam, yb)
                    loss2.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                    sam_wrapper.second_step()
                    did_opt_step = True
                else:
                    scaler.scale(loss).backward()
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

                    if stage_use_amp:
                        scale_before = float(scaler.get_scale())
                        scaler.step(opt)
                        scaler.update()
                        scale_after = float(scaler.get_scale())
                        did_opt_step = scale_after >= scale_before
                        if not did_opt_step:
                            amp_skipped_steps += 1
                    else:
                        opt.step()
                        did_opt_step = True

                if ema is not None:
                    ema.update()

                if did_opt_step and scheduler is not None and args.scheduler == "onecycle":
                    scheduler.step()

                train_losses.append(float(loss.detach().cpu().item()))
                pred = torch.argmax(logits.detach(), dim=-1)
                tr_correct += int((pred == yb_orig).sum().detach().cpu().item())
                tr_seen += int(yb_orig.numel())

            if scheduler is not None and args.scheduler == "cosine" and epoch >= int(args.warmup_epochs):
                scheduler.step()

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            train_acc = (tr_correct / max(1, tr_seen))

            if stage_use_amp and amp_skipped_steps:
                print(
                    f"note: AMP skipped {amp_skipped_steps} optimizer steps "
                    f"in stage={stage_name} epoch={global_epoch_offset + epoch + 1} (overflow/NaNs)"
                )
            if nonfinite_loss_batches:
                print(
                    f"note: skipped {nonfinite_loss_batches} batches with non-finite loss "
                    f"in stage={stage_name} epoch={global_epoch_offset + epoch + 1}"
                )

            # Always disable deploy-sim for the float validation pass.
            if deploy_sim_enabled:
                _set_deploy_sim(False)

            if ema is not None:
                with ema.average_parameters():
                    val_metrics = evaluate(model, val_loader, args.device, cfg.num_classes, criterion)
            else:
                val_metrics = evaluate(model, val_loader, args.device, cfg.num_classes, criterion)

            val_q_metrics = None
            if args.eval_quant and q_s_input is not None:
                if deploy_sim_enabled:
                    _set_deploy_sim(True)
                if ema is not None:
                    with ema.average_parameters():
                        val_q_metrics = evaluate(
                            model,
                            val_loader,
                            args.device,
                            cfg.num_classes,
                            criterion,
                            input_fakequant_s=float(q_s_input),
                        )
                else:
                    val_q_metrics = evaluate(
                        model,
                        val_loader,
                        args.device,
                        cfg.num_classes,
                        criterion,
                        input_fakequant_s=float(q_s_input),
                    )
                if deploy_sim_enabled:
                    _set_deploy_sim(False)

            val_loss = float(val_metrics["loss"])
            val_acc = float(val_metrics["acc"])
            val_macro_f1 = float(val_metrics["macro_f1"])
            val_q_loss = float(val_q_metrics["loss"]) if val_q_metrics is not None else None
            val_q_acc = float(val_q_metrics["acc"]) if val_q_metrics is not None else None
            val_q_macro_f1 = float(val_q_metrics["macro_f1"]) if val_q_metrics is not None else None

            lr_now = float(opt.param_groups[0]["lr"])
            global_epoch = int(global_epoch_offset + epoch + 1)

            history["epoch"].append(global_epoch)
            history["stage"].append(stage_name)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(val_macro_f1)
            history["lr"].append(lr_now)

            stage_history["epoch"].append(global_epoch)
            stage_history["stage"].append(stage_name)
            stage_history["train_loss"].append(train_loss)
            stage_history["val_loss"].append(val_loss)
            stage_history["train_acc"].append(train_acc)
            stage_history["val_acc"].append(val_acc)
            stage_history["val_macro_f1"].append(val_macro_f1)
            stage_history["lr"].append(lr_now)

            print(
                f"epoch {global_epoch}/{int(args.epochs)} "
                f"stage={stage_name} ({epoch+1}/{int(stage_epochs)}) "
                f"lr={lr_now:.2e} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_macro_f1:.4f}"
            )
            if val_q_metrics is not None:
                print(
                    f"          val_quant_acc={float(val_q_metrics['acc']):.4f} val_quant_f1={float(val_q_metrics['macro_f1']):.4f}"
                )

            # Always save last (raw weights)
            torch.save(
                {
                    "model_type": args.model,
                    "cfg": cfg.__dict__ if hasattr(cfg, '__dict__') else {},
                    "state_dict": model.state_dict(),
                    "ema": (ema.state_dict() if ema is not None else None),
                    "stage": {
                        "name": stage_name,
                        "global_epoch": global_epoch,
                        "stage_epoch": int(epoch + 1),
                        "stage_epochs": int(stage_epochs),
                    },
                    "quant": {
                        "enabled": bool(args.iqat or args.qat),
                        "iqat": bool(args.iqat),
                        "qat_requested": bool(args.qat),
                        "qat_active": bool(enable_qat),
                        "s_input": float(q_s_input) if q_s_input is not None else None,
                        "input_percentile": float(args.iqat_percentile),
                        "iqat_scale_jitter": float(args.iqat_scale_jitter),
                        "qat_momentum": float(args.qat_momentum),
                    },
                    "weights": {
                        "last_state_dict": "raw",
                        "deploy_preference": "ema" if (ema is not None) else "raw",
                    },
                },
                stage_last_path,
            )

            cur = _metric_value(
                metric=str(args.metric),
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_macro_f1,
                val_q_loss=val_q_loss,
                val_q_acc=val_q_acc,
                val_q_macro_f1=val_q_macro_f1,
            )
            if best_metric is None:
                improved = True
            else:
                improved = (cur > best_metric) if improved_dir_higher else (cur < best_metric)

            if improved:
                best_metric = float(cur)
                bad_epochs = 0
                ckpt_best = {
                    "model_type": args.model,
                    "cfg": cfg.__dict__ if hasattr(cfg, '__dict__') else {},
                    "ema": (ema.state_dict() if ema is not None else None),
                    "stage": {
                        "name": stage_name,
                        "global_epoch": global_epoch,
                        "stage_epoch": int(epoch + 1),
                        "stage_epochs": int(stage_epochs),
                        "best_metric": str(args.metric),
                        "best_metric_value": float(best_metric),
                    },
                    "quant": {
                        "enabled": bool(args.iqat or args.qat),
                        "iqat": bool(args.iqat),
                        "qat_requested": bool(args.qat),
                        "qat_active": bool(enable_qat),
                        "s_input": float(q_s_input) if q_s_input is not None else None,
                        "input_percentile": float(args.iqat_percentile),
                        "iqat_scale_jitter": float(args.iqat_scale_jitter),
                        "qat_momentum": float(args.qat_momentum),
                    },
                    "weights": {
                        "best_state_dict": "ema" if (ema is not None) else "raw",
                    },
                }
                if ema is None:
                    ckpt_best["state_dict"] = model.state_dict()
                else:
                    ckpt_best["state_dict_raw"] = model.state_dict()
                    with ema.average_parameters():
                        ckpt_best["state_dict"] = model.state_dict()
                torch.save(ckpt_best, stage_best_path)
                best_saved = True
            else:
                bad_epochs += 1

            if args.early_stop and args.early_stop > 0 and bad_epochs >= int(args.early_stop):
                print(f"early stop: stage={stage_name} no improvement on {args.metric} for {bad_epochs} epochs")
                break

        return {
            "stage": stage_name,
            "epochs_ran": int(len(stage_history["epoch"])),
            "best_path": str(stage_best_path),
            "last_path": str(stage_last_path),
            "best_saved": bool(best_saved),
            "best_metric": float(best_metric) if best_metric is not None else None,
            "history": stage_history,
        }

    # -------------------------
    # Two-phase training orchestration
    # -------------------------
    total_epochs = int(args.epochs)
    if total_epochs <= 0:
        raise SystemExit("--epochs must be > 0")

    stages_report: Dict[str, object] = {}

    if not bool(args.qat):
        # Single-stage training (backward-compatible)
        stage1 = _run_stage(
            stage_name="train",
            stage_epochs=total_epochs,
            global_epoch_offset=0,
            enable_qat=False,
            stage_best_path=best_path,
            stage_last_path=last_path,
        )
        stages_report["train"] = stage1
    else:
        if total_epochs < 2:
            raise SystemExit("--qat requires --epochs >= 2 (need pre-QAT stage + at least 1 QAT epoch)")

        # Reserve at least 1 epoch for QAT so the total budget is respected.
        pre_max = max(1, total_epochs - 1)
        print(f"quant: two-phase schedule => preqat_max_epochs={pre_max}, qat_min_epochs=1, total_epochs={total_epochs}")

        stage_pre = _run_stage(
            stage_name="preqat",
            stage_epochs=int(pre_max),
            global_epoch_offset=0,
            enable_qat=False,
            stage_best_path=preqat_best_path,
            stage_last_path=preqat_last_path,
        )
        stages_report["preqat"] = stage_pre

        pre_best_or_last = preqat_best_path if (bool(stage_pre.get("best_saved")) and preqat_best_path.exists()) else preqat_last_path
        if not pre_best_or_last.exists():
            raise SystemExit("pre-QAT stage produced no checkpoint; cannot start QAT")

        print(f"quant: starting QAT from pre-QAT checkpoint: {pre_best_or_last}")
        _load_state_for_stage(pre_best_or_last)

        # Enable QAT hooks and disable AMP for stability.
        assert qat_mgr is not None
        qat_mgr.attach_default(model)
        print(f"quant: QAT hooks enabled (momentum={args.qat_momentum}); AMP disabled")

        epochs_ran_pre = int(stage_pre.get("epochs_ran", 0))
        qat_epochs = max(1, total_epochs - epochs_ran_pre)
        qat_lr = float(args.lr) * float(args.qat_lr_mult) if float(args.qat_lr_mult) > 0 else 0.0
        if qat_lr > 0:
            print(f"quant: QAT lr={qat_lr:.2e} (--qat-lr-mult={args.qat_lr_mult})")
        stage_qat = _run_stage(
            stage_name="qat",
            stage_epochs=int(qat_epochs),
            global_epoch_offset=int(epochs_ran_pre),
            enable_qat=True,
            stage_best_path=best_path,
            stage_last_path=last_path,
            lr_override=qat_lr,
            enable_sam=False,    # SAM corrupts fake-quant observer statistics
            enable_rdrop=False,  # R-Drop second forward corrupts QAT observers
        )
        stages_report["qat"] = stage_qat

    # -------------------------
    # Self-distillation stage (optional)
    # -------------------------
    sd_epochs = int(args.self_distill_epochs)
    if sd_epochs > 0:
        # Load best checkpoint as teacher
        sd_load_path = best_path if best_path.exists() else last_path
        if not sd_load_path.exists():
            print("[warn] no checkpoint for self-distill, skipping")
        else:
            print(f"self-distill: loading teacher from {sd_load_path}")
            _load_state_for_stage(sd_load_path)

            # Generate soft labels from the teacher
            print("self-distill: generating soft labels from teacher...")
            soft_label_loader = DataLoader(
                NpzSequenceDataset(tr_path, max_samples=args.max_train, train=False, seed=int(args.seed)),
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin,
            )
            soft_probs, _ = _generate_soft_labels(
                model, soft_label_loader, args.device,
                temperature=float(args.self_distill_temp),
            )
            print(f"self-distill: soft labels generated, shape={soft_probs.shape}")

            # Create dataset with soft labels
            sd_train_ds = NpzSequenceDatasetWithSoftLabels(
                tr_path,
                soft_probs=soft_probs,
                max_samples=args.max_train,
                train=True,
                jitter=float(args.jitter),
                scale=float(args.scale),
                time_mask=float(args.time_mask),
                time_warp=float(args.time_warp),
                p_drop_gyro=float(args.p_drop_gyro),
                p_drop_acc=float(args.p_drop_acc),
                p_drop_axis=float(args.p_drop_axis),
                seed=int(args.seed) + 42,
            )

            # Create weighted sampler for distillation dataset
            sd_sampler = None
            sd_shuffle = True
            if args.sampler == "weighted":
                sd_y = sd_train_ds.y.astype(np.int64)
                sd_counts = np.bincount(sd_y, minlength=args.num_classes).astype(np.float64)
                sd_w_per_class = 1.0 / np.maximum(sd_counts, 1.0)
                sd_sample_w = sd_w_per_class[sd_y]
                sd_sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sd_sample_w).double(),
                    num_samples=len(sd_sample_w),
                    replacement=True,
                )
                sd_shuffle = False

            sd_train_loader = DataLoader(
                sd_train_ds,
                batch_size=args.batch,
                shuffle=sd_shuffle,
                sampler=sd_sampler,
                drop_last=drop_last_train,
                num_workers=args.num_workers,
                pin_memory=pin,
                persistent_workers=bool(args.num_workers > 0),
            )

            # Distillation loss
            distill_criterion = DistillationLoss(
                base_criterion=criterion,
                temperature=float(args.self_distill_temp),
                alpha=float(args.self_distill_alpha),
            ).to(args.device)

            sd_lr = float(args.lr) * float(args.self_distill_lr_mult)
            sd_best_path = args.out.with_name(args.out.stem + ".sd.pt")
            sd_last_path = args.out.with_name(args.out.stem + ".sd.last.pt")

            epochs_so_far = sum(
                int(v.get("epochs_ran", 0)) for v in stages_report.values() if isinstance(v, dict)
            )

            print(f"self-distill: starting {sd_epochs} epochs with lr={sd_lr:.2e}, "
                  f"temp={args.self_distill_temp}, alpha={args.self_distill_alpha}")

            stage_sd = _run_stage(
                stage_name="self_distill",
                stage_epochs=sd_epochs,
                global_epoch_offset=epochs_so_far,
                enable_qat=bool(args.qat),  # keep QAT hooks consistent
                stage_best_path=sd_best_path,
                stage_last_path=sd_last_path,
                lr_override=sd_lr,
                custom_criterion=distill_criterion,
                custom_train_loader=sd_train_loader,
                enable_sam=False,    # SAM too aggressive for distillation fine-tuning
                enable_rdrop=False,  # R-Drop not needed with soft labels
            )
            stages_report["self_distill"] = stage_sd

            # Use self-distill best as the new best
            if bool(stage_sd.get("best_saved")) and sd_best_path.exists():
                import shutil
                shutil.copy2(sd_best_path, best_path)
                print(f"self-distill: updated best checkpoint: {best_path}")
            elif sd_last_path.exists():
                import shutil
                shutil.copy2(sd_last_path, last_path)

    # Load best checkpoint for final reporting (prefer best, else last)
    load_path = best_path if best_path.exists() else last_path
    if not load_path.exists():
        raise SystemExit("no checkpoint saved")

    best_ckpt = torch.load(load_path, map_location="cpu")
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    model.to(args.device)

    # For final reporting: if checkpoint contains EMA, evaluate using EMA weights.
    ema_report: Optional[ExponentialMovingAverage] = None
    if ("ema" in best_ckpt) and (best_ckpt["ema"] is not None):
        try:
            ema_report = ExponentialMovingAverage(model.parameters(), decay=float(args.ema_decay))
            ema_report.load_state_dict(best_ckpt["ema"])
        except Exception as e:
            print(f"[warn] could not restore EMA for reporting: {e}")
            ema_report = None

    # Helper: choose between standard eval and TTA eval for final reporting
    def _final_eval(mdl, ldr, *, input_fq_s=None):
        tta_n = int(args.tta)
        if tta_n > 0:
            return evaluate_tta(
                mdl, ldr, args.device, cfg.num_classes, criterion,
                n_aug=tta_n, jitter=float(args.tta_jitter), scale=float(args.tta_scale),
                input_fakequant_s=input_fq_s,
            )
        return evaluate(
            mdl, ldr, args.device, cfg.num_classes, criterion,
            input_fakequant_s=input_fq_s,
        )

    if args.tta > 0:
        print(f"tta: n_aug={args.tta}, jitter={args.tta_jitter}, scale={args.tta_scale}")

    if ema_report is not None:
        with ema_report.average_parameters():
            final_val = _final_eval(model, val_loader)
    else:
        final_val = _final_eval(model, val_loader)
    final_val_q = None
    if args.eval_quant and q_s_input is not None:
        if ema_report is not None:
            with ema_report.average_parameters():
                final_val_q = _final_eval(model, val_loader, input_fq_s=float(q_s_input))
        else:
            final_val_q = _final_eval(model, val_loader, input_fq_s=float(q_s_input))

    test_metrics = None
    test_metrics_q = None
    if te_path is not None and te_path.exists():
        test_ds = NpzSequenceDataset(te_path)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            persistent_workers=bool(args.num_workers > 0),
        )
        if ema_report is not None:
            with ema_report.average_parameters():
                test_metrics = _final_eval(model, test_loader)
        else:
            test_metrics = _final_eval(model, test_loader)
        if args.eval_quant and q_s_input is not None:
            if ema_report is not None:
                with ema_report.average_parameters():
                    test_metrics_q = _final_eval(model, test_loader, input_fq_s=float(q_s_input))
            else:
                test_metrics_q = _final_eval(model, test_loader, input_fq_s=float(q_s_input))

    # Save plots + metrics
    _maybe_plot(history, plot_path)

    report = {
        "model_type": args.model,
        "cfg": cfg.__dict__ if hasattr(cfg, '__dict__') else {},
        "quant": {
            "enabled": bool(args.iqat or args.qat),
            "iqat": bool(args.iqat),
            "qat": bool(args.qat),
            "s_input": float(q_s_input) if q_s_input is not None else None,
            "input_percentile": float(args.iqat_percentile),
            "iqat_scale_jitter": float(args.iqat_scale_jitter),
            "qat_momentum": float(args.qat_momentum),
            "two_phase": bool(args.qat),
        },
        "paths": {
            "best": str(best_path),
            "last": str(last_path),
            "preqat_best": str(preqat_best_path) if bool(args.qat) else None,
            "preqat_last": str(preqat_last_path) if bool(args.qat) else None,
            "plot": str(plot_path),
        },
        "stages": stages_report,
        "history": history,
        "final_val": {
            "loss": float(final_val["loss"]),
            "acc": float(final_val["acc"]),
            "macro_f1": float(final_val["macro_f1"]),
            "per_class_acc": np.asarray(final_val["per_class_acc"]).tolist(),
            "confusion_matrix": np.asarray(final_val["confusion_matrix"]).tolist(),
            "n": int(final_val["n"]),
        },
        "final_val_quant": None,
        "final_test": None,
        "final_test_quant": None,
    }
    if final_val_q is not None:
        report["final_val_quant"] = {
            "loss": float(final_val_q["loss"]),
            "acc": float(final_val_q["acc"]),
            "macro_f1": float(final_val_q["macro_f1"]),
            "per_class_acc": np.asarray(final_val_q["per_class_acc"]).tolist(),
            "confusion_matrix": np.asarray(final_val_q["confusion_matrix"]).tolist(),
            "n": int(final_val_q["n"]),
        }
    if test_metrics is not None:
        report["final_test"] = {
            "loss": float(test_metrics["loss"]),
            "acc": float(test_metrics["acc"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "per_class_acc": np.asarray(test_metrics["per_class_acc"]).tolist(),
            "confusion_matrix": np.asarray(test_metrics["confusion_matrix"]).tolist(),
            "n": int(test_metrics["n"]),
        }
    if test_metrics_q is not None:
        report["final_test_quant"] = {
            "loss": float(test_metrics_q["loss"]),
            "acc": float(test_metrics_q["acc"]),
            "macro_f1": float(test_metrics_q["macro_f1"]),
            "per_class_acc": np.asarray(test_metrics_q["per_class_acc"]).tolist(),
            "confusion_matrix": np.asarray(test_metrics_q["confusion_matrix"]).tolist(),
            "n": int(test_metrics_q["n"]),
        }

    metrics_path.write_text(json.dumps(report, indent=2) + "\n")

    # also dump config json for convenience
    cfg_path = args.out.with_suffix(".json")
    cfg_path.write_text(json.dumps(cfg.__dict__, indent=2) + "\n")

    print(f"saved best: {best_path}")
    print(f"saved last: {last_path}")
    if bool(args.qat):
        print(f"saved preqat best: {preqat_best_path}")
        print(f"saved preqat last: {preqat_last_path}")
    print(f"metrics: {metrics_path}")
    if plot_path.exists():
        print(f"curves: {plot_path}")
    if te_path is not None:
        if test_metrics is None:
            print(f"test: skipped (missing {te_path})")
        else:
            print(f"test: acc={report['final_test']['acc']:.4f} f1={report['final_test']['macro_f1']:.4f}")
            if report.get("final_test_quant") is not None:
                print(f"test_quant: acc={report['final_test_quant']['acc']:.4f} f1={report['final_test_quant']['macro_f1']:.4f}")

    # Print total number of parameters for reference.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model total parameters: {total_params:,}")

    if qat_mgr is not None:
        qat_mgr.close()


if __name__ == "__main__":
    main()
