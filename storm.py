from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.int_layernorm import IntegerLayerNorm


def _make_ln(cfg: "STORMConfig", dim: int, eps: float = 1e-5) -> nn.Module:
    if cfg.int_layernorm:
        return IntegerLayerNorm(dim, eps=eps, lut_size=cfg.int_ln_lut_size)
    return nn.LayerNorm(dim, eps=eps)


class DefaultOps:

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.softmax(x, dim=dim)


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


@dataclass
class STORMConfig:
    in_ch: int = 6
    d_model: int = 64
    nhead: int = 4
    depth: int = 2
    ffn_mult: int = 2
    num_classes: int = 8
    attention_window: int = 0  # 0 => global

    stem_kernel: int = 5
    posmix_kernel: int = 3
    attnpool_hidden: int = 32

    # Integer-only LayerNorm (LUT rsqrt approximation)
    int_layernorm: bool = False
    int_ln_lut_size: int = 256

    # Regularization
    drop_path_rate: float = 0.0   # stochastic depth (0 = disabled)
    feat_dropout: float = 0.0     # dropout before classifier (0 = disabled)


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.ops = DefaultOps()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [B,T,C]
        out: Dict[str, torch.Tensor] = {}
        h1 = self.fc1(x)
        out["fc1_out"] = h1
        h1a = self.ops.gelu(h1)
        out["gelu_out"] = h1a
        y = self.fc2(h1a)
        out["fc2_out"] = y
        return y, out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: STORMConfig, drop_path: float = 0.0):
        super().__init__()
        assert cfg.d_model % cfg.nhead == 0
        self.cfg = cfg
        self.ops = DefaultOps()
        self.ln_mha = _make_ln(cfg, cfg.d_model, eps=1e-5)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.ln_mlp = _make_ln(cfg, cfg.d_model, eps=1e-5)
        self.mlp = MLPBlock(cfg.d_model, cfg.ffn_mult * cfg.d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [B,T,C]
        aux: Dict[str, torch.Tensor] = {}

        z = self.ln_mha(x)
        aux["ln_mha_out"] = z
        qkv = self.qkv(z)
        aux["qkv_out"] = qkv
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Multi-head attention (global, windowed optional)
        B, T, C = x.shape
        dh = C // self.cfg.nhead
        q = q.view(B, T, self.cfg.nhead, dh).transpose(1, 2)  # [B,H,T,dh]
        k = k.view(B, T, self.cfg.nhead, dh).transpose(1, 2)
        v = v.view(B, T, self.cfg.nhead, dh).transpose(1, 2)

        if self.cfg.attention_window and self.cfg.attention_window > 0:
            # naive window mask
            w = self.cfg.attention_window
            idx = torch.arange(T, device=x.device)
            dist = (idx[None, :] - idx[:, None]).abs()
            mask = dist > w  # [T,T]
            scores = torch.matmul(q, k.transpose(-2, -1)) / (dh**0.5)  # [B,H,T,T]
            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
            attn = self.ops.softmax(scores, dim=-1)
        else:
            attn = self.ops.softmax(torch.matmul(q, k.transpose(-2, -1)) / (dh**0.5), dim=-1)

        ctx = torch.matmul(attn, v)  # [B,H,T,dh]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        aux["ctx_out"] = ctx
        y = self.proj(ctx)
        aux["proj_out"] = y
        x = x + self.drop_path(y)
        aux["mha_out"] = x

        z2 = self.ln_mlp(x)
        aux["ln_mlp_out"] = z2
        mlp_out, mlp_aux = self.mlp(z2)
        for kname, tval in mlp_aux.items():
            aux[f"mlp_{kname}"] = tval
        x = x + self.drop_path(mlp_out)
        aux["mlp_out"] = x
        return x, aux


class AttnPool(nn.Module):
    def __init__(self, cfg: STORMConfig):
        super().__init__()
        self.ops = DefaultOps()
        self.ln = _make_ln(cfg, cfg.d_model, eps=1e-5)
        self.fc0 = nn.Linear(cfg.d_model, cfg.attnpool_hidden)
        self.fc1 = nn.Linear(cfg.attnpool_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [B,T,C]
        aux: Dict[str, torch.Tensor] = {}
        z = self.ln(x)
        aux["ln_out"] = z
        h = self.fc0(z)
        aux["fc0_out"] = h
        h = self.ops.gelu(h)
        aux["gelu_out"] = h
        s = self.fc1(h).squeeze(-1)  # [B,T]
        aux["scores_out"] = s
        w = self.ops.softmax(s, dim=-1)  # [B,T]
        aux["weights_out"] = w
        feat = torch.einsum("bt,btc->bc", w, z)  # weighted sum of LN output
        aux["feat_out"] = feat
        return feat, aux


class STORM(nn.Module):
    def __init__(self, cfg: STORMConfig):
        super().__init__()
        self.cfg = cfg
        self.ops = DefaultOps()
        self.stem = nn.Conv1d(cfg.in_ch, cfg.d_model, kernel_size=cfg.stem_kernel, stride=1, padding=cfg.stem_kernel // 2, dilation=1)
        self.posmix = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=cfg.posmix_kernel, stride=1, padding=cfg.posmix_kernel // 2, dilation=1, groups=cfg.d_model)

        # Stochastic depth: linearly increasing drop rate per block
        dpr = [cfg.drop_path_rate * i / max(cfg.depth - 1, 1) for i in range(cfg.depth)]
        self.blocks = nn.ModuleList([TransformerBlock(cfg, drop_path=dpr[i]) for i in range(cfg.depth)])
        self.final_ln = _make_ln(cfg, cfg.d_model, eps=1e-5)
        self.attnpool = AttnPool(cfg)
        self.classifier_ln = _make_ln(cfg, cfg.d_model, eps=1e-5)
        self.feat_drop = nn.Dropout(cfg.feat_dropout) if cfg.feat_dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor, *, return_intermediates: bool = False):
        # x: [B,T,Cin]
        aux: Dict[str, torch.Tensor] = {}

        # stem conv expects [B,Cin,T]
        y = self.stem(x.transpose(1, 2)).transpose(1, 2)  # [B,T,C]
        aux["stem_pre_silu"] = y
        y = self.ops.silu(y)
        aux["stem_out"] = y

        # posmix depthwise conv
        y_dw = self.posmix(y.transpose(1, 2)).transpose(1, 2)
        aux["posmix_dw_out"] = y_dw
        y = y + y_dw
        aux["posmix_out"] = y

        for i, blk in enumerate(self.blocks):
            y, blk_aux = blk(y)
            for k, v in blk_aux.items():
                aux[f"blk{i}_{k}"] = v

        y = self.final_ln(y)
        aux["final_norm_out"] = y

        feat, ap_aux = self.attnpool(y)
        for k, v in ap_aux.items():
            aux[f"attnpool_{k}"] = v

        aux["classifier_in"] = feat
        feat2 = self.classifier_ln(feat)
        feat2 = self.feat_drop(feat2)
        aux["classifier_ln_out"] = feat2
        logits = self.classifier(feat2)
        aux["classifier_out"] = logits

        if return_intermediates:
            return logits, aux
        return logits

    def set_ops(self, ops) -> None:
        self.ops = ops
        for m in self.modules():
            if hasattr(m, "ops"):
                setattr(m, "ops", ops)