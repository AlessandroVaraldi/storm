from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# =====================================================================
# 1D-CNN Baseline  (~19.2 k params with in_ch=6, num_classes=8)
# =====================================================================

@dataclass
class CNN1DBaselineConfig:
    in_ch: int = 6
    num_classes: int = 8
    channels: tuple = (32, 48, 64)
    kernels: tuple = (7, 5, 3)


class CNN1DBaseline(nn.Module):
    """3-layer 1D-CNN with BatchNorm, ReLU, and global average pooling."""

    def __init__(self, cfg: CNN1DBaselineConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        ks = cfg.kernels
        layers = []
        in_c = cfg.in_ch
        for c, k in zip(ch, ks):
            layers += [
                nn.Conv1d(in_c, c, k, padding=k // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ]
            in_c = c
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(ch[-1], cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        x = self.features(x)            # [B, ch[-1], T]
        x = x.mean(dim=-1)              # global avg pool → [B, ch[-1]]
        return self.classifier(x)


# =====================================================================
# LSTM Baseline  (~19.5 k params with in_ch=6, num_classes=8)
# =====================================================================

@dataclass
class LSTMBaselineConfig:
    in_ch: int = 6
    hidden_size: int = 38
    num_layers: int = 2
    num_classes: int = 8
    dropout: float = 0.1


class LSTMBaseline(nn.Module):
    """2-layer LSTM; uses last-timestep hidden state for classification."""

    def __init__(self, cfg: LSTMBaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.in_ch,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        out, _ = self.lstm(x)            # [B, T, H]
        last = out[:, -1, :]             # [B, H]
        last = self.drop(last)
        return self.classifier(last)


# =====================================================================
# Transformer Baseline  (~19.9 k params with in_ch=6, num_classes=8)
#
# Accuracy-oriented: global attention, standard LayerNorm, GELU, dropout.
# No quantisation, no windowed attention, no integer ops.
# =====================================================================

@dataclass
class TransformerBaselineConfig:
    in_ch: int = 6
    d_model: int = 32
    nhead: int = 4
    depth: int = 2
    dim_feedforward: int = 64     # ffn_mult ≈ 2
    num_classes: int = 8
    dropout: float = 0.1
    stem_kernel: int = 13


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (no learnable parameters)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerBaseline(nn.Module):
    """Standard Transformer encoder for HAR, accuracy-oriented."""

    def __init__(self, cfg: TransformerBaselineConfig):
        super().__init__()
        self.cfg = cfg
        # Conv1d stem for temporal feature extraction
        self.stem = nn.Conv1d(cfg.in_ch, cfg.d_model,
                              kernel_size=cfg.stem_kernel,
                              padding=cfg.stem_kernel // 2)
        self.pos_enc = _SinusoidalPE(cfg.d_model)
        self.drop_in = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)   # [B, T, d]
        x = self.pos_enc(x)
        x = self.drop_in(x)
        x = self.encoder(x)                                 # [B, T, d]
        x = self.final_ln(x)
        x = x.mean(dim=1)                                   # GAP → [B, d]
        return self.classifier(x)


# =====================================================================
# Registry: name → (model_class, config_class)
# =====================================================================

BASELINE_REGISTRY = {
    "cnn1d":       (CNN1DBaseline, CNN1DBaselineConfig),
    "lstm":        (LSTMBaseline, LSTMBaselineConfig),
    "transformer": (TransformerBaseline, TransformerBaselineConfig),
}


def build_baseline(name: str, *, in_ch: int = 6, num_classes: int = 8) -> nn.Module:
    """Instantiate a baseline model by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name!r}. "
                         f"Available: {list(BASELINE_REGISTRY.keys())}")
    cls, cfg_cls = BASELINE_REGISTRY[name]
    cfg = cfg_cls(in_ch=in_ch, num_classes=num_classes)
    return cls(cfg)
