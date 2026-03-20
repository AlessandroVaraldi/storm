from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants — must match ker_layernorm_int.h
# ---------------------------------------------------------------------------
_FRAC = 14
_SCALE = 1 << _FRAC          # 16384
_TOTAL = _FRAC + _FRAC       # 28
_ROUND = 1 << (_TOTAL - 1)   # rounding bias


# ---------------------------------------------------------------------------
# Helpers — rsqrt LUT construction and lookup
# ---------------------------------------------------------------------------

def build_rsqrt_lut(
    n: int = 256,
    var_min: float = 1e-5,
    var_max: float = 127.0 ** 2,
) -> Tuple[torch.Tensor, float, float]:
    """Build a 1/sqrt(x) LUT with *n* entries, log-uniformly spaced.

    Returns
    -------
    lut : Tensor [n]   – float32 rsqrt values (≥ 0).
    log_min : float     – ln(var_min).
    log_max : float     – ln(var_max).
    """
    log_min = math.log(max(var_min, 1e-12))
    log_max = math.log(max(var_max, var_min * 2))
    log_bins = torch.linspace(log_min, log_max, n)
    var_bins = torch.exp(log_bins)
    lut = (1.0 / torch.sqrt(var_bins)).float()
    return lut, log_min, log_max


def lut_rsqrt(
    var: torch.Tensor,
    lut: torch.Tensor,
    log_min: float,
    log_max: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Evaluate 1/sqrt(var + eps) via table lookup (no interpolation).

    This mirrors the C kernel's behaviour: the float domain ``var`` is mapped
    to an integer index via its log, and the nearest table entry is returned.
    """
    var_safe = (var + eps).clamp(min=eps)
    log_var = torch.log(var_safe)

    n = lut.numel()
    idx_f = (log_var - log_min) / (log_max - log_min) * (n - 1)
    idx = idx_f.clamp(0, n - 1).long()

    return lut[idx]


# ---------------------------------------------------------------------------
# Integer-only LayerNorm (training simulation)
# ---------------------------------------------------------------------------

class IntegerLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "lut_size", "elementwise_affine", "sim_int8_stats", "sim_int_affine"]

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        lut_size: int = 256,
        elementwise_affine: bool = True,
        sim_int8_stats: bool = False,
        sim_int_affine: bool = False,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.lut_size = int(lut_size)
        self.elementwise_affine = bool(elementwise_affine)
        self.sim_int8_stats = bool(sim_int8_stats)
        self.sim_int_affine = bool(sim_int_affine)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # Scale used when sim_int8_stats=True.  Set from outside (e.g. QAT).
        self.scale_x: float = 1.0
        # Scale used when sim_int_affine=True  (output requantisation scale).
        self.scale_y: float = 1.0

        # ----- build rsqrt LUT -----
        lut, log_min, log_max = build_rsqrt_lut(
            n=self.lut_size,
            var_min=self.eps,
            var_max=127.0 ** 2,
        )
        self.register_buffer("rsqrt_lut", lut)
        self.register_buffer("lut_log_min", torch.tensor(log_min))
        self.register_buffer("lut_log_max", torch.tensor(log_max))

    # ------------------------------------------------------------------
    def _lut_rsqrt(self, var: torch.Tensor) -> torch.Tensor:
        return lut_rsqrt(
            var,
            self.rsqrt_lut,
            float(self.lut_log_min),
            float(self.lut_log_max),
            eps=self.eps,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))

        if self.sim_int8_stats and self.scale_x > 0:
            # Simulate int32 accumulation from int8 values (matches C kernel).
            sx = float(self.scale_x)
            x_q = torch.round(x / sx).clamp(-127, 127)  # fake int8
            mean_i = x_q.mean(dim=dims, keepdim=True)
            var_i = x_q.var(dim=dims, unbiased=False, keepdim=True)
            mean = mean_i * sx
            var = var_i * (sx * sx)
        else:
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)

        # LUT-based rsqrt (forward) + exact rsqrt (gradient via STE)
        inv_std_lut = self._lut_rsqrt(var)
        inv_std_exact = torch.rsqrt(var + self.eps)
        inv_std = inv_std_exact + (inv_std_lut - inv_std_exact).detach()

        # ---------- Affine transform ----------
        if self.sim_int_affine and self.scale_x > 0 and self.scale_y > 0:
            # Simulate Q14 fixed-point inner loop (matches C kernel exactly).
            # Forward: quantised affine.   Gradient: STE (exact float affine).
            y_int = self._q14_affine(x, mean, inv_std)
            y_exact = self._float_affine(x, mean, inv_std)
            # STE: forward uses int path, backward uses float path
            y = y_exact + (y_int - y_exact).detach()
        else:
            y = self._float_affine(x, mean, inv_std)

        return y

    # ------------------------------------------------------------------
    def _float_affine(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
    ) -> torch.Tensor:
        """Standard float affine: (x - mean) * inv_std * gamma + beta."""
        y = (x - mean) * inv_std
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _q14_affine(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
    ) -> torch.Tensor:
        sx = float(self.scale_x)
        sy = float(self.scale_y)
        inv_sy = 1.0 / sy

        # Precomputed per-channel coefficients [C]
        gamma_f = self.weight if self.weight is not None else torch.ones_like(mean.squeeze())
        beta_f = self.bias if self.bias is not None else torch.zeros_like(mean.squeeze())

        gamma_q14 = torch.round(gamma_f * _SCALE).long()      # [C]
        beta_q14 = torch.round(beta_f * inv_sy * _SCALE).long()  # [C]

        # Fake-quantise input to int8
        x_q = torch.round(x / sx).clamp(-127, 127).long()  # [..., C]

        # Per-token scalars: inv_std is [batch..., 1] float
        K_f = inv_std * sx * inv_sy       # [..., 1]
        Km_f = inv_std * mean * inv_sy    # [..., 1]

        K_q14 = torch.round(K_f * _SCALE).long()   # [..., 1]
        Km_q14 = torch.round(Km_f * _SCALE).long()  # [..., 1]

        # Integer inner loop (vectorised over channels)
        centered = K_q14 * x_q - Km_q14                          # Q14, int64
        val = gamma_q14 * centered + (beta_q14 << _FRAC)         # Q28, int64
        y_raw = (val + _ROUND) >> _TOTAL                         # Q0

        # Saturate to int8
        y_raw = y_raw.clamp(-128, 127)

        # Convert back to float in the output scale
        return (y_raw.float() * sy)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"lut_size={self.lut_size}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"sim_int8_stats={self.sim_int8_stats}, "
            f"sim_int_affine={self.sim_int_affine}"
        )


# ---------------------------------------------------------------------------
# Utility: replace nn.LayerNorm → IntegerLayerNorm in-place
# ---------------------------------------------------------------------------

def replace_layernorm_with_integer(
    module: nn.Module,
    *,
    lut_size: int = 256,
    sim_int8_stats: bool = False,
    sim_int_affine: bool = False,
) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm) and not isinstance(child, IntegerLayerNorm):
            int_ln = IntegerLayerNorm(
                child.normalized_shape,
                eps=child.eps,
                lut_size=lut_size,
                elementwise_affine=child.elementwise_affine,
                sim_int8_stats=sim_int8_stats,
                sim_int_affine=sim_int_affine,
            )
            # Copy learned parameters
            if child.elementwise_affine:
                int_ln.weight.data.copy_(child.weight.data)
                int_ln.bias.data.copy_(child.bias.data)
            setattr(module, name, int_ln)
        else:
            replace_layernorm_with_integer(
                child, lut_size=lut_size, sim_int8_stats=sim_int8_stats,
                sim_int_affine=sim_int_affine,
            )
    return module
