from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


INT8_QMIN = -128
INT8_QMAX = 127


def _maxabs(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def choose_symmetric_scale(x: np.ndarray, qmax: int = INT8_QMAX, *, eps: float = 1e-12) -> float:
    m = _maxabs(x)
    if m < eps:
        return 1.0 / qmax
    return m / float(qmax)


def quantize_symmetric_int8(x: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    q = np.rint(np.asarray(x, dtype=np.float64) / scale)
    q = np.clip(q, INT8_QMIN, INT8_QMAX)
    return q.astype(np.int8)


def quantize_per_out_channel_int8(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim < 1:
        raise ValueError("weights must have at least 1 dim")
    cout = w.shape[0]
    scales = np.empty((cout,), dtype=np.float32)
    wq = np.empty_like(w, dtype=np.int8)
    for o in range(cout):
        s = choose_symmetric_scale(w[o])
        scales[o] = s
        wq[o] = quantize_symmetric_int8(w[o], float(s))
    return wq, scales


def q31_from_float(x: float) -> int:
    # model.h expects Q31 stored in int32_t and converted by /2^31.
    # Clamp into [0, (2^31-1)/2^31).
    if x < 0:
        raise ValueError("expected non-negative scale")
    v = int(round(x * (1 << 31)))
    if v >= (1 << 31):
        v = (1 << 31) - 1
    return int(v)


@dataclass(frozen=True)
class MultShift:
    m: int  # int32
    r: int  # int32 (can be negative)


def mult_shift_from_real(real_scale: float) -> MultShift:

    if not math.isfinite(real_scale) or real_scale <= 0.0:
        return MultShift(m=0, r=0)

    mantissa, exp2 = math.frexp(real_scale)  # real_scale = mantissa * 2**exp2, mantissa in [0.5,1)
    m = int(round(mantissa * (1 << 31)))
    if m >= (1 << 31):
        m = (1 << 31) - 1

    r = 31 - exp2
    return MultShift(m=int(m), r=int(r))


def requant_params_per_channel(s_in: float, s_w: np.ndarray, s_out: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (M, R) int32 arrays for per-channel requant.

    real_scale[o] = (s_in * s_w[o]) / s_out
    """
    if s_in <= 0 or s_out <= 0:
        raise ValueError("s_in and s_out must be > 0")

    s_w = np.asarray(s_w, dtype=np.float64)
    M = np.empty_like(s_w, dtype=np.int32)
    R = np.empty_like(s_w, dtype=np.int32)
    for o in range(s_w.shape[0]):
        ms = mult_shift_from_real(float(s_in) * float(s_w[o]) / float(s_out))
        M[o] = np.int32(ms.m)
        R[o] = np.int32(ms.r)
    return M, R


def quantize_bias_int32(bias_f: np.ndarray, s_in: float, s_w: np.ndarray) -> np.ndarray:
    """Quantize bias into int32 accumulator units.

    b_q[o] = round( b_f[o] / (s_in * s_w[o]) )
    """
    if s_in <= 0:
        raise ValueError("s_in must be > 0")
    b = np.asarray(bias_f, dtype=np.float64)
    s_w = np.asarray(s_w, dtype=np.float64)
    if b.shape[0] != s_w.shape[0]:
        raise ValueError("bias and s_w must have same first dim")

    denom = float(s_in) * s_w
    denom = np.where(denom == 0, 1e-12, denom)
    bq = np.rint(b / denom)
    bq = np.clip(bq, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return bq.astype(np.int32)


def c_array_initializer(values: Iterable, *, values_per_line: int = 16) -> str:
    vals = list(values)
    if not vals:
        return "{}"
    lines = []
    for i in range(0, len(vals), values_per_line):
        chunk = vals[i : i + values_per_line]
        lines.append("    " + ", ".join(str(int(x)) if isinstance(x, (np.integer, int)) else str(x) for x in chunk))
    return "{\n" + ",\n".join(lines) + "\n}"


def c_float_array_initializer(values: Iterable[float], *, values_per_line: int = 8) -> str:
    vals = list(values)
    if not vals:
        return "{}"
    lines = []
    for i in range(0, len(vals), values_per_line):
        chunk = vals[i : i + values_per_line]
        lines.append("    " + ", ".join(f"{float(x):.8e}f" for x in chunk))
    return "{\n" + ",\n".join(lines) + "\n}"
