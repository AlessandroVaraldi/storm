from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_INT8_QMIN = -127.0
_INT8_QMAX = 127.0


def _fake_quant_dequant_symm_int8(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # x_hat = clamp(round(x/s),[-127,127]) * s
    s = torch.clamp(s.to(dtype=torch.float32), min=1e-12)
    x_fp32 = x.to(torch.float32)
    q = torch.round(x_fp32 / s).clamp(_INT8_QMIN, _INT8_QMAX)
    return (q * s).to(dtype=x.dtype)


def fake_quant_dequant_ste(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    x_hat = _fake_quant_dequant_symm_int8(x, s)
    return x + (x_hat - x).detach()


def _per_out_channel_scale(w: torch.Tensor) -> torch.Tensor:
    # Returns per-out-channel symmetric scale for int8.
    # Linear: w [Cout,Cin] => reduce over dim=1
    # Conv1d: w [Cout,Cin/groups,K] => reduce over dims 1,2
    w_fp32 = w.detach().to(torch.float32)
    if w_fp32.ndim == 2:
        maxabs = w_fp32.abs().amax(dim=1)
    elif w_fp32.ndim == 3:
        maxabs = w_fp32.abs().amax(dim=(1, 2))
    else:
        raise ValueError(f"unsupported weight rank {w_fp32.ndim}")
    return torch.clamp(maxabs / 127.0, min=1e-12)


def fake_quant_weight_per_out_channel_ste(w: torch.Tensor) -> torch.Tensor:
    sw = _per_out_channel_scale(w)
    if w.ndim == 2:
        sw_b = sw[:, None]
    else:
        sw_b = sw[:, None, None]
    w_hat = _fake_quant_dequant_symm_int8(w, sw_b)
    return w + (w_hat - w).detach()


class QuantLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_weight_quant = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not bool(self.enable_weight_quant):
            return F.linear(input, self.weight, self.bias)
        wq = fake_quant_weight_per_out_channel_ste(self.weight)
        return F.linear(input, wq, self.bias)


class QuantConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_weight_quant = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not bool(self.enable_weight_quant):
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        wq = fake_quant_weight_per_out_channel_ste(self.weight)
        return F.conv1d(
            input,
            wq,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _replace_child(parent: nn.Module, name: str, new_child: nn.Module) -> None:
    setattr(parent, name, new_child)


def replace_linear_conv_with_quant(m: nn.Module) -> nn.Module:
    for name, child in list(m.named_children()):
        new_child: Optional[nn.Module] = None
        if isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
            q = QuantLinear(child.in_features, child.out_features, bias=(child.bias is not None))
            q.load_state_dict(child.state_dict(), strict=True)
            new_child = q
        elif isinstance(child, nn.Conv1d) and not isinstance(child, QuantConv1d):
            q = QuantConv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
            )
            q.load_state_dict(child.state_dict(), strict=True)
            new_child = q

        if new_child is not None:
            _replace_child(m, name, new_child)
            child = new_child

        replace_linear_conv_with_quant(child)

    return m


def set_weight_quant_enabled(m: nn.Module, enabled: bool) -> None:
    for mod in m.modules():
        if hasattr(mod, "enable_weight_quant"):
            try:
                setattr(mod, "enable_weight_quant", bool(enabled))
            except Exception:
                pass


_LUT_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


def _parse_int16_lut_from_header(header_path: Path, *, array_name: str) -> torch.Tensor:
    txt = header_path.read_text()

    # Find the declaration and size.
    decl = re.search(
        rf"static\s+const\s+int16_t\s+{re.escape(array_name)}\s*\[\s*(\d+)\s*\]\s*=",
        txt,
    )
    if not decl:
        raise ValueError(f"could not find array declaration {array_name} in {header_path}")
    n = int(decl.group(1))

    # Find the initializer braces for this declaration.
    brace_open = txt.find("{", decl.end())
    if brace_open < 0:
        raise ValueError(f"could not find '{{' for {array_name} in {header_path}")
    brace_close = txt.find("};", brace_open)
    if brace_close < 0:
        raise ValueError(f"could not find '}};' for {array_name} in {header_path}")

    body = txt[brace_open + 1 : brace_close]
    ints = [int(v) for v in re.findall(r"-?\d+", body)]
    if len(ints) != n:
        raise ValueError(f"array {array_name} expected {n} entries, got {len(ints)}")
    return torch.tensor(ints, dtype=torch.int16)


def get_lut(array_name: str, *, device: torch.device) -> torch.Tensor:
    # LUT headers live in the luts/ folder next to this script.
    luts_dir = Path(__file__).resolve().parent.parent / "deployment" / "luts"
    if array_name == "lut_gelu":
        header = luts_dir / "lut_gelu.h"
    elif array_name == "lut_sigmoid":
        header = luts_dir / "lut_sigmoid.h"
    else:
        raise KeyError(f"unknown LUT {array_name}")

    key = (str(header), array_name)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = _parse_int16_lut_from_header(header, array_name=array_name)

    return _LUT_CACHE[key].to(device=device, non_blocking=True)


def index_mapping_params(s_in: float, *, xmin: float, xmax: float, L: int, rshift: int) -> Tuple[int, int, int]:
    # Match export_model_header.index_mapping_params
    index_scale = (L - 1) / (xmax - xmin)
    alpha = int(round(float(s_in) * index_scale * (1 << rshift)))
    beta = int(round((-xmin) * index_scale * (1 << rshift)))
    return alpha, beta, rshift


def _lut_eval_q15(
    q_int: torch.Tensor,
    *,
    s_in: float,
    lut_q15: torch.Tensor,
    xmin: float,
    xmax: float,
    rshift: int = 12,
) -> torch.Tensor:
    # q_int: int32/int64 tensor with range [-127,127]
    alpha, beta, rshift = index_mapping_params(s_in, xmin=xmin, xmax=xmax, L=int(lut_q15.numel()), rshift=int(rshift))
    idx = (q_int.to(torch.int64) * int(alpha) + int(beta)) >> int(rshift)
    idx = idx.clamp(0, int(lut_q15.numel()) - 1).to(torch.long)
    y_q15 = lut_q15[idx].to(torch.int32)
    return y_q15


@dataclass
class DeploySimOps:
    device: torch.device
    enable_lut: bool = True
    enable_softmax_q15: bool = True

    def __post_init__(self) -> None:
        self._lut_gelu = get_lut("lut_gelu", device=self.device)
        self._lut_sigmoid = get_lut("lut_sigmoid", device=self.device)

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_lut:
            return F.gelu(x)

        # Estimate input scale and quantize to int8
        s_in_t = torch.clamp(x.detach().to(torch.float32).abs().amax() / 127.0, min=1e-12)
        s_in = float(s_in_t.cpu().item())
        q = torch.round(x.to(torch.float32) / float(s_in)).clamp(_INT8_QMIN, _INT8_QMAX).to(torch.int32)

        y_q15 = _lut_eval_q15(q, s_in=s_in, lut_q15=self._lut_gelu, xmin=-8.0, xmax=8.0, rshift=12)
        y = (y_q15.to(torch.float32) / 32768.0).to(dtype=x.dtype)

        # Requantize output to int8-like grid (STE) to mimic integer-only gelu_lut_q15
        s_out = torch.clamp(y.detach().to(torch.float32).abs().amax() / 127.0, min=1e-12)
        return fake_quant_dequant_ste(y, s_out)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_lut:
            return F.silu(x)

        s_in_t = torch.clamp(x.detach().to(torch.float32).abs().amax() / 127.0, min=1e-12)
        s_in = float(s_in_t.cpu().item())
        q = torch.round(x.to(torch.float32) / float(s_in)).clamp(_INT8_QMIN, _INT8_QMAX).to(torch.int32)

        sig_q15 = _lut_eval_q15(q, s_in=s_in, lut_q15=self._lut_sigmoid, xmin=-8.0, xmax=8.0, rshift=12)
        sig = (sig_q15.to(torch.float32) / 32768.0).to(dtype=x.dtype)
        real_x = (q.to(torch.float32) * float(s_in)).to(dtype=x.dtype)
        y = real_x * sig

        s_out = torch.clamp(y.detach().to(torch.float32).abs().amax() / 127.0, min=1e-12)
        return fake_quant_dequant_ste(y, s_out)

    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        w = torch.softmax(x, dim=dim)
        if not self.enable_softmax_q15:
            return w
        wq = torch.round(w.to(torch.float32) * 32768.0).clamp(0.0, 32767.0)
        w_hat = (wq / 32768.0).to(dtype=w.dtype)
        return w + (w_hat - w).detach()