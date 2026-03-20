from __future__ import annotations

import argparse
from email import header
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import nullcontext

import numpy as np
import torch

from utils.quant_utils import (
    MultShift,
    c_array_initializer,
    c_float_array_initializer,
    choose_symmetric_scale,
    mult_shift_from_real,
    q31_from_float,
    quantize_bias_int32,
    quantize_per_out_channel_int8,
    quantize_symmetric_int8,
    requant_params_per_channel,
)
from storm import TinyTransformerHAR, TinyTransformerHARConfig
from utils.int_layernorm import build_rsqrt_lut


@dataclass
class ExportScales:
    s_input: float
    s_resid: float
    s_final: float
    s_mlp_fc1_out: float
    s_mlp_gelu_out: float
    s_ap_fc0_out: float
    s_ap_gelu_out: float
    s_ap_fc1_out: float
    s_cls_in: float


def _json_from_npz_scalar(v: object) -> Dict[str, object]:
    if isinstance(v, np.ndarray):
        if v.shape == ():
            v = v.item()
        else:
            raise ValueError("expected scalar NPZ field for embedded JSON meta")
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    if not isinstance(v, str):
        raise ValueError(f"expected JSON string, got {type(v)!r}")
    obj = json.loads(v)
    if not isinstance(obj, dict):
        raise ValueError("embedded meta JSON must decode to an object")
    return obj


def load_standardization_from_meta(
    *,
    calib_npz: Path,
    meta_json: Optional[Path],
    expected_channels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    meta: Optional[Dict[str, object]] = None

    if meta_json is not None:
        meta_obj = json.loads(meta_json.read_text())
        if not isinstance(meta_obj, dict):
            raise ValueError(f"{meta_json} does not contain a JSON object")
        meta = meta_obj
    else:
        calib = np.load(calib_npz, allow_pickle=False)
        if "meta" in calib.files:
            meta = _json_from_npz_scalar(calib["meta"])

    if meta is None:
        raise SystemExit(
            "could not load standardization params: provide --meta-json or use a calib NPZ "
            "that contains the embedded 'meta' field from create_dataset.py"
        )

    std_meta = meta.get("standardization", None)
    if not isinstance(std_meta, dict):
        raise SystemExit("meta is missing 'standardization' object")

    mean = np.asarray(std_meta.get("mean", []), dtype=np.float32)
    std = np.asarray(std_meta.get("std", []), dtype=np.float32)

    if mean.shape != (expected_channels,) or std.shape != (expected_channels,):
        raise SystemExit(
            f"expected standardization mean/std of shape ({expected_channels},), "
            f"got mean={mean.shape}, std={std.shape}"
        )

    std = np.maximum(std, np.float32(1e-12))
    return mean, std


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _scale_from_maxabs(maxabs: float) -> float:
    if not math.isfinite(maxabs) or maxabs <= 0:
        return 1.0 / 127.0
    return maxabs / 127.0


def _fake_quant_dequant_np(x: np.ndarray, s: float) -> np.ndarray:
    if not math.isfinite(float(s)) or float(s) <= 0.0:
        return x
    q = np.round(x / float(s))
    q = np.clip(q, -127.0, 127.0)
    return (q * float(s)).astype(np.float32)


def _stat_from_absmax_list(values: List[float], *, method: str, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if method == "maxabs":
        return float(arr.max())
    if method == "percentile":
        return float(np.percentile(arr, float(percentile)))
    raise ValueError(f"unknown method {method}")


def collect_activation_stats(
    model: TinyTransformerHAR,
    data: np.ndarray,
    device: str,
    *,
    batch: int = 64,
    method: str = "percentile",
    percentile: float = 99.9,
) -> Dict[str, float]:
    """Collect per-tensor activation stats from model intermediates.

    Implementation detail: records per-batch maxabs and then reduces with
    either max (maxabs) or percentile over those per-batch maxima.
    """
    model.eval()
    per_key_batch_max: Dict[str, List[float]] = {}

    with torch.no_grad():
        for i in range(0, data.shape[0], batch):
            xb = torch.from_numpy(data[i : i + batch]).to(device)
            _, aux = model(xb, return_intermediates=True)
            for k, v in aux.items():
                arr = _np(v)
                m = float(np.max(np.abs(arr))) if arr.size else 0.0
                per_key_batch_max.setdefault(k, []).append(m)

    out: Dict[str, float] = {}
    for k, vals in per_key_batch_max.items():
        out[k] = _stat_from_absmax_list(vals, method=method, percentile=percentile)
    return out


def collect_maxabs(model: TinyTransformerHAR, data: np.ndarray, device: str, *, batch: int = 64) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}

    def upd(name: str, arr: np.ndarray) -> None:
        m = float(np.max(np.abs(arr))) if arr.size else 0.0
        out[name] = max(out.get(name, 0.0), m)

    with torch.no_grad():
        for i in range(0, data.shape[0], batch):
            xb = torch.from_numpy(data[i : i + batch]).to(device)
            _, aux = model(xb, return_intermediates=True)
            for k, v in aux.items():
                # Skip logits for scale purposes except classifier_out
                upd(k, _np(v))
    return out


def choose_export_scales(cfg: TinyTransformerHARConfig, maxabs: Dict[str, float]) -> ExportScales:
    # Input scale from dataset maxabs is provided via maxabs["__input__"]
    s_input = _scale_from_maxabs(maxabs.get("__input__", 1.0))

    # Residual-chain scale: use posmix_out maxabs (fallback to stem_out)
    s_resid = _scale_from_maxabs(maxabs.get("posmix_out", maxabs.get("stem_out", 1.0)))

    # Final LN output (used by AttnPool LN and also good as softmax scale for AttnPool scores)
    s_final = _scale_from_maxabs(maxabs.get("final_norm_out", s_resid * 127.0))

    # MLP hidden scales (shared across blocks due to single GELU_* macros)
    mlp_fc1_keys = [f"blk{i}_mlp_fc1_out" for i in range(cfg.depth)]
    mlp_gelu_keys = [f"blk{i}_mlp_gelu_out" for i in range(cfg.depth)]
    mlp_fc1_max = max((maxabs.get(k, 0.0) for k in mlp_fc1_keys), default=0.0)
    mlp_gelu_max = max((maxabs.get(k, 0.0) for k in mlp_gelu_keys), default=0.0)
    s_mlp_fc1_out = _scale_from_maxabs(mlp_fc1_max if mlp_fc1_max > 0 else (s_resid * 127.0))
    s_mlp_gelu_out = _scale_from_maxabs(mlp_gelu_max if mlp_gelu_max > 0 else (s_resid * 127.0))

    # AttnPool hidden scales
    s_ap_fc0_out = _scale_from_maxabs(maxabs.get("attnpool_fc0_out", s_resid * 127.0))
    s_ap_gelu_out = _scale_from_maxabs(maxabs.get("attnpool_gelu_out", s_resid * 127.0))

    # AttnPool score scale (int8 logits scale). We intentionally tie this to s_final so that
    # AP_SOFTMAX_SCALE can simply be s_final (matches the C implementation’s intent).
    s_ap_fc1_out = s_final

    # Pooled feature scale: weighted sum of ZLN, so it naturally stays in s_final
    s_cls_in = s_final

    return ExportScales(
        s_input=s_input,
        s_resid=s_resid,
        s_final=s_final,
        s_mlp_fc1_out=s_mlp_fc1_out,
        s_mlp_gelu_out=s_mlp_gelu_out,
        s_ap_fc0_out=s_ap_fc0_out,
        s_ap_gelu_out=s_ap_gelu_out,
        s_ap_fc1_out=s_ap_fc1_out,
        s_cls_in=s_cls_in,
    )


def index_mapping_params(s_in: float, *, xmin: float, xmax: float, L: int, rshift: int) -> Tuple[int, int, int]:
    # idx ≈ ((x_q*s_in - xmin)/(xmax-xmin))*(L-1)
    index_scale = (L - 1) / (xmax - xmin)
    alpha = int(round(s_in * index_scale * (1 << rshift)))
    beta = int(round((-xmin) * index_scale * (1 << rshift)))
    return alpha, beta, rshift


def write_model_h(
    out_path: Path,
    *,
    cfg: TinyTransformerHARConfig,
    weights: Dict[str, Tuple[str, np.ndarray]],
    conv_defs: List[Tuple[int, int, int, int, int, int, int, int, str]],
    fc_defs: List[Tuple[int, int, int, str]],
    ln_params: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    act_scales: List[float],
    act_names: List[str],
    silu_params: Tuple[int, int, int],
    gelu_params: Tuple[int, int, int, int, int],
    ap_gelu_params: Tuple[int, int, int, int, int],
    mhsa_softmax_scale: float,
    ap_softmax_scale: float,
    s_input: float,
    preproc_mean: np.ndarray,
    preproc_std: np.ndarray,
    preproc_invstd: np.ndarray,
    preproc_invstd_over_sinput: np.ndarray,
    preproc_bias_q: np.ndarray,
    meta_notes: str,
    flash_only_section: bool = False,
    rsqrt_lut: Optional[Tuple[np.ndarray, float, float]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    act_scales_q31 = [q31_from_float(s) for s in act_scales]

    header = []
    header.append("// Auto-generated by export_model_header.py — DO NOT EDIT\n")
    header.append("#pragma once\n")
    header.append("#include <stdint.h>\n\n")

    meta = {
        "version": 1,
        "model": "TinyTransformer-HAR",
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "depth": cfg.depth,
        "ffn_mult": cfg.ffn_mult,
        "in_ch": cfg.in_ch,
        "num_classes": cfg.num_classes,
        "attn_window": cfg.attention_window,
        "s_input": float(s_input),
        "force_power2_scales": False,
        "act_scales_names": act_names,
        "act_scales_values": act_scales,
        "act_params": {
            "silu_alpha": silu_params[0],
            "silu_beta": silu_params[1],
            "silu_rshift": silu_params[2],
            "gelu_alpha": gelu_params[0],
            "gelu_beta": gelu_params[1],
            "gelu_rshift": gelu_params[2],
            "gelu_M_out": gelu_params[3],
            "gelu_R_out": gelu_params[4],
            "ap_gelu_alpha": ap_gelu_params[0],
            "ap_gelu_beta": ap_gelu_params[1],
            "ap_gelu_rshift": ap_gelu_params[2],
            "ap_gelu_M_out": ap_gelu_params[3],
            "ap_gelu_R_out": ap_gelu_params[4],
            "mhsa_softmax_scale": mhsa_softmax_scale,
            "ap_softmax_scale": ap_softmax_scale,
        },
        "preprocessing": {
            "layout": ["ax", "ay", "az", "gx", "gy", "gz"],
            "mean": [float(x) for x in np.asarray(preproc_mean).reshape(-1).tolist()],
            "std": [float(x) for x in np.asarray(preproc_std).reshape(-1).tolist()],
            "invstd": [float(x) for x in np.asarray(preproc_invstd).reshape(-1).tolist()],
            "invstd_over_sinput": [float(x) for x in np.asarray(preproc_invstd_over_sinput).reshape(-1).tolist()],
            "bias_q": [float(x) for x in np.asarray(preproc_bias_q).reshape(-1).tolist()],
            "formula_q": "q = clamp(round(raw_i16 * unit_per_lsb * invstd_over_sinput[c] + bias_q[c]), -127, 127)",
        },
        "notes": meta_notes,
    }
    header.append("/*\n" + json.dumps(meta, indent=2) + "\n*/\n\n")

    header.append("// ---- Model constants ----\n")
    header.append(f"#define MODEL_VERSION             1\n")
    header.append(f"#define MODEL_D_MODEL             {cfg.d_model}\n")
    header.append(f"#define MODEL_NHEAD               {cfg.nhead}\n")
    header.append(f"#define MODEL_DEPTH               {cfg.depth}\n")
    header.append(f"#define MODEL_NUM_CLASSES         {cfg.num_classes}\n")
    header.append(f"#define MODEL_IN_CHANNELS         {cfg.in_ch}\n")
    header.append(f"#define MODEL_ATTENTION_WINDOW    {cfg.attention_window}\n")
    header.append(f"#define MODEL_USE_POWER2_SCALES   0\n\n")

    header.append("// ---- Input quantization ----\n")
    header.append(f"#define MODEL_S_INPUT             {float(s_input):.8e}f\n\n")

    header.append("// ---- Input preprocessing (raw sensor -> model int8) ----\n")
    header.append(f"#define MODEL_PREPROC_CHANNELS    {cfg.in_ch}\n")
    header.append("#define MODEL_PREPROC_LAYOUT_AX_AY_AZ_GX_GY_GZ 1\n\n")

    header.append("// ---- Integer math & softmax config ----\n")
    header.append("#define Q15_ONE                  32767\n")
    header.append("#define SOFTMAX_QBITS            15\n")
    header.append("#define SOFTMAX_EXP_XMIN         -10.0f\n")
    header.append("#define SOFTMAX_EXP_XMAX         0.0f\n\n")

    header.append("// ---- Activation LUT index & requant parameters ----\n")
    header.append(f"#define SILU_ALPHA               {silu_params[0]}\n")
    header.append(f"#define SILU_BETA                {silu_params[1]}\n")
    header.append(f"#define SILU_RSHIFT              {silu_params[2]}\n")
    header.append(f"#define GELU_ALPHA               {gelu_params[0]}\n")
    header.append(f"#define GELU_BETA                {gelu_params[1]}\n")
    header.append(f"#define GELU_RSHIFT              {gelu_params[2]}\n")
    header.append(f"#define GELU_M_OUT               {gelu_params[3]}\n")
    header.append(f"#define GELU_R_OUT               {gelu_params[4]}\n")
    header.append(f"#define AP_GELU_ALPHA            {ap_gelu_params[0]}\n")
    header.append(f"#define AP_GELU_BETA             {ap_gelu_params[1]}\n")
    header.append(f"#define AP_GELU_RSHIFT           {ap_gelu_params[2]}\n")
    header.append(f"#define AP_GELU_M_OUT            {ap_gelu_params[3]}\n")
    header.append(f"#define AP_GELU_R_OUT            {ap_gelu_params[4]}\n")
    header.append(f"#define MHSA_SOFTMAX_SCALE       {mhsa_softmax_scale:.8e}f\n")
    header.append(f"#define AP_SOFTMAX_SCALE         {ap_softmax_scale:.8e}f\n")

    # Precompute softmax mul_bins for integer-only LUT index mapping.
    # Matches softmax_row_q15f_intbins() in ker_softmax.h:
    #   bpu = (LUT_EXP_SIZE-1) / (LUT_EXP_XMAX - LUT_EXP_XMIN)
    #   mul_bins = round(scale * bpu * (1 << SM_SOFTMAX_INTBINS_RSHIFT))
    _sm_lut_size = 1024
    _sm_xmin, _sm_xmax = -10.0, 0.0
    _sm_rshift = 20  # SM_SOFTMAX_INTBINS_RSHIFT
    _sm_bpu = (_sm_lut_size - 1) / (_sm_xmax - _sm_xmin)
    _mhsa_mul = mhsa_softmax_scale * _sm_bpu * (1 << _sm_rshift)
    _ap_mul = ap_softmax_scale * _sm_bpu * (1 << _sm_rshift)
    mhsa_softmax_mul_bins = int(round(_mhsa_mul))
    ap_softmax_mul_bins = int(round(_ap_mul))
    header.append(f"#define MHSA_SOFTMAX_MUL_BINS    {mhsa_softmax_mul_bins}\n")
    header.append(f"#define AP_SOFTMAX_MUL_BINS      {ap_softmax_mul_bins}\n\n")

    header.append(
        "// Per-layer activation scales (s_act), per-tensor, float stored for tooling/debug.\n"
    )

    section_attr = '__attribute__((section(".xheep_data_flash_only")))' if flash_only_section else ""
    act_scale_attrs = " ".join([a for a in [section_attr, "__attribute__((aligned(16)))"] if a])
    header.append(
        "static const int32_t %s g_act_scales_q31[%d] = %s;\n"
        % (act_scale_attrs, len(act_scales_q31), c_array_initializer(act_scales_q31, values_per_line=10))
    )
    header.append("// Names (indices):\n")
    for i, name in enumerate(act_names):
        header.append(f"//   {i}: {name}\n")
    header.append("\n")

    def emit_blob(name: str, ctype: str, arr: np.ndarray, align: int = 16) -> None:
        arr = np.asarray(arr)
        flat = arr.reshape(-1)
        if ctype == "int8_t":
            init = c_array_initializer(flat.tolist(), values_per_line=16)
        elif ctype == "int32_t":
            init = c_array_initializer([int(x) for x in flat.tolist()], values_per_line=8)
        elif ctype == "float":
            init = c_float_array_initializer([float(x) for x in flat.tolist()], values_per_line=8)
        else:
            raise ValueError(f"unsupported ctype {ctype}")

        attrs = []
        if flash_only_section:
            attrs.append('__attribute__((section(".xheep_data_flash_only")))')
        attrs.append(f"__attribute__((aligned({align})))")
        attr_str = " ".join(attrs)
        header.append(f"static const {ctype} {attr_str} {name}[{flat.size}] = {init};\n")

    header.append("// ---- Preprocessing arrays ----\n")
    emit_blob("g_preproc_mean", "float", np.asarray(preproc_mean, dtype=np.float32))
    emit_blob("g_preproc_std", "float", np.asarray(preproc_std, dtype=np.float32))
    emit_blob("g_preproc_invstd", "float", np.asarray(preproc_invstd, dtype=np.float32))
    emit_blob("g_preproc_invstd_over_sinput", "float", np.asarray(preproc_invstd_over_sinput, dtype=np.float32))
    emit_blob("g_preproc_bias_q", "float", np.asarray(preproc_bias_q, dtype=np.float32))
    header.append("\n")

    # ---- Convs (must match transformer.h CONV* macros) ----
    for (idx, outc, inc, k, stride, pad, dil, groups, comment) in conv_defs:
        header.append(f"// ---- Conv1d {idx} : {comment} ----\n")
        header.append(f"#define CONV{idx}_OUTC   {outc}\n")
        header.append(f"#define CONV{idx}_INC    {inc}\n")
        header.append(f"#define CONV{idx}_K      {k}\n")
        header.append(f"#define CONV{idx}_STRIDE {stride}\n")
        header.append(f"#define CONV{idx}_PAD    {pad}\n")
        header.append(f"#define CONV{idx}_DIL    {dil}\n")
        header.append(f"#define CONV{idx}_GROUPS {groups}\n")
        emit_blob(f"conv{idx}_W", "int8_t", weights[f"conv{idx}_W"][1])
        emit_blob(f"conv{idx}_B", "int32_t", weights[f"conv{idx}_B"][1])
        emit_blob(f"conv{idx}_M", "int32_t", weights[f"conv{idx}_M"][1])
        emit_blob(f"conv{idx}_R", "int32_t", weights[f"conv{idx}_R"][1])
        header.append("\n")

    # ---- Linears (must match transformer.h FC* macros) ----
    for (idx, out_dim, in_dim, comment) in fc_defs:
        header.append(f"// ---- Linear {idx} : {comment} ----\n")
        header.append(f"#define FC{idx}_OUT   {out_dim}\n")
        header.append(f"#define FC{idx}_IN    {in_dim}\n")
        emit_blob(f"fc{idx}_W", "int8_t", weights[f"fc{idx}_W"][1])
        emit_blob(f"fc{idx}_B", "int32_t", weights[f"fc{idx}_B"][1])
        emit_blob(f"fc{idx}_M", "int32_t", weights[f"fc{idx}_M"][1])
        emit_blob(f"fc{idx}_R", "int32_t", weights[f"fc{idx}_R"][1])
        header.append("\n")

    header.append("\n// ---- LayerNorm params (float32) ----\n")
    for ln_name, (gamma, beta, eps) in ln_params.items():
        d = gamma.shape[0]
        header.append(f"#define {ln_name.upper()}_D    {d}\n")
        header.append(f"#define {ln_name.upper()}_EPS  {eps:.8e}f\n")
        emit_blob(f"{ln_name}_gamma", "float", gamma)
        emit_blob(f"{ln_name}_beta", "float", beta)
        header.append("\n")

    # ---- rsqrt LUT for integer-only LayerNorm ----
    if rsqrt_lut is not None:
        lut_arr, lut_log_min, lut_log_max = rsqrt_lut
        lut_n = int(lut_arr.shape[0])
        header.append("\n// ---- Integer LayerNorm: rsqrt LUT ----\n")
        header.append(f"#define LN_INT_RSQRT_LUT_SIZE    {lut_n}\n")
        header.append(f"#define LN_INT_RSQRT_LOG_MIN     {float(lut_log_min):.8e}f\n")
        header.append(f"#define LN_INT_RSQRT_LOG_MAX     {float(lut_log_max):.8e}f\n")
        header.append(f"#define LN_INT_RSQRT_LOG_RANGE   {float(lut_log_max - lut_log_min):.8e}f\n")
        header.append(f"#define LN_INT_ENABLED           1\n")
        emit_blob("ln_rsqrt_lut", "float", lut_arr)
        header.append("\n")
    else:
        header.append("\n#define LN_INT_ENABLED           0\n")

    header.append("// End of model.h\n")

    out_path.write_text("".join(header))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint from train.py")
    ap.add_argument("--calib", type=Path, required=True, help="Calibration NPZ (same format as train data)")
    ap.add_argument("--meta-json", type=Path, default=None, help="Optional meta.json from dataset creation; overrides embedded NPZ meta for preprocessing export")
    ap.add_argument("--out", type=Path, default=Path("model.h"), help="Output model.h path")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--calib-method", type=str, default="percentile", choices=["maxabs", "percentile"], help="How to reduce per-batch maxabs into a single scale")
    ap.add_argument("--calib-percentile", type=float, default=99.9, help="Percentile used when --calib-method=percentile")
    ap.add_argument("--calib-input-quant", action="store_true", help="Collect activation stats under fake-quantized input (int8->dequant)")
    ap.add_argument(
        "--flash-only-section",
        action="store_true",
        help='Emit exported const arrays with __attribute__((section(".xheep_data_flash_only")))',
    )
    ap.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        help="If checkpoint contains EMA state, export using EMA-averaged weights.",
    )
    ap.add_argument(
        "--no-use-ema",
        dest="use_ema",
        action="store_false",
        help="Disable EMA weight averaging during export, even if checkpoint contains EMA state.",
    )
    ap.set_defaults(use_ema=True)
    ap.add_argument("--s-input", type=float, default=0.0, help="Override input scale (0 => derive from calib/ckpt)")
    ap.add_argument("--no-prefer-ckpt-s-input", action="store_true", help="Do not prefer ckpt['quant']['s_input'] if present")
    ap.add_argument(
        "--int-ln",
        action="store_true",
        default=None,
        help="Force export of integer LayerNorm rsqrt LUT (overrides checkpoint config)",
    )
    ap.add_argument(
        "--no-int-ln",
        dest="int_ln",
        action="store_false",
        help="Force disable integer LayerNorm export, even if checkpoint was trained with --int-ln",
    )
    ap.add_argument("--int-ln-lut-size", type=int, default=0, help="Override rsqrt LUT size (0 => use checkpoint config or default 256)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = TinyTransformerHARConfig(**ckpt["cfg"])

    # --int-ln / --no-int-ln override checkpoint config
    if args.int_ln is not None:
        cfg.int_layernorm = bool(args.int_ln)
    if args.int_ln_lut_size > 0:
        cfg.int_ln_lut_size = int(args.int_ln_lut_size)

    model = TinyTransformerHAR(cfg)
    # ---- choose which weights to export ----
    state_dict = ckpt.get("state_dict", None) if isinstance(ckpt, dict) else None
    state_dict_raw = ckpt.get("state_dict_raw", None) if isinstance(ckpt, dict) else None
    if state_dict is None:
        raise SystemExit("checkpoint missing key 'state_dict'")

    # Load deploy weights first (default behavior).
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)

    # Optional: apply EMA state (for older checkpoints that didn't materialize EMA weights
    # into state_dict, or if user explicitly requests EMA export).
    # This context manager keeps EMA weights active for stats collection and quantization.
    weights_src = "state_dict"
    ema_ctx = nullcontext()
    if bool(args.use_ema) and isinstance(ckpt, dict) and (ckpt.get("ema") is not None):
        try:
            from torch_ema import ExponentialMovingAverage  # type: ignore

            # Instantiate EMA helper on current model params, then load shadow params.
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
            ema.load_state_dict(ckpt["ema"])
            ema_ctx = ema.average_parameters()
            weights_src = "ema"
        except Exception as e:
            print(f"[warn] --use-ema requested but could not apply EMA state: {e}")
            print("[warn] falling back to checkpoint state_dict weights")

    # If user didn't request EMA but the checkpoint contains both deploy and raw weights,
    # record a nicer source string.
    if weights_src == "state_dict" and state_dict_raw is not None:
        weights_src = "deploy(state_dict)"

    with ema_ctx:
        calib = np.load(args.calib)
        if "X" in calib:
            x = calib["X"].astype(np.float32)
        else:
            x = calib["x"].astype(np.float32)
        if x.shape[0] > args.calib_samples:
            x = x[: args.calib_samples]

        ckpt_quant = ckpt.get("quant", {}) if isinstance(ckpt, dict) else {}
        ckpt_s_input = ckpt_quant.get("s_input", None)
        if ckpt_s_input is not None:
            try:
                ckpt_s_input = float(ckpt_s_input)
            except Exception:
                ckpt_s_input = None

        # Choose s_input (override > ckpt > calib-derived)
        s_input: Optional[float] = None
        if args.s_input and float(args.s_input) > 0.0:
            s_input = float(args.s_input)
            s_input_src = "override"
        elif (not args.no_prefer_ckpt_s_input) and ckpt_s_input is not None and ckpt_s_input > 0.0:
            s_input = float(ckpt_s_input)
            s_input_src = "ckpt"
        else:
            # derive from input stats
            x_absmax_per_sample = np.max(np.abs(x.reshape(x.shape[0], -1)), axis=1) if x.shape[0] else np.asarray([0.0])
            if args.calib_method == "maxabs":
                x_stat = float(np.max(x_absmax_per_sample))
            else:
                x_stat = float(np.percentile(x_absmax_per_sample, float(args.calib_percentile)))
            s_input = _scale_from_maxabs(x_stat)
            s_input_src = "calib"

        x_for_stats = x
        if args.calib_input_quant:
            x_for_stats = _fake_quant_dequant_np(x_for_stats, float(s_input))

        act_stats: Dict[str, float] = {"__input__": float(s_input) * 127.0}
        act_stats.update(
            collect_activation_stats(
                model,
                x_for_stats,
                args.device,
                batch=64,
                method=str(args.calib_method),
                percentile=float(args.calib_percentile),
            )
        )

        scales = choose_export_scales(cfg, act_stats)
        scales = ExportScales(
            s_input=float(s_input),
            s_resid=scales.s_resid,
            s_final=scales.s_final,
            s_mlp_fc1_out=scales.s_mlp_fc1_out,
            s_mlp_gelu_out=scales.s_mlp_gelu_out,
            s_ap_fc0_out=scales.s_ap_fc0_out,
            s_ap_gelu_out=scales.s_ap_gelu_out,
            s_ap_fc1_out=scales.s_ap_fc1_out,
            s_cls_in=scales.s_cls_in,
        )

        preproc_mean, preproc_std = load_standardization_from_meta(
            calib_npz=args.calib,
            meta_json=args.meta_json,
            expected_channels=cfg.in_ch,
        )
        preproc_invstd = (1.0 / np.maximum(preproc_std, np.float32(1e-12))).astype(np.float32)
        preproc_invstd_over_sinput = (preproc_invstd / np.float32(scales.s_input)).astype(np.float32)
        preproc_bias_q = (-preproc_mean * preproc_invstd_over_sinput).astype(np.float32)

        # LUT index mapping params
        silu_alpha, silu_beta, silu_rshift = index_mapping_params(
            scales.s_resid, xmin=-8.0, xmax=8.0, L=1024, rshift=12
        )

        gelu_alpha, gelu_beta, gelu_rshift = index_mapping_params(
            scales.s_mlp_fc1_out, xmin=-8.0, xmax=8.0, L=1024, rshift=12
        )
        # GELU Q15 -> int8 in FC2 input scale (shared)
        gelu_scale = 1.0 / (scales.s_mlp_gelu_out * (1 << 15))
        ms_gelu = mult_shift_from_real(gelu_scale)

        ap_gelu_alpha, ap_gelu_beta, ap_gelu_rshift = index_mapping_params(
            scales.s_ap_fc0_out, xmin=-8.0, xmax=8.0, L=1024, rshift=12
        )
        ap_gelu_scale = 1.0 / (scales.s_ap_gelu_out * (1 << 15))
        ms_ap_gelu = mult_shift_from_real(ap_gelu_scale)

        # Softmax scales
        dh = cfg.d_model // cfg.nhead
        mhsa_softmax_scale = (scales.s_resid * scales.s_resid) * (1.0 / math.sqrt(float(dh)))
        ap_softmax_scale = scales.s_ap_fc1_out

        # Quantize weights and compute b/M/R blobs
        # 32-bit requant decomposition: M_small = round(M / 2^M_SHIFT), R2 = R - R1 - M_SHIFT
        # Must match REQUANT32_M_SHIFT and fc_requant_R1() in ker_linear.h.
        REQUANT32_M_SHIFT = 15

        def _requant32_R1(Cin_eff: int) -> int:
            """Pre-shift for 32-bit-only requant (matches fc_requant_R1 in C)."""
            max_acc = Cin_eff * 127 * 127
            acc_bits = max_acc.bit_length() + 1  # +1 for sign
            r1 = acc_bits + 16 - 31
            return max(0, r1)

        def _decompose_requant_32bit(M: np.ndarray, R: np.ndarray, Cin_eff: int) -> Tuple[np.ndarray, np.ndarray]:
            """Decompose (M, R) into (M_small, R2) for 32-bit requant."""
            R1 = _requant32_R1(Cin_eff)
            half = 1 << (REQUANT32_M_SHIFT - 1)
            M_small = (M.astype(np.int64) + half) >> REQUANT32_M_SHIFT
            R2 = R - R1 - REQUANT32_M_SHIFT
            assert np.all(R2 > 0), f"R2 must be > 0 for all channels (min={R2.min()}, R1={R1})"
            assert np.all(np.abs(M_small) < (1 << 16)), f"M_small overflow (max={np.abs(M_small).max()})"
            return M_small.astype(np.int32), R2.astype(np.int32)

        w_blobs: Dict[str, Tuple[str, np.ndarray]] = {}

        def add_conv_pw(name_prefix: str, weight: torch.Tensor, bias: torch.Tensor, s_in: float, s_out: float) -> None:
            # weight: [Cout,Cin,K]
            wq, sw = quantize_per_out_channel_int8(_np(weight))
            bq = quantize_bias_int32(_np(bias), s_in=s_in, s_w=sw)
            M, R = requant_params_per_channel(s_in=s_in, s_w=sw, s_out=s_out)
            Cin_K = weight.shape[1] * weight.shape[2]  # Cin * K
            M_s, R2 = _decompose_requant_32bit(M, R, Cin_K)
            w_blobs[f"{name_prefix}_W"] = ("int8_t", wq.astype(np.int8).reshape(-1))
            w_blobs[f"{name_prefix}_B"] = ("int32_t", bq.astype(np.int32).reshape(-1))
            w_blobs[f"{name_prefix}_M"] = ("int32_t", M_s.reshape(-1))
            w_blobs[f"{name_prefix}_R"] = ("int32_t", R2.reshape(-1))

        def add_conv_dw(name_prefix: str, weight: torch.Tensor, bias: torch.Tensor, s_in: float, s_out: float) -> None:
            # weight: [C,1,K] -> [C,K]
            w = _np(weight).squeeze(1)
            wq, sw = quantize_per_out_channel_int8(w)
            bq = quantize_bias_int32(_np(bias), s_in=s_in, s_w=sw)
            M, R = requant_params_per_channel(s_in=s_in, s_w=sw, s_out=s_out)
            K = weight.shape[2]
            M_s, R2 = _decompose_requant_32bit(M, R, K)
            w_blobs[f"{name_prefix}_W"] = ("int8_t", wq.astype(np.int8).reshape(-1))
            w_blobs[f"{name_prefix}_B"] = ("int32_t", bq.astype(np.int32).reshape(-1))
            w_blobs[f"{name_prefix}_M"] = ("int32_t", M_s.reshape(-1))
            w_blobs[f"{name_prefix}_R"] = ("int32_t", R2.reshape(-1))

        def add_fc(name_prefix: str, weight: torch.Tensor, bias: torch.Tensor, s_in: float, s_out: float) -> None:
            # weight: [Cout,Cin]
            wq, sw = quantize_per_out_channel_int8(_np(weight))
            bq = quantize_bias_int32(_np(bias), s_in=s_in, s_w=sw)
            M, R = requant_params_per_channel(s_in=s_in, s_w=sw, s_out=s_out)
            Cin = weight.shape[1]
            M_s, R2 = _decompose_requant_32bit(M, R, Cin)
            w_blobs[f"{name_prefix}_W"] = ("int8_t", wq.astype(np.int8).reshape(-1))
            w_blobs[f"{name_prefix}_B"] = ("int32_t", bq.astype(np.int32).reshape(-1))
            w_blobs[f"{name_prefix}_M"] = ("int32_t", M_s.reshape(-1))
            w_blobs[f"{name_prefix}_R"] = ("int32_t", R2.reshape(-1))

        # Conv stem (conv0)
        add_conv_pw("conv0", model.stem.weight, model.stem.bias, scales.s_input, scales.s_resid)

        # Posmix depthwise (conv1), requant to input scale for residual add
        add_conv_dw("conv1", model.posmix.weight, model.posmix.bias, scales.s_resid, scales.s_resid)

        conv_meta: List[Tuple[int, int, int, int, int, int, int, int, str]] = [
            (0, cfg.d_model, cfg.in_ch, cfg.stem_kernel, 1, cfg.stem_kernel // 2, 1, 1, "stem.conv"),
            (1, cfg.d_model, cfg.d_model, cfg.posmix_kernel, 1, cfg.posmix_kernel // 2, 1, cfg.d_model, "posmix.dwconv"),
        ]

        # FCs: index order must match transformer.h helpers
        fc_defs: List[Tuple[str, torch.Tensor, torch.Tensor, float, float]] = []
        fc_meta: List[Tuple[int, int, int, str]] = []

        # Blocks - split QKV into separate Q, K, V for token-major layout optimization
        for i, blk in enumerate(model.blocks):
            # Split QKV from [3*C, C] into three [C, C] layers
            qkv_weight = _np(blk.qkv.weight)
            qkv_bias = _np(blk.qkv.bias)

            q_weight = torch.from_numpy(qkv_weight[0:cfg.d_model, :]).to(args.device)
            k_weight = torch.from_numpy(qkv_weight[cfg.d_model:2*cfg.d_model, :]).to(args.device)
            v_weight = torch.from_numpy(qkv_weight[2*cfg.d_model:3*cfg.d_model, :]).to(args.device)

            q_bias = torch.from_numpy(qkv_bias[0:cfg.d_model]).to(args.device)
            k_bias = torch.from_numpy(qkv_bias[cfg.d_model:2*cfg.d_model]).to(args.device)
            v_bias = torch.from_numpy(qkv_bias[2*cfg.d_model:3*cfg.d_model]).to(args.device)

            # New layout: 6 FCs per block [Q, K, V, proj, fc1, fc2]
            fc_defs.append((f"fc{6*i+0}", q_weight, q_bias, scales.s_resid, scales.s_resid))
            fc_defs.append((f"fc{6*i+1}", k_weight, k_bias, scales.s_resid, scales.s_resid))
            fc_defs.append((f"fc{6*i+2}", v_weight, v_bias, scales.s_resid, scales.s_resid))
            fc_defs.append((f"fc{6*i+3}", blk.proj.weight, blk.proj.bias, scales.s_resid, scales.s_resid))
            # mlp.fc1 outputs shared hidden scale
            fc_defs.append((f"fc{6*i+4}", blk.mlp.fc1.weight, blk.mlp.fc1.bias, scales.s_resid, scales.s_mlp_fc1_out))
            # mlp.fc2 input is GELU output scale, output returns to residual scale
            fc_defs.append((f"fc{6*i+5}", blk.mlp.fc2.weight, blk.mlp.fc2.bias, scales.s_mlp_gelu_out, scales.s_resid))

            fc_meta.append((6 * i + 0, cfg.d_model, cfg.d_model, f"blk{i}.mha.q"))
            fc_meta.append((6 * i + 1, cfg.d_model, cfg.d_model, f"blk{i}.mha.k"))
            fc_meta.append((6 * i + 2, cfg.d_model, cfg.d_model, f"blk{i}.mha.v"))
            fc_meta.append((6 * i + 3, cfg.d_model, cfg.d_model, f"blk{i}.mha.proj"))
            fc_meta.append((6 * i + 4, cfg.ffn_mult * cfg.d_model, cfg.d_model, f"blk{i}.mlp.fc1"))
            fc_meta.append((6 * i + 5, cfg.d_model, cfg.ffn_mult * cfg.d_model, f"blk{i}.mlp.fc2"))

        # AttnPool FC0 and FC1
        base = 6 * cfg.depth
        fc_defs.append((f"fc{base+0}", model.attnpool.fc0.weight, model.attnpool.fc0.bias, scales.s_final, scales.s_ap_fc0_out))
        fc_defs.append((f"fc{base+1}", model.attnpool.fc1.weight, model.attnpool.fc1.bias, scales.s_ap_gelu_out, scales.s_ap_fc1_out))

        fc_meta.append((base + 0, cfg.attnpool_hidden, cfg.d_model, "attnpool.fc0"))
        fc_meta.append((base + 1, 1, cfg.attnpool_hidden, "attnpool.fc1"))

        # Classifier FC (weights used for int32 logits; still export M/R for completeness)
        fc_defs.append((f"fc{base+2}", model.classifier.weight, model.classifier.bias, scales.s_cls_in, scales.s_cls_in))
        fc_meta.append((base + 2, cfg.num_classes, cfg.d_model, "classifier.fc"))

        for name, w, b, s_in, s_out in fc_defs:
            add_fc(name, w, b, s_in, s_out)

        # LN params blobs
        ln_params: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

        for i, blk in enumerate(model.blocks):
            ln_params[f"ln_blk{i}_mha_ln"] = (
                _np(blk.ln_mha.weight).astype(np.float32),
                _np(blk.ln_mha.bias).astype(np.float32),
                float(blk.ln_mha.eps),
            )
            ln_params[f"ln_blk{i}_mlp_ln"] = (
                _np(blk.ln_mlp.weight).astype(np.float32),
                _np(blk.ln_mlp.bias).astype(np.float32),
                float(blk.ln_mlp.eps),
            )

        ln_params["ln_final_norm"] = (
            _np(model.final_ln.weight).astype(np.float32),
            _np(model.final_ln.bias).astype(np.float32),
            float(model.final_ln.eps),
        )
        ln_params["ln_attn_pool_ln"] = (
            _np(model.attnpool.ln.weight).astype(np.float32),
            _np(model.attnpool.ln.bias).astype(np.float32),
            float(model.attnpool.ln.eps),
        )
        ln_params["ln_classifier_ln"] = (
            _np(model.classifier_ln.weight).astype(np.float32),
            _np(model.classifier_ln.bias).astype(np.float32),
            float(model.classifier_ln.eps),
        )

        # Activation scales list expected by transformer.h (interleaved per block):
        act_names = ["stem_out", "posmix_out"]
        for i in range(cfg.depth):
            act_names.append(f"blk{i}_mha_out")
            act_names.append(f"blk{i}_mlp_out")
        act_names += ["final_norm_out", "attnpool_out", "classifier_in", "classifier_out"]

        # We keep residual-chain scales equal for correctness of residual adds.
        act_scales: List[float] = []
        act_scales.append(scales.s_resid)  # stem_out
        act_scales.append(scales.s_resid)  # posmix_out
        for _ in range(cfg.depth):
            act_scales.append(scales.s_resid)  # blk mha out
            act_scales.append(scales.s_resid)  # blk mlp out
        act_scales.append(scales.s_final)
        act_scales.append(scales.s_final)  # attnpool_out
        act_scales.append(scales.s_cls_in)
        act_scales.append(scales.s_cls_in)  # classifier_out (not used by this runtime)

        meta_notes = (
            "int8 symmetric quant (zp=0); int32 accum; per-channel weights; per-tensor activations; "
            "integer requant (M,r); residual paths keep s_out==s_in. "
            f"calib_method={args.calib_method} calib_percentile={float(args.calib_percentile)} "
            f"calib_input_quant={bool(args.calib_input_quant)} s_input_src={s_input_src}. "
            f"weights_src={weights_src}."
            + (f" int_layernorm=True lut_size={cfg.int_ln_lut_size}." if cfg.int_layernorm else "")
        )

        # Build rsqrt LUT if model uses integer LayerNorm
        rsqrt_lut_export: Optional[Tuple[np.ndarray, float, float]] = None
        if cfg.int_layernorm:
            lut_t, log_min, log_max = build_rsqrt_lut(
                n=cfg.int_ln_lut_size,
                var_min=1e-5,
                var_max=127.0 ** 2,
            )
            rsqrt_lut_export = (lut_t.numpy().astype(np.float32), float(log_min), float(log_max))
            print(f"int-ln: exporting rsqrt LUT ({cfg.int_ln_lut_size} entries)")

        write_model_h(
            args.out,
            cfg=cfg,
            weights=w_blobs,
            conv_defs=conv_meta,
            fc_defs=fc_meta,
            ln_params=ln_params,
            act_scales=act_scales,
            act_names=act_names,
            silu_params=(silu_alpha, silu_beta, silu_rshift),
            gelu_params=(gelu_alpha, gelu_beta, gelu_rshift, int(ms_gelu.m), int(ms_gelu.r)),
            ap_gelu_params=(ap_gelu_alpha, ap_gelu_beta, ap_gelu_rshift, int(ms_ap_gelu.m), int(ms_ap_gelu.r)),
            mhsa_softmax_scale=float(mhsa_softmax_scale),
            ap_softmax_scale=float(ap_softmax_scale),
            s_input=float(scales.s_input),
            preproc_mean=preproc_mean,
            preproc_std=preproc_std,
            preproc_invstd=preproc_invstd,
            preproc_invstd_over_sinput=preproc_invstd_over_sinput,
            preproc_bias_q=preproc_bias_q,
            meta_notes=meta_notes,
            flash_only_section=bool(args.flash_only_section),
            rsqrt_lut=rsqrt_lut_export,
        )

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()