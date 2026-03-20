from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import re

import numpy as np


INT8_QMIN = -128
INT8_QMAX = 127


def _npz_get_first(data: "np.lib.npyio.NpzFile", keys: list[str]) -> np.ndarray:
    for k in keys:
        if k in data:
            return np.asarray(data[k])
    raise KeyError(f"npz missing keys {keys!r}; has {list(data.keys())!r}")


def _scale_from_maxabs(maxabs: float) -> float:
    if not np.isfinite(maxabs) or maxabs <= 0:
        return 1.0 / 127.0
    return float(maxabs) / 127.0


def _xorshift32_step(state: int) -> int:
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def gen_prng_int8(n: int, *, seed: int = 0x12345678) -> np.ndarray:
    """Match fill_rand_i8() in main.c (xorshift32, mod 255, [-127,127])."""
    # Inline xorshift32 + collect into a Python list (avoids per-element
    # numpy scalar creation and numpy indexing overhead).
    _M = 0xFFFFFFFF
    vals = [0] * n
    x = int(seed) & _M
    for i in range(n):
        x ^= (x << 13) & _M
        x ^= (x >> 17) & _M
        x ^= (x << 5)  & _M
        vals[i] = (x % 255) - 127
    return np.array(vals, dtype=np.int8)


def quantize_symmetric_int8(x: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    q = np.rint(np.asarray(x, dtype=np.float64) / float(scale))
    q = np.clip(q, INT8_QMIN, INT8_QMAX)
    return q.astype(np.int8)


def format_c_int8_array(name: str, values: np.ndarray, *, values_per_line: int = 16) -> str:
    vals = np.asarray(values, dtype=np.int8).reshape(-1).tolist()   # C-fast conversion
    n = len(vals)
    lines: list[str] = []
    for i in range(0, n, values_per_line):
        lines.append("    " + ", ".join(map(str, vals[i : i + values_per_line])))
    body = "{\n" + ",\n".join(lines) + "\n}"
    return f"static const int8_t {name}[{n}] = {body};\n"


def format_c_int32_array(name: str, values: list[int] | np.ndarray, *, values_per_line: int = 16) -> str:
    vals = np.asarray(values, dtype=np.int32).reshape(-1).tolist()   # C-fast conversion
    n = len(vals)
    lines: list[str] = []
    for i in range(0, n, values_per_line):
        lines.append("    " + ", ".join(map(str, vals[i : i + values_per_line])))
    body = "{\n" + ",\n".join(lines) + "\n}"
    return f"static const int32_t {name}[{n}] = {body};\n"


@dataclass(frozen=True)
class TestVector:
    x_ct_flat: np.ndarray  # int8, flattened [Cin,T] (T fastest)
    cin: int
    t: int
    label: int
    vector_id: int


@dataclass
class _NpzCache:
    """Pre-loaded NPZ arrays so we don't re-read the file per vector."""
    x: np.ndarray   # float32 [N, ...]
    y: np.ndarray   # int64   [N]


def _load_npz(npz_path: Path) -> _NpzCache:
    data = np.load(npz_path, allow_pickle=False)
    x = _npz_get_first(data, ["X", "x"]).astype(np.float32)
    y = _npz_get_first(data, ["y", "Y"]).astype(np.int64)
    if x.ndim != 3:
        raise ValueError(f"expected x to be rank-3 [N,T,C] (or [N,C,T]); got shape {x.shape}")
    return _NpzCache(x=x, y=y)


def load_npz_vector(
    npz_path: Path, index: int, *, cache: _NpzCache | None = None,
) -> tuple[np.ndarray, int]:
    if cache is None:
        cache = _load_npz(npz_path)

    x, y = cache.x, cache.y

    if index < 0 or index >= x.shape[0]:
        raise IndexError(f"index {index} out of range for N={x.shape[0]}")

    xs = x[index]
    ys = int(y[index])

    # Heuristic: if second dim is small (Cin) and third dim is large (T), assume [C,T].
    if xs.shape[0] <= 16 and xs.shape[1] >= 32:
        xs = xs.T  # to [T,C]

    # Now expect [T,Cin]
    if xs.shape[0] < xs.shape[1]:
        # typical T=128, Cin=6; if flipped still, fix it
        pass

    return xs, ys


def load_npz_num_samples(npz_path: Path, *, cache: _NpzCache | None = None) -> int:
    if cache is None:
        cache = _load_npz(npz_path)
    return int(cache.x.shape[0])


def compute_s_input_from_calib(calib_npz: Path, *, max_samples: int = 512) -> float:
    data = np.load(calib_npz, allow_pickle=False)
    x = _npz_get_first(data, ["X", "x"]).astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected calib x to be rank-3; got {x.shape}")
    if x.shape[0] > max_samples:
        x = x[:max_samples]

    m = float(np.max(np.abs(x)))
    return _scale_from_maxabs(m)


def compute_s_input_from_model_h(model_h: Path) -> float:
    """Extract s_input from exported model.h.

    Tries JSON meta (preferred), then falls back to MODEL_S_INPUT macro.
    """
    txt = model_h.read_text(encoding="utf-8", errors="replace")

    # 1) JSON meta block at top: /* { ... } */
    m = re.search(r"/\*\s*(\{.*?\})\s*\*/", txt, flags=re.S)
    if m:
        try:
            meta = json.loads(m.group(1))
            if isinstance(meta, dict) and "s_input" in meta:
                v = float(meta["s_input"])
                if np.isfinite(v) and v > 0:
                    return v
        except Exception:
            pass

    # 2) Macro fallback
    m2 = re.search(r"#define\s+MODEL_S_INPUT\s+([0-9eE+\-\.]+)f?", txt)
    if m2:
        v = float(m2.group(1))
        if np.isfinite(v) and v > 0:
            return v

    raise ValueError("could not extract s_input from model.h (no meta.s_input and no MODEL_S_INPUT)")


def build_test_vector(
    *,
    mode: Literal["prng", "npz"],
    cin: int,
    t: int,
    vector_id: int,
    label: int,
    seed: int,
    npz_path: Path | None,
    npz_index: int,
    s_input: float | None,
    calib_npz: Path | None,
    calib_samples: int,
    npz_cache: _NpzCache | None = None,
) -> TestVector:
    if mode == "prng":
        x_flat = gen_prng_int8(cin * t, seed=seed)
        return TestVector(x_ct_flat=x_flat, cin=cin, t=t, label=label, vector_id=vector_id)

    if npz_path is None:
        raise ValueError("--npz is required for mode=npz")

    xs_tc, y = load_npz_vector(npz_path, npz_index, cache=npz_cache)
    if xs_tc.ndim != 2:
        raise ValueError(f"expected sample x to be rank-2 [T,Cin]; got {xs_tc.shape}")

    if xs_tc.shape[0] != t or xs_tc.shape[1] != cin:
        raise ValueError(f"sample shape mismatch: expected (T={t},Cin={cin}), got {xs_tc.shape}")

    if s_input is None:
        if calib_npz is None:
            raise ValueError("provide --s-input or --calib to derive it")
        s_input = compute_s_input_from_calib(calib_npz, max_samples=calib_samples)

    x_q_tc = quantize_symmetric_int8(xs_tc, s_input)
    x_q_ct = x_q_tc.T  # [Cin,T]
    x_flat = x_q_ct.reshape(-1)
    return TestVector(x_ct_flat=x_flat, cin=cin, t=t, label=y, vector_id=vector_id)


def write_header(out_path: Path, tv: TestVector, *, notes: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = []
    header.append("// Auto-generated by gen_test_vector.py — DO NOT EDIT\n")
    header.append("#pragma once\n")
    header.append("#include <stdint.h>\n\n")
    header.append("/*\n" + json.dumps(notes, indent=2) + "\n*/\n\n")

    header.append(f"#define TT_TEST_VECTOR_CIN {tv.cin}\n")
    header.append(f"#define TT_TEST_VECTOR_T {tv.t}\n")
    header.append(f"#define TT_TEST_VECTOR_LABEL {int(tv.label)}\n")
    header.append(f"#define TT_TEST_VECTOR_ID {int(tv.vector_id)}\n\n")

    header.append(format_c_int8_array("g_test_x_in", tv.x_ct_flat))

    out_path.write_text("".join(header))


def write_header_multi(out_path: Path, tvs: list[TestVector], *, notes: dict[str, Any]) -> None:
    if not tvs:
        raise ValueError("no test vectors to write")

    cin = tvs[0].cin
    t = tvs[0].t
    for tv in tvs:
        if tv.cin != cin or tv.t != t:
            raise ValueError("all test vectors must share the same Cin and T")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    header: list[str] = []
    header.append("// Auto-generated by gen_test_vector.py — DO NOT EDIT\n")
    header.append("#pragma once\n")
    header.append("#include <stdint.h>\n\n")
    header.append("/*\n" + json.dumps(notes, indent=2) + "\n*/\n\n")

    header.append(f"#define TT_TEST_VECTOR_CIN {cin}\n")
    header.append(f"#define TT_TEST_VECTOR_T {t}\n")
    header.append(f"#define TT_TEST_VECTOR_COUNT {len(tvs)}\n")
    header.append("#define TT_TEST_VECTOR_HAS_TABLES 1\n")

    # Backward-compatible defines: expose the first vector's ID/label
    header.append(f"#define TT_TEST_VECTOR_LABEL {int(tvs[0].label)}\n")
    header.append(f"#define TT_TEST_VECTOR_ID {int(tvs[0].vector_id)}\n\n")

    ids = [int(tv.vector_id) for tv in tvs]
    labels = [int(tv.label) for tv in tvs]

    header.append(format_c_int32_array("g_test_vector_ids", ids))
    header.append("\n")
    header.append(format_c_int32_array("g_test_vector_labels", labels))
    header.append("\n")

    # Flat buffer: concatenate vectors. In C you index as base = g_test_x_in + i*(Cin*T).
    x_all = np.concatenate([tv.x_ct_flat.reshape(-1) for tv in tvs], axis=0).astype(np.int8)
    header.append(format_c_int8_array("g_test_x_in", x_all))

    out_path.write_text("".join(header))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate sw/applications/transformer_streaming/test_vector.h")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "test_vector.h")
    ap.add_argument("--mode", choices=["prng", "npz"], default="prng")
    ap.add_argument("--cin", type=int, default=6, help="Input channels (MODEL_IN_CHANNELS)")
    ap.add_argument("--t", type=int, default=64, help="Sequence length (TT_T)")
    ap.add_argument("--id", type=int, default=0, help="Arbitrary test vector ID")
    ap.add_argument("--count", type=int, default=1, help="Generate N vectors starting at --index (npz) or --seed (prng)")
    ap.add_argument("--label", type=int, default=-1, help="Label for prng mode (for npz mode it is read from file)")

    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x12345678, help="PRNG seed (hex ok), prng mode only")

    ap.add_argument("--npz", type=Path, default=None, help="NPZ with X/y or x/y (mode=npz)")
    ap.add_argument("--index", type=int, default=0, help="Sample index in NPZ (mode=npz)")
    ap.add_argument("--shuffle", action="store_true", help="For mode=npz and --count>1, pick random indices (instead of consecutive)")
    ap.add_argument("--shuffle-seed", type=lambda s: int(s, 0), default=0xC0FFEE, help="RNG seed for --shuffle (hex ok)")
    ap.add_argument("--s-input", type=float, default=None, help="Input scale used by exporter (mode=npz)")
    ap.add_argument("--calib", type=Path, default=None, help="Calibration NPZ to derive s_input as maxabs/127 (mode=npz)")
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--model-h", type=Path, default=None, help="Exported model.h to derive s_input (preferred over --calib)")

    args = ap.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be >= 1")

    # Resolve s_input once (npz mode) for reproducibility
    resolved_s_input = (
        args.s_input
        if args.s_input is not None
        else (compute_s_input_from_model_h(args.model_h) if args.model_h is not None else None)
    )

    tvs: list[TestVector] = []
    chosen_indices: list[int] = []

    # Pre-load NPZ once if we need it (avoids re-reading per vector)
    npz_cache: _NpzCache | None = None
    if args.mode == "npz" and args.npz is not None:
        npz_cache = _load_npz(args.npz)

    if args.mode == "npz" and args.shuffle and int(args.count) > 1:
        if args.npz is None:
            raise SystemExit("--npz is required")
        n = load_npz_num_samples(args.npz, cache=npz_cache)
        rng = np.random.default_rng(int(args.shuffle_seed) & 0xFFFFFFFF)
        # sample without replacement if possible, else with replacement
        replace = int(args.count) > n
        chosen_indices = [int(i) for i in rng.choice(n, size=int(args.count), replace=replace).tolist()]
    else:
        chosen_indices = [int(args.index) + i for i in range(int(args.count))]

    for i in range(int(args.count)):
        tvs.append(
            build_test_vector(
                mode=args.mode,
                cin=args.cin,
                t=args.t,
                vector_id=int(args.id) + i,
                label=args.label,
                seed=(int(args.seed) + i) & 0xFFFFFFFF,
                npz_path=args.npz,
                npz_index=int(chosen_indices[i]),
                s_input=resolved_s_input,
                calib_npz=args.calib,
                calib_samples=args.calib_samples,
                npz_cache=npz_cache,
            )
        )

    notes: dict[str, Any] = {
        "mode": args.mode,
        "cin": args.cin,
        "t": args.t,
        "id": args.id,
        "count": int(args.count),
    }
    if args.mode == "prng":
        notes["seed"] = hex(args.seed)
        notes["label"] = int(args.label)
    else:
        notes["npz"] = str(args.npz) if args.npz is not None else None
        notes["index"] = int(args.index)
        notes["indices"] = chosen_indices
        notes["shuffle"] = bool(args.shuffle)
        notes["shuffle_seed"] = hex(int(args.shuffle_seed) & 0xFFFFFFFF)
        notes["s_input"] = float(args.s_input) if args.s_input is not None else None
        notes["calib"] = str(args.calib) if args.calib is not None else None
        notes["model_h"] = str(args.model_h) if args.model_h is not None else None

    if len(tvs) == 1:
        write_header(args.out, tvs[0], notes=notes)
    else:
        write_header_multi(args.out, tvs, notes=notes)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
