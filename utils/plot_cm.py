#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LABELS = [
    "walking",
    "running",
    "upstairs",
    "downstairs",
    "sitting",
    "standing",
    "lying",
    "other",
]


def project_root_from_here() -> Path:
    # .../polheepo/sw/applications/storm/python/<this_file>
    here = Path(__file__).resolve()
    return here.parents[4]


def build_main_cpu(exe_path: Path, *, force: bool = False) -> None:
    root = project_root_from_here()
    storm_dir = root / "sw" / "applications" / "storm"
    src = storm_dir / "main_cpu.c"

    if not src.exists():
        raise FileNotFoundError(f"Cannot find {src}")

    needs_build = force or (not exe_path.exists()) or (src.stat().st_mtime > exe_path.stat().st_mtime)
    if not needs_build:
        return

    exe_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gcc",
        "-std=c11",
        "-O2",
        "-I",
        str(storm_dir),
        "-I",
        str(storm_dir / "include"),
        str(src),
        "-lm",
        "-o",
        str(exe_path),
    ]

    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write("Failed to build main_cpu\n")
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def run_main_cpu(exe_path: Path) -> str:
    proc = subprocess.run([str(exe_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write("main_cpu failed\n")
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def parse_confusion_matrix(output: str) -> np.ndarray:
    lines = output.splitlines()

    i0 = None
    i1 = None
    for i, line in enumerate(lines):
        if line.startswith("CONFUSION_MATRIX_BEGIN"):
            i0 = i
            break
    if i0 is None:
        raise ValueError("CONFUSION_MATRIX_BEGIN not found in output")

    for i in range(i0 + 1, len(lines)):
        if lines[i].startswith("CONFUSION_MATRIX_END"):
            i1 = i
            break
    if i1 is None:
        raise ValueError("CONFUSION_MATRIX_END not found in output")

    m = re.search(r"nclass=(\d+)", lines[i0])
    if not m:
        raise ValueError("Could not parse nclass from CONFUSION_MATRIX_BEGIN line")
    nclass = int(m.group(1))

    # Read the next nclass non-empty lines as CSV rows.
    rows = []
    for line in lines[i0 + 1 : i1]:
        if not line.strip():
            continue
        rows.append([int(x) for x in line.split(",")])

    cm = np.asarray(rows, dtype=np.int64)
    if cm.shape != (nclass, nclass):
        raise ValueError(f"Unexpected confusion matrix shape {cm.shape}, expected {(nclass, nclass)}")
    return cm


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
    out = np.zeros_like(cm, dtype=np.float64)
    np.divide(cm, row_sum, out=out, where=(row_sum != 0.0))
    return out


def plot_cm_relative(cm_rel: np.ndarray, *, labels: list[str], out_path: Path | None, show: bool) -> None:
    nclass = cm_rel.shape[0]
    labels = labels[:nclass]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_rel, cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative frequency")

    ax.set_xlabel("pred")
    ax.set_ylabel("true")

    ax.set_xticks(np.arange(nclass))
    ax.set_yticks(np.arange(nclass))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add numbers inside cells (relative values).
    for r in range(nclass):
        for c in range(nclass):
            v = float(cm_rel[r, c])
            ax.text(
                c,
                r,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=("white" if v >= 0.5 else "black"),
            )

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run main_cpu and plot relative confusion matrix (viridis).")
    parser.add_argument("--exe", type=str, default=None, help="Path to main_cpu executable (optional).")
    parser.add_argument("--no-build", action="store_true", help="Do not try to build main_cpu.")
    parser.add_argument("--force-build", action="store_true", help="Always rebuild main_cpu.")
    parser.add_argument("--save", type=str, default="confusion_matrix.png", help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show interactive window (in addition to saving).")
    args = parser.parse_args()

    root = project_root_from_here()
    default_exe = root / "sw" / "applications" / "storm" / "build_host" / "main_cpu"
    exe_path = Path(args.exe).expanduser().resolve() if args.exe else default_exe

    if not args.no_build:
        build_main_cpu(exe_path, force=args.force_build)

    out = run_main_cpu(exe_path)
    cm = parse_confusion_matrix(out)
    cm_rel = normalize_rows(cm)

    # Ensure label count matches class count if MODEL_NUM_CLASSES changes.
    labels = LABELS
    if len(labels) < cm_rel.shape[0]:
        labels = labels + [f"class_{i}" for i in range(len(labels), cm_rel.shape[0])]

    out_path = Path(args.save).expanduser().resolve() if args.save else None
    plot_cm_relative(cm_rel, labels=labels, out_path=out_path, show=args.show)

    # Also print a small summary + path for CI/logging.
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
