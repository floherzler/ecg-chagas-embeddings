#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = argparse.ArgumentParser(
        description="Generate a seaborn pair plot (scatterplot matrix) from CODE15 exams.csv."
    )
    parser.add_argument(
        "--exams_csv",
        type=Path,
        default=Path(
            "/home/flo178/projects/master-thesis/datasets/physionet2025/code15/exams.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis/code15_correlations"),
        help="Output directory (default: ./analysis/code15_correlations).",
    )
    parser.add_argument(
        "--drop",
        type=str,
        default="exam_id,trace_file,patient_id",
        help="Comma-separated columns to drop.",
    )
    parser.add_argument(
        "--vars",
        type=str,
        default="",
        help="Optional comma-separated columns to include (after drop). Default: all remaining.",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=5000,
        help="Row sample size for feasibility (0 = use all rows; can be very slow).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--kind",
        choices=["scatter", "reg"],
        default="scatter",
        help="Off-diagonal plot kind. reg adds a linear fit (slower).",
    )
    parser.add_argument(
        "--diag_kind",
        choices=["hist", "kde"],
        default="hist",
        help="Diagonal plot kind.",
    )
    parser.add_argument(
        "--corner",
        action="store_true",
        help="Plot only the lower triangle (recommended).",
    )
    parser.add_argument("--height", type=float, default=1.7)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    df = pd.read_csv(args.exams_csv, low_memory=False)
    drop_cols = [c.strip() for c in str(args.drop).split(",") if c.strip()]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if args.vars.strip():
        keep = [c.strip() for c in str(args.vars).split(",") if c.strip()]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"Requested --vars missing from CSV: {missing}")
        df = df[keep].copy()

    # Convert booleans to 0/1 for plotting and numeric correlations.
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

    # Coerce numeric-like columns.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Drop columns that are still non-numeric and not boolean-coded.
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in drop_cols
    ]
    df = df[numeric_cols].copy()

    # Remove all-NaN / constant columns (pairplot can choke on them).
    nunique = df.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    df = df[keep_cols].copy()

    if args.sample_n and int(args.sample_n) > 0 and len(df) > int(args.sample_n):
        df = df.sample(n=int(args.sample_n), random_state=int(args.seed))

    # Keep pairplot stable (avoid massive figure sizes).
    sns.set_theme(style="ticks", context="talk")
    g = sns.pairplot(
        df,
        kind=str(args.kind),
        diag_kind=str(args.diag_kind),
        corner=bool(args.corner),
        plot_kws={"s": 10, "alpha": 0.25, "linewidth": 0.0},
        diag_kws={"bins": 30},
        height=float(args.height),
    )
    g.fig.suptitle(f"CODE15 pair plot (N={len(df)})", y=1.02)
    plt.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "code15_pairplot.png"
    g.fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(g.fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
