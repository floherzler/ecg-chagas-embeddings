#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_binary(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().any():
        return (out > 0.5).astype(int)
    ss = s.astype("string[python]").str.strip().str.lower()
    return ss.isin(["1", "true", "t", "yes", "y"]).astype(int)


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def main() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = argparse.ArgumentParser(
        description="Focused CODE15 correlation report (better than a full pairplot)."
    )
    parser.add_argument(
        "--exams_csv",
        type=Path,
        default=Path(
            "/home/flo178/projects/master-thesis/datasets/physionet2025/code15/exams.csv"
        ),
        help="CODE15 exams.csv containing RBBB, death, timey, etc.",
    )
    parser.add_argument(
        "--master_meta",
        type=Path,
        default=Path(
            "/home/flo178/projects/master-thesis/datasets/physionet2025/processedMaster/metadata.csv"
        ),
        help="processedMaster/metadata.csv (used to get the Chagas label).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis/code15_correlations"),
        help="Output directory (default: ./analysis/code15_correlations).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
    )
    args = parser.parse_args()

    exams = pd.read_csv(args.exams_csv, low_memory=False)
    meta = pd.read_csv(args.master_meta, low_memory=False, usecols=["exam_id", "source", "chagas"])
    meta = meta[meta["source"] == "CODE-15%"].copy()

    exams["exam_id"] = exams["exam_id"].astype(str)
    meta["exam_id"] = meta["exam_id"].astype(str)
    df = exams.merge(meta[["exam_id", "chagas"]], on="exam_id", how="inner")
    df["chagas"] = _to_binary(df["chagas"])

    # Derived
    df["delta_age"] = pd.to_numeric(df["nn_predicted_age"], errors="coerce") - pd.to_numeric(
        df["age"], errors="coerce"
    )
    df["RBBB"] = _to_binary(df["RBBB"]) if "RBBB" in df.columns else 0
    df["death"] = _to_binary(df["death"]) if "death" in df.columns else 0

    # Set up style
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.38)

    # 1) RBBB vs Chagas: 2x2 heatmap of proportions (with counts annotated)
    ax0 = fig.add_subplot(gs[0, 0])
    sub0 = df[["RBBB", "chagas"]].dropna()
    cm = pd.crosstab(sub0["chagas"], sub0["RBBB"])  # rows: chagas, cols: rbbb
    cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    cm_prop = cm / max(1, int(cm.to_numpy().sum()))
    sns.heatmap(
        cm_prop,
        ax=ax0,
        cmap="Blues",
        vmin=0.0,
        vmax=float(cm_prop.to_numpy().max()) if cm_prop.to_numpy().size else 1.0,
        cbar=True,
        annot=cm.apply(lambda col: col.map(lambda v: f"{int(v)}")),
        fmt="",
        linewidths=0.5,
        linecolor="#eeeeee",
    )
    ax0.set_xlabel("RBBB")
    ax0.set_ylabel("Chagas")
    ax0.set_xticklabels(["0", "1"])
    ax0.set_yticklabels(["0", "1"], rotation=0)
    ax0.set_title("RBBB × Chagas (counts; color=global proportion)")

    # 2) Age vs predicted age: density + regression
    ax1 = fig.add_subplot(gs[0, 1])
    sub1 = df[["age", "nn_predicted_age"]].copy()
    sub1["age"] = pd.to_numeric(sub1["age"], errors="coerce")
    sub1["nn_predicted_age"] = pd.to_numeric(sub1["nn_predicted_age"], errors="coerce")
    sub1 = sub1.dropna()
    if len(sub1) > 0:
        hb = ax1.hexbin(
            sub1["age"].to_numpy(),
            sub1["nn_predicted_age"].to_numpy(),
            gridsize=60,
            bins="log",
            cmap="Greys",
            mincnt=1,
        )
        fig.colorbar(hb, ax=ax1, shrink=0.85, label="count (log)")
        sns.regplot(
            data=sub1.sample(n=min(8000, len(sub1)), random_state=0),
            x="age",
            y="nn_predicted_age",
            scatter=False,
            ax=ax1,
            color="#E45756",
            line_kws={"linewidth": 2.5},
        )
    ax1.set_title("Age vs NN-predicted age")
    ax1.set_xlabel("age")
    ax1.set_ylabel("nn_predicted_age")

    # 3) Δage vs Chagas: violin + box
    ax2 = fig.add_subplot(gs[1, 0])
    sub2 = df[["delta_age", "chagas"]].copy()
    sub2["delta_age"] = pd.to_numeric(sub2["delta_age"], errors="coerce")
    sub2 = sub2.dropna()
    if len(sub2) > 0:
        plot2 = sub2.copy()
        plot2["chagas"] = plot2["chagas"].astype(int)
        sns.violinplot(
            data=sub2,
            x="chagas",
            y="delta_age",
            inner=None,
            cut=0,
            linewidth=0.8,
            hue="chagas",
            palette={0: "#4C78A8", 1: "#F58518"},
            legend=False,
            ax=ax2,
        )
        sns.boxplot(
            data=plot2,
            x="chagas",
            y="delta_age",
            width=0.25,
            showcaps=True,
            boxprops={"facecolor": "white", "zorder": 3},
            whiskerprops={"zorder": 3},
            medianprops={"color": "black", "linewidth": 2.0},
            showfliers=False,
            ax=ax2,
        )
    ax2.set_title("Δage = nn_predicted_age − age vs Chagas")
    ax2.set_xlabel("chagas")
    ax2.set_ylabel("delta_age (years)")

    # 4) timey vs (Chagas, death): distributions (time-to-event style)
    ax3 = fig.add_subplot(gs[1, 1])
    sub3 = df[["timey", "chagas", "death"]].copy()
    sub3["timey"] = pd.to_numeric(sub3["timey"], errors="coerce")
    sub3["death"] = pd.to_numeric(sub3["death"], errors="coerce")
    sub3 = sub3.dropna(subset=["timey", "chagas", "death"])
    if len(sub3) > 0:
        # Use a log-ish x-axis if timey is highly skewed; keep linear but clip extreme outliers.
        hi = float(np.nanpercentile(sub3["timey"].to_numpy(), 99.0))
        sub3 = sub3[sub3["timey"] <= hi]
        sub3["cond"] = (
            "chagas="
            + sub3["chagas"].astype(int).astype(str)
            + ", death="
            + sub3["death"].astype(int).astype(str)
        )
        order = ["chagas=0, death=0", "chagas=0, death=1", "chagas=1, death=0", "chagas=1, death=1"]
        palette = {
            "chagas=0, death=0": "#4C78A8",
            "chagas=0, death=1": "#9ecae9",
            "chagas=1, death=0": "#F58518",
            "chagas=1, death=1": "#d62728",
        }
        sns.kdeplot(
            data=sub3,
            x="timey",
            hue="cond",
            hue_order=[c for c in order if c in set(sub3["cond"].tolist())],
            common_norm=False,
            fill=True,
            alpha=0.25,
            linewidth=2.0,
            palette=palette,
            ax=ax3,
        )
    ax3.set_title("timey distribution by (Chagas, death) (first-exam rows only)")
    ax3.set_xlabel("timey (years)")
    ax3.set_ylabel("density")

    fig.suptitle(f"CODE15 correlation report (N={len(df)})", y=0.995)

    # No footnote text: the timey panel encodes the distinction directly via the 4 conditions.

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "code15_correlation_report.png"
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
