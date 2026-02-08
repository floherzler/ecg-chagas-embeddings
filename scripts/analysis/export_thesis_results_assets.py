#!/usr/bin/env python
"""Export thesis-ready tables and overview plots from analysis CSV files."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:
    sns = None


def _parse_run_id(run_id: str) -> tuple[int, int, int, int]:
    rid = str(run_id)
    m_track = re.match(r"t(\d+)-", rid)
    track = int(m_track.group(1)) if m_track else 99
    m_exp = re.search(r"-exp(\d+)-", rid)
    exp = int(m_exp.group(1)) if m_exp else 999
    if "-bp-sc-norm-" in rid:
        pre = 2
    elif "-bp-sc-" in rid:
        pre = 1
    elif "-bp-" in rid:
        pre = 0
    else:
        pre = 9
    rot = 1 if rid.endswith("-rot10") else 0
    return (track, pre, exp, rot)


def _parse_run_parts(run_id: str) -> tuple[str, str, str, str]:
    rid = str(run_id).strip()
    track_m = re.match(r"^(t\d+)-", rid)
    track = track_m.group(1) if track_m else "t?"
    if "-bp-sc-norm-" in rid:
        preproc = "bp-sc-norm"
    elif "-bp-sc-" in rid:
        preproc = "bp-sc"
    elif "-bp-" in rid:
        preproc = "bp"
    else:
        preproc = "other"
    # loss/objective is the token right after preprocessing
    # run_id pattern: tX-expYY-<preproc>-<loss>[-rot10]
    parts = rid.split("-")
    loss = parts[-2] if rid.endswith("-rot10") and len(parts) >= 2 else parts[-1]
    if loss == "rot10" and len(parts) >= 2:
        loss = parts[-2]
    rot = "+rot10" if rid.endswith("-rot10") else ""
    return track, preproc, loss, rot


def _fmt_float(x: object, decimals: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == "" or x.lower() == "na":
                return "NA"
        v = float(x)
        if np.isnan(v):
            return "NA"
        return f"{v:.{decimals}f}"
    except Exception:
        return "NA"


def _bold_best(values: list[float], higher_is_better: bool) -> set[int]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return set()
    target = np.nanmax(arr) if higher_is_better else np.nanmin(arr)
    return {
        i
        for i, v in enumerate(values)
        if isinstance(v, (float, int)) and np.isfinite(v) and v == target
    }


def _render_grouped_per_run_table(
    df: pd.DataFrame,
    metrics: list[str],
    higher_is_better: dict[str, bool],
    caption: str,
    label: str,
) -> str:
    cols = ["model"] + metrics
    # Global best per metric across the whole table (for \overall{})
    global_best: dict[str, set[int]] = {}
    for m in metrics:
        vals = []
        for v in df[m].tolist():
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        global_best[m] = _bold_best(vals, higher_is_better.get(m, True))
    # Build rows with parsed model label.
    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        track, preproc, loss, rot = _parse_run_parts(r["run_id"])
        # Use a single-line compact model label with dashes.
        model = rf"\shortstack{{{track}-{preproc}-{loss}{rot}}}"
        row: dict[str, object] = {"model": model, "track": track, "preproc": preproc}
        for m in metrics:
            row[m] = r.get(m)
        rows.append(row)
    out = pd.DataFrame(rows)
    # Group ordering
    track_order = ["t1", "t3"]
    preproc_order = ["bp", "bp-sc", "bp-sc-norm"]

    # Header
    header = [
        "% --- begin: local helpers for this table file (safe to keep here) ---",
        r"\providecommand{\best}[1]{\textbf{#1}}",
        r"\providecommand{\overall}[1]{\underline{\textbf{#1}}}",
        r"% --- end: local helpers ---",
        "",
        r"\begin{table}[H]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{l{'r' * len(metrics)}}}",
        r"        \toprule",
        "        " + " & ".join(cols) + r" \\",
        r"        \midrule",
        "",
    ]
    lines = header
    # Blocks
    for t_i, track in enumerate(track_order):
        for p_i, preproc in enumerate(preproc_order):
            block = out[(out["track"] == track) & (out["preproc"] == preproc)].copy()
            if block.empty:
                continue
            lines.append(
                rf"        \multicolumn{{{len(cols)}}}{{l}}{{\textit{{Track {track} \;|\; {preproc}}}}} \\"
            )
            lines.append(r"        \midrule")
            # determine best per metric in block
            best_idx: dict[str, set[int]] = {}
            for m in metrics:
                vals = []
                for v in block[m].tolist():
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(np.nan)
                best_idx[m] = _bold_best(vals, higher_is_better.get(m, True))
            # emit rows
            for i, r in block.reset_index(drop=True).iterrows():
                row_cells = [r["model"]]
                for m in metrics:
                    s = _fmt_float(r[m], decimals=4)
                    if s != "NA":
                        # Mark global best with \overall{}, otherwise block-best with \best{}
                        if (block.index.to_list()[i] in global_best[m]):
                            s = rf"\overall{{{s}}}"
                        elif i in best_idx[m]:
                            s = rf"\best{{{s}}}"
                    row_cells.append(s)
                lines.append("        " + " & ".join(row_cells) + r" \\")
            lines.append(r"        \addlinespace[4pt]\midrule")
            lines.append("")
        if t_i == 0:
            lines.append(r"        \addlinespace[6pt]\midrule\midrule")
            lines.append("")

    # Footer
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = out["run_id"].map(_parse_run_id)
    out["_k0"] = keys.map(lambda x: x[0])
    out["_k1"] = keys.map(lambda x: x[1])
    out["_k2"] = keys.map(lambda x: x[2])
    out["_k3"] = keys.map(lambda x: x[3])
    out = out.sort_values(["_k0", "_k1", "_k2", "_k3", "run_id"]).drop(
        columns=["_k0", "_k1", "_k2", "_k3"]
    )
    return out


def _fmt_group_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str],
    *,
    decimals: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for grp, gdf in df.groupby(group_cols, dropna=False):
        if not isinstance(grp, tuple):
            grp = (grp,)
        row: dict[str, object] = {c: v for c, v in zip(group_cols, grp)}
        for m in metric_cols:
            if m not in gdf.columns:
                continue
            v = pd.to_numeric(gdf[m], errors="coerce").dropna()
            if v.empty:
                row[m] = np.nan
                continue
            q1, med, q3 = float(v.quantile(0.25)), float(v.median()), float(v.quantile(0.75))
            row[m] = f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"
        row["n_models"] = int(len(gdf))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols)
    return out


def _write_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    *,
    escape: bool = True,
    table_pos: str = "H",
    size_cmd: str | None = None,
    resize_to_linewidth: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escape LaTeX-special chars (notably `_` in run/preprocessing names).
    body = df.to_latex(index=False, escape=escape, na_rep="NA", column_format=None)
    if resize_to_linewidth:
        body = "\\resizebox{\\linewidth}{!}{%\n" + body + "}\n"
    if size_cmd:
        body = size_cmd + "\n" + body
    wrapped = (
        f"\\begin{{table}}[{table_pos}]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )
    path.write_text(wrapped)


def _export_plots(scores: pd.DataFrame, emb: pd.DataFrame, spearman_path: Path, iou_path: Path, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="white", rc={"axes.grid": False})
    else:
        plt.style.use("ggplot")

    # Overview: TPR@5 by preprocessing and track
    if {"tpr_top0.05", "preprocessing", "track"}.issubset(scores.columns):
        plt.figure(figsize=(7.2, 4.0))
        if sns is not None:
            sns.boxplot(data=scores, x="preprocessing", y="tpr_top0.05", hue="track")
            sns.stripplot(
                data=scores,
                x="preprocessing",
                y="tpr_top0.05",
                hue="track",
                dodge=True,
                alpha=0.5,
                size=3,
                legend=False,
                color="black",
            )
        else:
            g = (
                scores.groupby(["preprocessing", "track"], as_index=False)["tpr_top0.05"]
                .median()
                .pivot(index="preprocessing", columns="track", values="tpr_top0.05")
            )
            g.plot(kind="bar", ax=plt.gca())
            plt.legend(title="track")
        plt.ylabel("TPR@5%")
        plt.xlabel("Preprocessing")
        plt.title("Clinical ranking performance by preprocessing")
        plt.tight_layout()
        plt.savefig(fig_dir / "overview_tpr5_by_preproc.png", dpi=220)
        plt.close()

    # Overview: label-strength split (verified vs CODE15) by preprocessing
    if {"preprocessing", "track", "tpr_top0.05_verified", "tpr_top0.05_code15"}.issubset(scores.columns):
        s2 = scores.copy()
        long = s2.melt(
            id_vars=["run_id", "track", "preprocessing"],
            value_vars=["tpr_top0.05_verified", "tpr_top0.05_code15"],
            var_name="split",
            value_name="tpr5",
        )
        long["split"] = long["split"].map(
            {
                "tpr_top0.05_verified": "Verified labels",
                "tpr_top0.05_code15": "CODE-15 labels",
            }
        )
        plt.figure(figsize=(8.2, 4.4))
        if sns is not None:
            sns.boxplot(data=long, x="preprocessing", y="tpr5", hue="split")
            sns.stripplot(
                data=long,
                x="preprocessing",
                y="tpr5",
                hue="split",
                dodge=True,
                alpha=0.45,
                size=3,
                legend=False,
                color="black",
            )
        else:
            g = (
                long.groupby(["preprocessing", "split"], as_index=False)["tpr5"]
                .median()
                .pivot(index="preprocessing", columns="split", values="tpr5")
            )
            g.plot(kind="bar", ax=plt.gca())
            plt.legend(title="split")
        plt.ylabel("TPR@5%")
        plt.xlabel("Preprocessing")
        plt.title("Label-strength split performance (verified vs CODE-15)")
        plt.tight_layout()
        plt.savefig(fig_dir / "overview_tpr5_labelsplit.png", dpi=220)
        plt.close()

    # Overview: embedding diagnostics vs TPR@5
    m = emb.copy()
    if "space" in m.columns:
        m = m[m["space"] == "enc"].copy()
    merged = scores.merge(
        m[["run_id", "GPU_0", "GPU_1", "CAC_0", "CAC_1", "SAA_0", "SAA_1"]],
        on="run_id",
        how="inner",
    )
    # Extract experiment id from run_id pattern tX-expYY-...
    merged["exp_id"] = (
        merged["run_id"]
        .astype(str)
        .str.extract(r"-exp(\d+)-", expand=False)
    )

    def _annotate_exp_ids(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        for _, r in df.iterrows():
            if pd.isna(r.get("exp_id")):
                continue
            ax.text(
                float(r[x_col]),
                float(r[y_col]),
                str(int(r["exp_id"])),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                path_effects=[pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()],
                zorder=5,
            )
    if {"CAC_1", "tpr_top0.05", "preprocessing", "track"}.issubset(merged.columns):
        plt.figure(figsize=(6.6, 4.4))
        if sns is not None:
            ax = sns.scatterplot(
                data=merged,
                x="CAC_1",
                y="tpr_top0.05",
                hue="preprocessing",
                style="track",
                s=180,
            )
            _annotate_exp_ids(ax, merged, "CAC_1", "tpr_top0.05")
        else:
            ax = plt.gca()
            for (track, pre), gdf in merged.groupby(["track", "preprocessing"]):
                plt.scatter(gdf["CAC_1"], gdf["tpr_top0.05"], label=f"{track}-{pre}", s=120, alpha=0.85)
            _annotate_exp_ids(ax, merged, "CAC_1", "tpr_top0.05")
            plt.legend(fontsize=8, ncol=2)
        plt.xlabel("CAC (class 1)")
        plt.ylabel("TPR@5%")
        plt.title("CAC$_1$ vs TPR@5%")
        plt.tight_layout()
        plt.savefig(fig_dir / "overview_cac1_vs_tpr5.png", dpi=220)
        plt.close()

    if {"CAC_0", "CAC_1", "preprocessing", "track"}.issubset(merged.columns):
        plt.figure(figsize=(6.6, 4.4))
        if sns is not None:
            ax = sns.scatterplot(
                data=merged,
                x="CAC_0",
                y="CAC_1",
                hue="preprocessing",
                style="track",
                s=180,
            )
            _annotate_exp_ids(ax, merged, "CAC_0", "CAC_1")
        else:
            ax = plt.gca()
            for (track, pre), gdf in merged.groupby(["track", "preprocessing"]):
                plt.scatter(gdf["CAC_0"], gdf["CAC_1"], label=f"{track}-{pre}", s=120, alpha=0.85)
            _annotate_exp_ids(ax, merged, "CAC_0", "CAC_1")
            plt.legend(fontsize=8, ncol=2)
        plt.xlabel("CAC (class 0)")
        plt.ylabel("CAC (class 1)")
        plt.title("CAC$_0$ vs CAC$_1$")
        plt.tight_layout()
        plt.savefig(fig_dir / "appendix_cac0_vs_cac1.png", dpi=220)
        plt.close()

    if {"GPU_1", "tpr_top0.05", "preprocessing", "track"}.issubset(merged.columns):
        plt.figure(figsize=(6.6, 4.4))
        if sns is not None:
            ax = sns.scatterplot(
                data=merged,
                x="GPU_1",
                y="tpr_top0.05",
                hue="preprocessing",
                style="track",
                s=180,
            )
            _annotate_exp_ids(ax, merged, "GPU_1", "tpr_top0.05")
        else:
            ax = plt.gca()
            for (track, pre), gdf in merged.groupby(["track", "preprocessing"]):
                plt.scatter(gdf["GPU_1"], gdf["tpr_top0.05"], label=f"{track}-{pre}", s=120, alpha=0.85)
            _annotate_exp_ids(ax, merged, "GPU_1", "tpr_top0.05")
            plt.legend(fontsize=8, ncol=2)
        plt.xlabel("GPU (class 1)")
        plt.ylabel("TPR@5%")
        plt.title("GPU$_1$ vs TPR@5%")
        plt.tight_layout()
        plt.savefig(fig_dir / "overview_gpu_vs_tpr5.png", dpi=220)
        plt.close()

    # Spearman heatmap (main overview)
    if spearman_path.exists():
        s = pd.read_csv(spearman_path, index_col=0)
        s = s.reindex(index=_sort_runs(pd.DataFrame({"run_id": s.index}))["run_id"], columns=_sort_runs(pd.DataFrame({"run_id": s.columns}))["run_id"])
        plt.figure(figsize=(8.8, 7.4))
        ax = plt.gca()
        im = ax.imshow(
            s.to_numpy(),
            cmap="magma",
            vmin=-1,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_xticks(range(len(s.columns)))
        ax.set_yticks(range(len(s.index)))
        ax.set_xticklabels(s.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(s.index, fontsize=7)
        ax.grid(False)
        plt.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        plt.title("Model ranking agreement (Spearman $\\rho$)")
        plt.xlabel("Model")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(fig_dir / "overview_spearman_heatmap.png", dpi=220)
        plt.close()

    # IoU heatmap (appendix support)
    if iou_path.exists():
        iou = pd.read_csv(iou_path, index_col=0)
        iou = iou.reindex(index=_sort_runs(pd.DataFrame({"run_id": iou.index}))["run_id"], columns=_sort_runs(pd.DataFrame({"run_id": iou.columns}))["run_id"])
        plt.figure(figsize=(8.8, 7.4))
        ax = plt.gca()
        im = ax.imshow(
            iou.to_numpy(),
            cmap="magma",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_xticks(range(len(iou.columns)))
        ax.set_yticks(range(len(iou.index)))
        ax.set_xticklabels(iou.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(iou.index, fontsize=7)
        ax.grid(False)
        plt.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        plt.title("Top-5% set overlap (IoU)")
        plt.xlabel("Model")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(fig_dir / "support_iou_heatmap.png", dpi=220)
        plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "analysis/embeddings_probe"
    assets_dir = out_dir / "thesis_assets"
    table_dir = assets_dir / "tables"
    fig_dir = assets_dir / "figures"
    thesis_table_dir = root / "thesis" / "tables"

    scores = _sort_runs(pd.read_csv(out_dir / "test_scores.csv"))
    emb = _sort_runs(pd.read_csv(out_dir / "embedding_metrics.csv"))
    pca = _sort_runs(pd.read_csv(out_dir / "pca_correlations.csv"))

    # --- Tables ---
    cls_metrics = ["tpr_top0.05", "pauc_fpr0.05", "ap", "tpr_top0.10", "auroc"]
    cls_split = [
        "tpr_top0.05_verified",
        "pauc_fpr0.05_verified",
        "ap_verified",
        "tpr_top0.05_code15",
        "pauc_fpr0.05_code15",
        "ap_code15",
    ]
    emb_metrics = ["GPU_0", "GPU_1", "CAC_0", "CAC_1", "SAA_0", "SAA_1"]

    t_cls = _fmt_group_table(scores, cls_metrics, ["track", "preprocessing"])
    t_cls_split = _fmt_group_table(scores, cls_split, ["track", "preprocessing"], decimals=2)
    t_cls_split_display = t_cls_split.rename(
        columns={
            "tpr_top0.05_verified": "TPR@5% (Ver)",
            "pauc_fpr0.05_verified": "pAUC@5% (Ver)",
            "ap_verified": "AP (Ver)",
            "tpr_top0.05_code15": "TPR@5% (C15)",
            "pauc_fpr0.05_code15": "pAUC@5% (C15)",
            "ap_code15": "AP (C15)",
        }
    )
    t_cls_split_display["group"] = (
        t_cls_split_display["track"].astype(str)
        + "/"
        + t_cls_split_display["preprocessing"].astype(str).str.replace("_", "-", regex=False)
        + " (n="
        + t_cls_split_display["n_models"].astype(int).astype(str)
        + ")"
    )
    t_cls_split_display = t_cls_split_display[
        [
            "group",
            "TPR@5% (Ver)",
            "pAUC@5% (Ver)",
            "AP (Ver)",
            "TPR@5% (C15)",
            "pAUC@5% (C15)",
            "AP (C15)",
        ]
    ]

    emb_enc = emb[emb["space"].astype(str).eq("enc")] if "space" in emb.columns else emb.copy()
    t_emb = _fmt_group_table(emb_enc, emb_metrics, ["track", "preprocessing"])

    # PCA correlation summary by group: absolute median
    pca_cols = [c for c in pca.columns if c.startswith("corr_")]
    pca2 = scores[["run_id", "track", "preprocessing"]].merge(pca[["run_id"] + pca_cols], on="run_id", how="left")
    for c in pca_cols:
        pca2[c] = pd.to_numeric(pca2[c], errors="coerce").abs()
    t_pca = _fmt_group_table(pca2, pca_cols, ["track", "preprocessing"])
    # Compact display table: keep only medians (2 decimals) and concise headers.
    def _median_from_summary(cell: object) -> float:
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return np.nan
        s = str(cell).strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        # Expected form: "0.585 [0.559, 0.600]"
        try:
            return float(s.split()[0])
        except Exception:
            try:
                return float(s)
            except Exception:
                return np.nan

    pca_short = t_pca.copy()
    for c in pca_cols:
        pca_short[c] = pca_short[c].map(_median_from_summary)
    pca_short["Group (n)"] = (
        pca_short["track"].astype(str)
        + "-"
        + pca_short["preprocessing"].astype(str).str.replace("_", "-", regex=False)
        + " ("
        + pca_short["n_models"].astype(int).astype(str)
        + ")"
    )
    pca_short = pca_short.rename(
        columns={
            "corr_chagas_pc1": r"\shortstack{Chg\\PC1}",
            "corr_chagas_pc2": r"\shortstack{Chg\\PC2}",
            "corr_age_pc1": r"\shortstack{Age\\PC1}",
            "corr_age_pc2": r"\shortstack{Age\\PC2}",
            "corr_rbbb_code15_pc1": r"\shortstack{RBBB\\PC1}",
            "corr_rbbb_code15_pc2": r"\shortstack{RBBB\\PC2}",
        }
    )
    pca_short = pca_short[
        [
            "Group (n)",
            r"\shortstack{Chg\\PC1}",
            r"\shortstack{Chg\\PC2}",
            r"\shortstack{Age\\PC1}",
            r"\shortstack{Age\\PC2}",
            r"\shortstack{RBBB\\PC1}",
            r"\shortstack{RBBB\\PC2}",
        ]
    ]
    for c in pca_short.columns[1:]:
        pca_short[c] = pca_short[c].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.2f}")

    _write_latex_table(
        t_cls,
        table_dir / "classification_by_group.tex",
        "Clinical metrics by track and preprocessing (median [Q1, Q3]).",
        "tab:cls_by_group",
    )
    _write_latex_table(
        t_cls_split_display,
        table_dir / "classification_labelsplit_by_group.tex",
        "Label-split clinical metrics by track and preprocessing (median [Q1, Q3]).",
        "tab:cls_split_by_group",
        escape=True,
        size_cmd="\\scriptsize",
        resize_to_linewidth=True,
    )
    _write_latex_table(
        t_emb,
        table_dir / "embedding_by_group.tex",
        "Embedding diagnostics by track and preprocessing (median [Q1, Q3]).",
        "tab:emb_by_group",
    )
    _write_latex_table(
        pca_short,
        table_dir / "pca_corr_by_group.tex",
        "PCA-correlation medians by track/preprocessing.",
        "tab:pca_corr_by_group",
        escape=False,
        size_cmd="\\small",
        resize_to_linewidth=True,
    )

    # also keep per-run compact tables for appendix
    keep_scores = ["run_id", "track", "preprocessing"] + [c for c in cls_metrics + cls_split if c in scores.columns]
    # Per-run clinical table with group blocks and bolded bests.
    cls_df = scores[["run_id", "track", "preprocessing"] + cls_metrics].copy()
    cls_metrics_renamed = {
        "tpr_top0.05": "TPR@5\\%",
        "pauc_fpr0.05": "pAUC@5\\%",
        "ap": "AP",
        "tpr_top0.10": "TPR@10\\%",
        "auroc": "AUROC",
    }
    cls_df = cls_df.rename(columns=cls_metrics_renamed)
    cls_table = _render_grouped_per_run_table(
        cls_df,
        metrics=list(cls_metrics_renamed.values()),
        higher_is_better={m: True for m in cls_metrics_renamed.values()},
        caption="Per-run clinical metrics (compact model labels: track / preprocessing / objective). Bold marks the best within each (track $\\times$ preprocessing) block.",
        label="tab:cls_per_run",
    )
    (table_dir / "classification_per_run.tex").write_text(cls_table)

    # Per-run TPR split table (CODE-15 / overall / verified) with group blocks and bolded bests.
    tpr_split_cols = [
        "tpr_top0.05_code15",
        "tpr_top0.05",
        "tpr_top0.05_verified",
    ]
    tpr_split_df = scores[["run_id", "track", "preprocessing"] + tpr_split_cols].copy().rename(
        columns={
            "tpr_top0.05_code15": "TPR@5\\% CODE-15",
            "tpr_top0.05": "TPR@5\\% Overall",
            "tpr_top0.05_verified": "TPR@5\\% Verified",
        }
    )
    tpr_split_metrics = [
        "TPR@5\\% CODE-15",
        "TPR@5\\% Overall",
        "TPR@5\\% Verified",
    ]
    tpr_split_table = _render_grouped_per_run_table(
        tpr_split_df,
        metrics=tpr_split_metrics,
        higher_is_better={m: True for m in tpr_split_metrics},
        caption="Per-run TPR@5\\% across label partitions (CODE-15, overall, verified) using compact model labels. Bold marks the best within each (track $\\times$ preprocessing) block.",
        label="tab:cls_tpr_split_per_run",
    )
    (table_dir / "classification_tpr_split_per_run.tex").write_text(tpr_split_table)
    # Per-run embedding table with group blocks and bolded bests.
    emb_keep = ["run_id", "track", "preprocessing"] + [c for c in emb_metrics if c in emb_enc.columns]
    emb_df = emb_enc[emb_keep].rename(
        columns={
            "GPU_0": "GPU$_0$~$\\downarrow$",
            "GPU_1": "GPU$_1$~$\\downarrow$",
            "CAC_0": "CAC$_0$~$\\uparrow$",
            "CAC_1": "CAC$_1$~$\\uparrow$",
            "SAA_0": "SAA$_0$~$\\uparrow$",
            "SAA_1": "SAA$_1$~$\\uparrow$",
        }
    )
    emb_metrics_named = [
        "GPU$_0$~$\\downarrow$",
        "GPU$_1$~$\\downarrow$",
        "CAC$_0$~$\\uparrow$",
        "CAC$_1$~$\\uparrow$",
        "SAA$_0$~$\\uparrow$",
        "SAA$_1$~$\\uparrow$",
    ]
    emb_table = _render_grouped_per_run_table(
        emb_df,
        metrics=emb_metrics_named,
        higher_is_better={
            "GPU$_0$": False,
            "GPU$_1$": False,
            "CAC$_0$": True,
            "CAC$_1$": True,
            "SAA$_0$": True,
            "SAA$_1$": True,
        },
        caption="Per-run embedding metrics (encoder space). Bold marks the best within each (track $\\times$ preprocessing) block.",
        label="tab:emb_per_run",
    )
    (table_dir / "embedding_per_run.tex").write_text(emb_table)

    # Mirror generated LaTeX tables into thesis-local directory for stable \\input paths.
    thesis_table_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(table_dir.glob("*.tex")):
        (thesis_table_dir / src.name).write_text(src.read_text())

    # --- Plots ---
    _export_plots(
        scores=scores,
        emb=emb,
        spearman_path=out_dir / "ranking_agreement/test/spearman_rho.csv",
        iou_path=out_dir / "ranking_agreement/test/top5_iou.csv",
        fig_dir=fig_dir,
    )

    print(f"Wrote tables to {table_dir}")
    print(f"Mirrored tables to {thesis_table_dir}")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
