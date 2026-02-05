#!/usr/bin/env python
"""Export thesis-ready tables and overview plots from analysis CSV files."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
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


def _fmt_group_table(df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
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
            row[m] = f"{med:.3f} [{q1:.3f}, {q3:.3f}]"
        row["n_models"] = int(len(gdf))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols)
    return out


def _write_latex_table(
    df: pd.DataFrame, path: Path, caption: str, label: str, *, escape: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escape LaTeX-special chars (notably `_` in run/preprocessing names).
    body = df.to_latex(index=False, escape=escape, na_rep="NA", column_format=None)
    wrapped = (
        "\\begin{table}[t]\n"
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
        sns.set_theme(style="whitegrid")
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

    # Overview: GPU vs TPR@5
    m = emb.copy()
    if "space" in m.columns:
        m = m[m["space"] == "enc"].copy()
    merged = scores.merge(
        m[["run_id", "GPU_0", "GPU_1", "CAC_0", "CAC_1", "SAA_0", "SAA_1"]],
        on="run_id",
        how="inner",
    )
    if {"GPU_1", "tpr_top0.05", "preprocessing", "track"}.issubset(merged.columns):
        plt.figure(figsize=(6.6, 4.4))
        if sns is not None:
            sns.scatterplot(
                data=merged,
                x="GPU_1",
                y="tpr_top0.05",
                hue="preprocessing",
                style="track",
                s=70,
            )
        else:
            for (track, pre), gdf in merged.groupby(["track", "preprocessing"]):
                plt.scatter(gdf["GPU_1"], gdf["tpr_top0.05"], label=f"{track}-{pre}", s=35, alpha=0.85)
            plt.legend(fontsize=8, ncol=2)
        plt.xlabel("GPU (class 1)")
        plt.ylabel("TPR@5%")
        plt.title("Embedding uniformity vs clinical utility")
        plt.tight_layout()
        plt.savefig(fig_dir / "overview_gpu_vs_tpr5.png", dpi=220)
        plt.close()

    # Spearman heatmap (main overview)
    if spearman_path.exists():
        s = pd.read_csv(spearman_path, index_col=0)
        s = s.reindex(index=_sort_runs(pd.DataFrame({"run_id": s.index}))["run_id"], columns=_sort_runs(pd.DataFrame({"run_id": s.columns}))["run_id"])
        plt.figure(figsize=(8.8, 7.4))
        if sns is not None:
            sns.heatmap(s, cmap="magma", vmin=-1, vmax=1)
        else:
            plt.imshow(s.to_numpy(), cmap="magma", vmin=-1, vmax=1, aspect="auto")
            plt.xticks(range(len(s.columns)), s.columns, rotation=90, fontsize=7)
            plt.yticks(range(len(s.index)), s.index, fontsize=7)
            plt.colorbar()
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
        if sns is not None:
            sns.heatmap(iou, cmap="magma", vmin=0, vmax=1)
        else:
            plt.imshow(iou.to_numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
            plt.xticks(range(len(iou.columns)), iou.columns, rotation=90, fontsize=7)
            plt.yticks(range(len(iou.index)), iou.index, fontsize=7)
            plt.colorbar()
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
    t_cls_split = _fmt_group_table(scores, cls_split, ["track", "preprocessing"])

    emb_enc = emb[emb["space"].astype(str).eq("enc")] if "space" in emb.columns else emb.copy()
    t_emb = _fmt_group_table(emb_enc, emb_metrics, ["track", "preprocessing"])

    # PCA correlation summary by group: absolute median
    pca_cols = [c for c in pca.columns if c.startswith("corr_")]
    pca2 = scores[["run_id", "track", "preprocessing"]].merge(pca[["run_id"] + pca_cols], on="run_id", how="left")
    for c in pca_cols:
        pca2[c] = pd.to_numeric(pca2[c], errors="coerce").abs()
    t_pca = _fmt_group_table(pca2, pca_cols, ["track", "preprocessing"])

    _write_latex_table(
        t_cls,
        table_dir / "classification_by_group.tex",
        "Clinical metrics by track and preprocessing (median [Q1, Q3]).",
        "tab:cls_by_group",
    )
    _write_latex_table(
        t_cls_split,
        table_dir / "classification_labelsplit_by_group.tex",
        "Label-split clinical metrics by track and preprocessing (median [Q1, Q3]).",
        "tab:cls_split_by_group",
    )
    _write_latex_table(
        t_emb,
        table_dir / "embedding_by_group.tex",
        "Embedding diagnostics by track and preprocessing (median [Q1, Q3]).",
        "tab:emb_by_group",
    )
    _write_latex_table(
        t_pca,
        table_dir / "pca_corr_by_group.tex",
        "Absolute PCA-correlation summaries by track and preprocessing (median [Q1, Q3]).",
        "tab:pca_corr_by_group",
    )

    # also keep per-run compact tables for appendix
    keep_scores = ["run_id", "track", "preprocessing"] + [c for c in cls_metrics + cls_split if c in scores.columns]
    # Compact per-run table for main text: short model label + core metrics only.
    compact_rows: list[dict[str, object]] = []
    for _, r in scores.iterrows():
        track, preproc, loss, rot = _parse_run_parts(r["run_id"])
        model = rf"\shortstack{{{track}\\{preproc}\\{loss}{rot}}}"
        compact_rows.append(
            {
                "model": model,
                "TPR@5\\%": r.get("tpr_top0.05"),
                "pAUC@5\\%": r.get("pauc_fpr0.05"),
                "AP": r.get("ap"),
                "TPR@10\\%": r.get("tpr_top0.10"),
                "AUROC": r.get("auroc"),
            }
        )
    t_cls_per_run = pd.DataFrame(compact_rows)

    _write_latex_table(
        t_cls_per_run,
        table_dir / "classification_per_run.tex",
        "Per-run clinical metrics (compact model labels: track / preprocessing / objective).",
        "tab:cls_per_run",
        escape=False,
    )
    emb_keep = ["run_id", "track", "preprocessing"] + [c for c in emb_metrics if c in emb_enc.columns]
    _write_latex_table(
        emb_enc[emb_keep],
        table_dir / "embedding_per_run.tex",
        "Per-run embedding metrics (encoder space).",
        "tab:emb_per_run",
    )

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
