#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _mark_probe_membership(rra: pd.DataFrame, probe: pd.DataFrame) -> pd.Series:
    # Prefer composite key matching because row_idx can be regenerated and
    # exam_id-only matching could collide across datasets in future artifacts.
    if (
        "dataset_source" in rra.columns
        and "exam_id" in rra.columns
        and "dataset_source" in probe.columns
        and "exam_id" in probe.columns
    ):
        r_key = rra["dataset_source"].astype(str) + "||" + rra["exam_id"].astype(str)
        p_key = set(
            (
                probe["dataset_source"].astype(str)
                + "||"
                + probe["exam_id"].astype(str)
            ).tolist()
        )
        return r_key.isin(p_key)
    if "exam_id" in rra.columns and "exam_id" in probe.columns:
        probe_exam = set(probe["exam_id"].astype(str).tolist())
        return rra["exam_id"].astype(str).isin(probe_exam)
    if "row_idx" in rra.columns and "row_idx" in probe.columns:
        probe_row = set(probe["row_idx"].astype(int).tolist())
        return rra["row_idx"].astype(int).isin(probe_row)
    raise ValueError("Need shared key (exam_id preferred, otherwise row_idx)")


def _summary_table(df: pd.DataFrame, probe_n: int) -> pd.DataFrame:
    pos = df["rra_rank_pos"].notna()
    neg = df["rra_rank_neg"].notna()
    any_rra = pos | neg
    in_probe = df["in_probe"].astype(bool)

    rows = []
    for name, mask in [
        ("robust_positive", pos),
        ("robust_negative", neg),
        ("robust_any", any_rra),
    ]:
        total = int(mask.sum())
        in_p = int((mask & in_probe).sum())
        rows.append(
            {
                "group": name,
                "total": total,
                "in_probe": in_p,
                "pct_in_probe": (100.0 * in_p / total) if total else np.nan,
            }
        )
    rows.append(
        {
            "group": "probe_total",
            "total": int(probe_n),
            "in_probe": int(probe_n),
            "pct_in_probe": 100.0,
        }
    )
    return pd.DataFrame(rows)


def _by_dataset(df: pd.DataFrame, robust_col: str) -> pd.DataFrame:
    m = df[robust_col].notna()
    part = df.loc[m].copy()
    if "dataset_source" not in part.columns:
        return pd.DataFrame(columns=["robust_set", "dataset_source", "total", "in_probe", "pct_in_probe"])
    g = (
        part.groupby("dataset_source")
        .agg(
            total=("in_probe", "size"),
            in_probe=("in_probe", "sum"),
        )
        .reset_index()
    )
    g["pct_in_probe"] = 100.0 * g["in_probe"] / g["total"].clip(lower=1)
    g.insert(0, "robust_set", "robust_positive" if robust_col == "rra_rank_pos" else "robust_negative")
    return g


def _by_label(df: pd.DataFrame, robust_col: str) -> pd.DataFrame:
    m = df[robust_col].notna()
    part = df.loc[m].copy()
    if "chagas" not in part.columns:
        return pd.DataFrame(columns=["robust_set", "chagas", "total", "in_probe", "pct_in_probe"])
    g = (
        part.groupby("chagas")
        .agg(
            total=("in_probe", "size"),
            in_probe=("in_probe", "sum"),
        )
        .reset_index()
    )
    g["pct_in_probe"] = 100.0 * g["in_probe"] / g["total"].clip(lower=1)
    g.insert(0, "robust_set", "robust_positive" if robust_col == "rra_rank_pos" else "robust_negative")
    return g


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize overlap between probe-set samples and RRA robust-ranked samples."
    )
    parser.add_argument(
        "--rra_csv",
        type=Path,
        default=Path("analysis/embeddings_probe/ranking_agreement/test/sample_rra_consensus.csv"),
    )
    parser.add_argument(
        "--probe_index",
        type=Path,
        default=Path("analysis/embeddings_probe/probe_index.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis/embeddings_probe/ranking_agreement/test"),
    )
    args = parser.parse_args()

    rra = pd.read_csv(args.rra_csv, low_memory=False)
    probe = pd.read_csv(args.probe_index, low_memory=False)
    rra["in_probe"] = _mark_probe_membership(rra, probe)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_summary = args.out_dir / "rra_probe_overlap_summary.csv"
    out_dataset = args.out_dir / "rra_probe_overlap_by_dataset.csv"
    out_label = args.out_dir / "rra_probe_overlap_by_label.csv"

    s = _summary_table(rra, probe_n=len(probe))
    d = pd.concat(
        [
            _by_dataset(rra, "rra_rank_pos"),
            _by_dataset(rra, "rra_rank_neg"),
        ],
        ignore_index=True,
    )
    l = pd.concat(
        [
            _by_label(rra, "rra_rank_pos"),
            _by_label(rra, "rra_rank_neg"),
        ],
        ignore_index=True,
    )

    s.to_csv(out_summary, index=False)
    d.to_csv(out_dataset, index=False)
    l.to_csv(out_label, index=False)

    print(f"Wrote {out_summary}")
    print(f"Wrote {out_dataset}")
    print(f"Wrote {out_label}")


if __name__ == "__main__":
    main()
