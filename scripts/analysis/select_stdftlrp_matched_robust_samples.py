#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_source(s: object) -> str:
    x = str(s).strip().upper().replace("-", "").replace("_", "").replace("%", "")
    if "CODE" in x:
        return "CODE15"
    if "SAMITROP" in x:
        return "SAMITROP"
    if "PTB" in x:
        return "PTBXL"
    return x


def _collect_common_xai_keys(out_dir: Path, lead_tag: str) -> tuple[set[str], list[str]]:
    run_paths = sorted((out_dir / "runs").glob(f"*/xai/{lead_tag}/stdftlrp_beat_agg.csv"))
    if not run_paths:
        raise FileNotFoundError(
            f"No per-run ST-DFT-LRP files found under {out_dir / 'runs'} for lead tag '{lead_tag}'."
        )

    key_sets: list[set[str]] = []
    run_ids: list[str] = []
    for p in run_paths:
        run_id = p.parts[-4]
        run_ids.append(run_id)
        df = pd.read_csv(p, usecols=["source", "exam_id"])
        src = df["source"].map(_normalize_source)
        keys = set((src.astype(str) + "||" + df["exam_id"].astype(str)).tolist())
        key_sets.append(keys)

    common_keys = set.intersection(*key_sets)
    return common_keys, run_ids


def _sort_robust(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Robust positives: smaller rank = more extreme high-score consensus.
    # Robust negatives: larger transformed rank = more extreme low-score consensus.
    out["_sort_key"] = out["rra_rank"]
    out.loc[out["robust_set"].eq("robust_neg"), "_sort_key"] = -out.loc[
        out["robust_set"].eq("robust_neg"), "rra_rank"
    ]
    out = out.sort_values(
        ["robust_set", "dataset_source", "chagas", "_sort_key", "exam_id"],
        ascending=[True, True, True, True, True],
    ).drop(columns="_sort_key")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select robust RRA samples available in all ST-DFT-LRP runs for a given lead tag."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis/embeddings_probe"),
        help="Probe pipeline output directory.",
    )
    parser.add_argument(
        "--rra_csv",
        type=Path,
        default=Path("analysis/embeddings_probe/ranking_agreement/test/sample_rra_consensus.csv"),
        help="RRA consensus CSV.",
    )
    parser.add_argument(
        "--lead_tag",
        type=str,
        default="lead_7",
        help="Lead tag under runs/*/xai/<lead_tag>/stdftlrp_beat_agg.csv.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="Top-k rows to keep per (robust_set, dataset_source, chagas).",
    )
    args = parser.parse_args()

    common_keys, run_ids = _collect_common_xai_keys(args.out_dir, args.lead_tag)

    rra = pd.read_csv(args.rra_csv, low_memory=False)
    src_col = "dataset_source" if "dataset_source" in rra.columns else "source"
    if src_col not in rra.columns or "exam_id" not in rra.columns:
        raise ValueError("RRA CSV must contain dataset source and exam_id columns.")
    rra["dataset_source"] = rra[src_col].map(_normalize_source)
    rra["exam_id"] = rra["exam_id"].astype(str)
    rra["key"] = rra["dataset_source"].astype(str) + "||" + rra["exam_id"]
    rra["in_all_runs"] = rra["key"].isin(common_keys)

    pos = rra[rra["in_all_runs"] & rra["rra_rank_pos"].notna()].copy()
    pos["robust_set"] = "robust_pos"
    pos["rra_rank"] = pd.to_numeric(pos["rra_rank_pos"], errors="coerce")

    neg = rra[rra["in_all_runs"] & rra["rra_rank_neg"].notna()].copy()
    neg["robust_set"] = "robust_neg"
    neg["rra_rank"] = pd.to_numeric(neg["rra_rank_neg"], errors="coerce")

    matched = pd.concat([pos, neg], ignore_index=True)
    keep_cols = [
        "robust_set",
        "dataset_source",
        "chagas",
        "exam_id",
        "rra_rank",
        "row_idx",
        "in_all_runs",
    ]
    keep_cols = [c for c in keep_cols if c in matched.columns]
    matched = _sort_robust(matched[keep_cols].copy())

    out_root = args.out_dir / "xai_summary" / args.lead_tag
    out_root.mkdir(parents=True, exist_ok=True)

    out_all = out_root / "matched_robust_samples_all_runs.csv"
    matched.to_csv(out_all, index=False)

    counts = (
        matched.groupby(["robust_set", "dataset_source", "chagas"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["robust_set", "dataset_source", "chagas"])
    )
    out_counts = out_root / "matched_robust_counts_all_runs.csv"
    counts.to_csv(out_counts, index=False)

    topk = (
        matched.groupby(["robust_set", "dataset_source", "chagas"], dropna=False, as_index=False)
        .head(int(args.top_k))
        .copy()
    )
    out_topk = out_root / f"matched_robust_top{int(args.top_k)}_all_runs.csv"
    topk.to_csv(out_topk, index=False)

    print(f"Lead tag: {args.lead_tag}")
    print(f"Runs with XAI outputs: {len(run_ids)}")
    print(f"Common samples across runs: {len(common_keys)}")
    print(f"Wrote {out_all}")
    print(f"Wrote {out_counts}")
    print(f"Wrote {out_topk}")


if __name__ == "__main__":
    main()
