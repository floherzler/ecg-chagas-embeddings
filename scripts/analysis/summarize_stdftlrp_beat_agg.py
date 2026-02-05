#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _normalize_source(s: str) -> str:
    s = str(s).strip().upper().replace("-", "").replace("_", "")
    if "CODE" in s:
        return "CODE15"
    if "SAMITROP" in s:
        return "SAMITROP"
    if "PTB" in s:
        return "PTBXL"
    return s


def _ensure_group_cols(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in group_cols:
        if col not in df.columns:
            df[col] = "NA"
    return df


def _build_group_key(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    parts = [df[c].astype(str) for c in group_cols]
    return parts[0].str.cat(parts[1:], sep=" | ")


def _pick_metric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    cols += [c for c in df.columns if c.startswith("p_lead_")]
    cols += [c for c in df.columns if c.startswith("lead_entropy")]
    cols += [c for c in df.columns if c.startswith("rel_weighted_")]
    cols += [c for c in df.columns if c.startswith("freq_rel_weighted_")]
    return sorted(set(cols))


def _summarize(df: pd.DataFrame, *, group_cols: list[str], min_n: int) -> pd.DataFrame:
    df = df.copy()
    # Clean up CSV headers (some files include extra whitespace)
    df.columns = [c.strip() for c in df.columns]
    if "source" in df.columns:
        df["source"] = df["source"].map(_normalize_source)
    # non-overlapping groups via full Cartesian of group_cols
    df = _ensure_group_cols(df, group_cols)
    df["group"] = _build_group_key(df, group_cols)
    metrics = _pick_metric_cols(df)
    if not metrics:
        raise ValueError("No metric columns found to summarize.")

    agg = df.groupby(["group", *group_cols], dropna=False)[metrics].mean().reset_index()
    counts = df.groupby(["group", *group_cols], dropna=False).size().reset_index(name="n")
    out = agg.merge(counts, on=["group", *group_cols], how="left")
    if min_n > 1:
        out = out[out["n"] >= min_n].reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-sample STDFT-LRP aggregates into group means.")
    parser.add_argument("--run_specs", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--group_by",
        action="append",
        default=[],
        help="Column to group by (repeatable). Defaults to: source,chagas.",
    )
    parser.add_argument("--min_n", type=int, default=10)
    args = parser.parse_args()

    _add_src_to_path()
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs

    group_cols = args.group_by if args.group_by else ["source", "chagas"]

    global_cfg, runs = load_run_specs(args.run_specs)
    out_root = args.out_dir / "xai_summary"
    out_root.mkdir(parents=True, exist_ok=True)

    per_run_rows: list[pd.DataFrame] = []
    for run in runs:
        # Current layout: runs/<run_id>/xai/lead_*/stdftlrp_beat_agg.csv
        lead_paths = sorted((args.out_dir / "runs" / run.run_id / "xai").glob("lead_*/stdftlrp_beat_agg.csv"))
        in_path = None
        if lead_paths:
            # Prefer lead_1 if present (lead II), otherwise take the first available lead.
            lead1 = [p for p in lead_paths if p.parent.name == "lead_1"]
            in_path = lead1[0] if lead1 else lead_paths[0]
        else:
            # Backward-compat: runs/<run_id>/xai/stdftlrp_beat_agg.csv
            legacy_path = args.out_dir / "runs" / run.run_id / "xai" / "stdftlrp_beat_agg.csv"
            if legacy_path.exists():
                in_path = legacy_path
            else:
                # Older layout: runs/<run_id>/stdftlrp_beat_agg.csv
                old_path = args.out_dir / "runs" / run.run_id / "stdftlrp_beat_agg.csv"
                if old_path.exists():
                    in_path = old_path
                else:
                    print(f"Skipping {run.run_id}: missing stdftlrp_beat_agg.csv")
                    continue
        df = pd.read_csv(in_path)
        summary = _summarize(df, group_cols=group_cols, min_n=int(args.min_n))
        summary.insert(0, "run_id", run.run_id)
        summary.insert(1, "track", str(run.track))
        summary.insert(2, "preprocessing", str(run.preprocessing))
        summary.insert(3, "loss", str(getattr(run, "loss", "")))
        per_run_rows.append(summary)

    if not per_run_rows:
        raise SystemExit("No summaries written (no input files found).")

    per_run = pd.concat(per_run_rows, ignore_index=True)
    per_run_path = out_root / "stdftlrp_summary_per_run.csv"
    per_run.to_csv(per_run_path, index=False)
    print(f"Wrote {per_run_path}")

    # Optional: model-group summary (mean across runs) using the same non-overlapping groups.
    model_group_cols = ["track", "preprocessing", "loss", "group"]
    metric_cols = _pick_metric_cols(per_run)
    by_model_group = (
        per_run.groupby(model_group_cols, dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    group_path = out_root / "stdftlrp_summary_by_model_group.csv"
    by_model_group.to_csv(group_path, index=False)
    print(f"Wrote {group_path}")


if __name__ == "__main__":
    main()
