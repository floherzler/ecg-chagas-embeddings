#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _read_allowlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    # preserve order, unique
    seen: set[str] = set()
    unique: list[str] = []
    for rid in out:
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(rid)
    return unique


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _logits_memmap_path(*, out_dir: Path, run_id: str, n: int) -> Path:
    # New layout
    p = out_dir / "runs" / run_id / "memmap" / f"{run_id}__logits__N{n}.fp32.mmap"
    if p.exists():
        return p
    # Legacy fallback
    return out_dir / "memmap" / f"{run_id}__logits__N{n}.fp32.mmap"


def _load_test_index(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "test_index.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Create it via `scripts/analysis/build_probe_set.py` or `run_probe_pipeline.py`."
        )
    df = pd.read_csv(p).sort_values("row_idx").reset_index(drop=True)
    if "chagas" not in df.columns and "y_true" in df.columns:
        df = df.rename(columns={"y_true": "chagas"})
    return df


def _load_or_build_test_metadata(*, out_dir: Path, meta_path: Path) -> pd.DataFrame:
    """
    Build a fold4-aligned metadata table for the full test set.

    We reuse `build_probe_metadata` because the required index schema is the same:
    row_idx, exam_id, dataset_source, chagas.
    """
    p = out_dir / "test_metadata.csv"
    if p.exists():
        return pd.read_csv(p)

    from ecg_chagas_embeddings.analysis.embeddings_probe import (
        DEFAULT_CODE15_EXAMS_PATH,
        DEFAULT_PTBXL_DB_PATH,
        DEFAULT_SAMITROP_EXAMS_PATH,
        build_probe_metadata,
        load_code15_exams,
        load_master_quality,
        load_ptbxl_database,
        load_samitrop_exams,
    )

    test_index = _load_test_index(out_dir)
    code15 = load_code15_exams(DEFAULT_CODE15_EXAMS_PATH)
    samitrop = load_samitrop_exams(DEFAULT_SAMITROP_EXAMS_PATH)
    ptbxl = load_ptbxl_database(DEFAULT_PTBXL_DB_PATH)
    master_quality = load_master_quality(meta_path=meta_path)

    df = build_probe_metadata(
        test_index,
        code15=code15,
        samitrop=samitrop,
        ptbxl=ptbxl,
        master_quality=master_quality,
    )
    df.to_csv(p, index=False)
    return df


def _patient_key(df: pd.DataFrame) -> pd.Series:
    pid = df.get("patient_id", pd.Series([np.nan] * len(df), index=df.index))
    pid = pid.astype("string[python]")
    ds = df["dataset_source"].astype("string[python]")
    exam = df["exam_id"].astype("string[python]")
    has_pid = pid.notna() & (pid != "NA") & (pid != "")
    key = pd.Series([""] * len(df), index=df.index, dtype="string[python]")
    key[has_pid] = ds[has_pid] + ":" + pid[has_pid]
    key[~has_pid] = ds[~has_pid] + ":" + exam[~has_pid]
    return key


@dataclass(frozen=True)
class CandidateGroup:
    name: str
    mask: np.ndarray  # boolean mask over rows
    sort_key: np.ndarray  # float array (higher is better)
    sort_desc: bool = True


def _pick_unique_patients(
    df: pd.DataFrame,
    *,
    group: CandidateGroup,
    patient_key_col: str,
    n: int,
    prefer_quality_mix: bool = True,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Pick up to n rows from df[group.mask], unique by patient_key_col.

    If prefer_quality_mix=True, try to mix qc_zhao2018_bp categories by round-robin.
    """
    if n <= 0:
        return df.iloc[0:0].copy()
    sub = df.loc[group.mask].copy()
    if sub.empty:
        return sub

    # Sort by key (stable). If ties remain, shuffle within ties for variety.
    order = np.argsort(group.sort_key[group.mask], kind="mergesort")
    if group.sort_desc:
        order = order[::-1]
    sub = sub.iloc[order].copy()

    used: set[str] = set()

    # Optional: simple round-robin over quality categories to encourage variety.
    if prefer_quality_mix and "qc_zhao2018_bp" in sub.columns:
        q = sub["qc_zhao2018_bp"].astype("string[python]").fillna("NA")
        cats = [c for c in q.unique().tolist() if c]
        # Stable order: excellent -> barely acceptable -> unacceptable -> NA -> others
        preferred = ["Excellent", "Barely acceptable", "Unacceptable", "NA"]
        cats = sorted(cats, key=lambda x: (preferred.index(x) if x in preferred else 999, x))
        buckets: dict[str, list[int]] = {c: [] for c in cats}
        for i, c in enumerate(q.tolist()):
            if c not in buckets:
                buckets[c] = []
            buckets[c].append(i)
        pick_idx: list[int] = []
        while len(pick_idx) < n and any(buckets.values()):
            progressed = False
            for c in cats:
                if not buckets.get(c):
                    continue
                # pop the next row, but enforce patient uniqueness
                while buckets[c]:
                    j = buckets[c].pop(0)
                    pk = str(sub.iloc[j][patient_key_col])
                    if pk in used:
                        continue
                    used.add(pk)
                    pick_idx.append(j)
                    progressed = True
                    break
                if len(pick_idx) >= n:
                    break
            if not progressed:
                break
        out = sub.iloc[pick_idx].copy()
        return out.reset_index(drop=True)

    # Default greedy unique-patient selection.
    picked_rows: list[int] = []
    idxs = np.arange(len(sub))
    # randomize a tiny bit to avoid always picking the same borderline cases
    # when many rows have identical sort scores.
    if len(sub) > 0:
        jitter = rng.normal(loc=0.0, scale=1e-9, size=len(sub))
        sub["_jitter"] = jitter
        sub = sub.sort_values(by=["_jitter"], kind="mergesort").drop(columns=["_jitter"])
        sub = sub.iloc[np.argsort(np.arange(len(sub)), kind="mergesort")]
    for i in idxs:
        pk = str(sub.iloc[int(i)][patient_key_col])
        if pk in used:
            continue
        used.add(pk)
        picked_rows.append(int(i))
        if len(picked_rows) >= n:
            break
    return sub.iloc[picked_rows].reset_index(drop=True)


def _write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    _add_src_to_path()

    from ecg_chagas_embeddings.analysis.embeddings_probe import DEFAULT_OUTPUT_DIR
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs

    parser = argparse.ArgumentParser(
        description="Select XAI candidate samples from fold4 based on model screening consensus (top-K%% across an allowlisted model pool)."
    )
    parser.add_argument(
        "--run_specs",
        type=Path,
        default=Path("configs/analysis/embeddings_probe_runs.toml"),
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allowlist",
        type=Path,
        required=True,
        help="Text file with one run_id per line (models used for consensus).",
    )
    parser.add_argument("--top_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--candidates_per_group",
        type=int,
        default=200,
        help="Max candidates sampled for each group (TP-like / FN-like / disagreement / etc.).",
    )
    parser.add_argument(
        "--disagreement_band",
        type=float,
        default=0.15,
        help="Half-width around 0.5 consensus for the disagreement pool (e.g. 0.15 => [0.35,0.65]).",
    )
    parser.add_argument(
        "--skip_missing_models",
        action="store_true",
        help="Skip allowlisted run_ids that are missing logits memmaps (instead of failing).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    allowlist_ids = _read_allowlist(args.allowlist)
    if not allowlist_ids:
        raise ValueError(f"Allowlist is empty: {args.allowlist}")

    global_cfg, runs = load_run_specs(args.run_specs)
    meta_path_raw = str(global_cfg.get("meta_path", "")).strip()
    if not meta_path_raw:
        raise ValueError("run_specs missing global 'meta_path'")
    meta_path = Path(meta_path_raw)

    test_index = _load_test_index(out_dir)
    N = int(len(test_index))
    if N <= 0:
        raise RuntimeError("Empty test_index.csv")
    k = int(math.ceil(float(args.top_frac) * N))
    k = max(1, min(N, k))

    # Resolve model pool from allowlist
    run_by_id = {r.run_id: r for r in runs}
    pool: list[str] = []
    missing_in_specs = [rid for rid in allowlist_ids if rid not in run_by_id]
    if missing_in_specs:
        raise ValueError(
            "Allowlist run_id(s) not present in run_specs: "
            + ", ".join(missing_in_specs[:20])
            + (" ..." if len(missing_in_specs) > 20 else "")
        )
    for rid in allowlist_ids:
        pool.append(rid)

    # Verify logits exist for the full test set.
    pool_kept: list[str] = []
    logits_paths: list[Path] = []
    for rid in pool:
        p = _logits_memmap_path(out_dir=out_dir, run_id=rid, n=N)
        if not p.exists():
            if args.skip_missing_models:
                print(
                    f"Skipping {rid}: missing full-test logits memmap {p}. "
                    "Create it by running `evaluate_test_models.py --save_logits` or the full pipeline."
                )
                continue
            raise FileNotFoundError(
                f"Missing full-test logits memmap for {rid}: {p}\n"
                "Create it by running `scripts/analysis/evaluate_test_models.py --save_logits` "
                "(or `scripts/analysis/run_probe_pipeline.py`)."
            )
        pool_kept.append(rid)
        logits_paths.append(p)

    M = int(len(pool_kept))
    if M < 2:
        raise RuntimeError(f"Need at least 2 models for consensus; got M={M}.")
    print(f"Using M={M} models for consensus (top_frac={args.top_frac:.3f}, k={k}, N={N})")

    # Build test metadata (includes patient_id, quality, phenotypes, etc.).
    test_meta = _load_or_build_test_metadata(out_dir=out_dir, meta_path=meta_path)
    if int(len(test_meta)) != N:
        raise RuntimeError(
            f"test_metadata.csv rows ({len(test_meta)}) do not match test_index.csv rows ({N})."
        )
    df = test_meta.copy()
    df["patient_key"] = _patient_key(df)

    # Compute membership and consensus counts using boolean operations.
    xai_dir = out_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    membership_path = xai_dir / f"top5_membership__N{N}__M{M}.u8.mmap"
    membership = np.memmap(membership_path, mode="w+", dtype="uint8", shape=(N, M))
    membership[:] = 0

    consensus_count = np.zeros(N, dtype=np.int32)
    sum_logits = np.zeros(N, dtype=np.float64)
    sumsq_logits = np.zeros(N, dtype=np.float64)
    sum_prob = np.zeros(N, dtype=np.float64)

    for m, (rid, p) in enumerate(tqdm(list(zip(pool_kept, logits_paths)), desc="Models", unit="model")):
        logits = np.memmap(p, mode="r", dtype="float32", shape=(N,))
        x = np.asarray(logits, dtype=np.float64)

        sum_logits += x
        sumsq_logits += x * x
        sum_prob += _sigmoid(x)

        idx = np.argpartition(x, -k)[-k:]
        membership[idx, m] = 1
        consensus_count[idx] += 1

    membership.flush()
    (xai_dir / "allowlist_run_ids.txt").write_text("\n".join(pool_kept) + "\n", encoding="utf-8")
    print(f"Wrote {membership_path}")
    print(f"Wrote {xai_dir / 'allowlist_run_ids.txt'}")

    consensus_frac = (consensus_count.astype(np.float64) / float(M)).astype(np.float32)
    mean_logit = (sum_logits / float(M)).astype(np.float32)
    var_logit = (sumsq_logits / float(M)) - (sum_logits / float(M)) ** 2
    std_logit = np.sqrt(np.maximum(var_logit, 0.0)).astype(np.float32)
    mean_prob = (sum_prob / float(M)).astype(np.float32)

    df["top5_count_models"] = consensus_count
    df["top5_frac_models"] = consensus_frac
    df["mean_logit_models"] = mean_logit
    df["std_logit_models"] = std_logit
    df["mean_prob_models"] = mean_prob

    # Full table (all test samples; easy to filter interactively).
    keep_cols = [
        c
        for c in [
            "row_idx",
            "exam_id",
            "dataset_source",
            "patient_id",
            "patient_key",
            "chagas",
            "age",
            "is_male",
            "delta_age",
            "normal_ecg",
            "RBBB",
            "LBBB",
            "1dAVb",
            "AF",
            "ptb_crbbb",
            "ptb_irbbb",
            "ptb_any_rbbb",
            "ptb_lafb",
            "death",
            "timey",
            "qc_zhao2018_bp",
            "qc_templatematch_bp",
            "resample_method",
            "top5_count_models",
            "top5_frac_models",
            "mean_prob_models",
            "mean_logit_models",
            "std_logit_models",
        ]
        if c in df.columns
    ]
    full_path = xai_dir / "test_consensus_full.csv"
    _write_df(full_path, df[keep_cols].sort_values("row_idx"))
    print(f"Wrote {full_path} (rows={len(df)})")

    # Candidate pools
    ch = df.get("chagas", pd.Series([np.nan] * len(df))).astype(float).to_numpy()
    cf = df["top5_frac_models"].to_numpy(dtype=np.float32)
    sd = df["std_logit_models"].to_numpy(dtype=np.float32)
    mp = df["mean_prob_models"].to_numpy(dtype=np.float32)

    band = float(args.disagreement_band)
    lo, hi = float(0.5 - band), float(0.5 + band)
    disagree_mask = (cf >= lo) & (cf <= hi)

    groups: list[CandidateGroup] = [
        # High-consensus screening set (likely TP/FP depending on label)
        CandidateGroup(
            name="high_consensus__chagas1",
            mask=(ch == 1) & np.isfinite(ch),
            sort_key=cf,  # highest consensus first
            sort_desc=True,
        ),
        CandidateGroup(
            name="high_consensus__chagas0",
            mask=(ch == 0) & np.isfinite(ch),
            sort_key=cf,
            sort_desc=True,
        ),
        # Low-consensus "unselected" set (likely FN/TN depending on label)
        CandidateGroup(
            name="low_consensus__chagas1",
            mask=(ch == 1) & np.isfinite(ch),
            sort_key=cf,  # lowest consensus first
            sort_desc=False,
        ),
        CandidateGroup(
            name="low_consensus__chagas0",
            mask=(ch == 0) & np.isfinite(ch),
            sort_key=cf,
            sort_desc=False,
        ),
        # Disagreement region: prefer high dispersion in model scores
        CandidateGroup(
            name="disagreement__any_label",
            mask=disagree_mask & np.isfinite(cf),
            sort_key=(sd + 0.25 * mp),  # emphasize score dispersion; slight bias to higher risk
            sort_desc=True,
        ),
    ]

    rng = np.random.default_rng(int(args.seed))
    patient_key_col = "patient_key"
    cand_rows: list[pd.DataFrame] = []
    for g in groups:
        picked = _pick_unique_patients(
            df,
            group=g,
            patient_key_col=patient_key_col,
            n=int(args.candidates_per_group),
            prefer_quality_mix=True,
            rng=rng,
        )
        if picked.empty:
            continue
        picked.insert(0, "group", g.name)
        cand_rows.append(picked)

    if cand_rows:
        candidates = pd.concat(cand_rows, axis=0, ignore_index=True)
    else:
        candidates = df.iloc[0:0].copy()

    # Enforce global uniqueness across the candidate table as well (optional but useful).
    # Keep first occurrence per patient_key, favoring earlier groups and higher-priority rows.
    if not candidates.empty and patient_key_col in candidates.columns:
        candidates = candidates.drop_duplicates(subset=[patient_key_col], keep="first").reset_index(drop=True)

    cand_path = xai_dir / "test_consensus_candidates.csv"
    _write_df(cand_path, candidates[["group"] + keep_cols] if keep_cols else candidates)
    print(f"Wrote {cand_path} (rows={len(candidates)})")

    # Helpful group summaries
    summary = (
        candidates.groupby(["group", "dataset_source", "chagas"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["group", "dataset_source", "chagas"])
    )
    summary_path = xai_dir / "test_consensus_candidates_summary.csv"
    _write_df(summary_path, summary)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

