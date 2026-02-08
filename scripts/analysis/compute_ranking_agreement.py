#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import pyflagr.RRA as RRA


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _load_index(out_dir: Path, *, set_name: str) -> pd.DataFrame:
    if set_name == "probe":
        p = out_dir / "probe_index.csv"
    elif set_name == "test":
        p = out_dir / "test_index.csv"
    else:
        raise ValueError(f"Unknown set_name: {set_name!r}")
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p).sort_values("row_idx").reset_index(drop=True)
    if "chagas" not in df.columns and "y_true" in df.columns:
        df = df.rename(columns={"y_true": "chagas"})
    return df


def _logits_memmap_path(*, out_dir: Path, run_id: str, n: int) -> Path:
    # New layout
    p = out_dir / "runs" / run_id / "memmap" / f"{run_id}__logits__N{n}.fp32.mmap"
    if p.exists():
        return p
    # Legacy fallback
    p = out_dir / "memmap" / f"{run_id}__logits__N{n}.fp32.mmap"
    return p


def _load_logits(
    *, out_dir: Path, run_ids: list[str], n: int, skip_missing: bool
) -> tuple[list[str], np.ndarray]:
    kept: list[str] = []
    cols: list[np.ndarray] = []
    for rid in tqdm(run_ids, desc="Load logits", unit="run"):
        p = _logits_memmap_path(out_dir=out_dir, run_id=rid, n=n)
        if not p.exists():
            if skip_missing:
                print(f"Skipping {rid}: missing logits memmap {p}")
                continue
            raise FileNotFoundError(f"Missing logits memmap for {rid}: {p}")
        x = np.memmap(p, mode="r", dtype="float32", shape=(n,))
        cols.append(np.asarray(x, dtype=np.float32))
        kept.append(rid)
    if not kept:
        raise RuntimeError("No runs available (all missing?)")
    scores = np.stack(cols, axis=0)  # [M, N]
    return kept, scores


def _rank_ordinal(scores: np.ndarray) -> np.ndarray:
    """
    Fast ordinal ranks (ties broken deterministically by stable sort).

    Returns ranks in [0..N-1] where higher score => higher rank.
    """
    scores = np.asarray(scores)
    M, N = scores.shape
    ranks = np.empty((M, N), dtype=np.float32)
    for i in range(M):
        order = np.argsort(scores[i], kind="mergesort")
        r = np.empty(N, dtype=np.int32)
        r[order] = np.arange(N, dtype=np.int32)
        ranks[i] = r.astype(np.float32)
    return ranks


def _spearman_rho_matrix(scores: np.ndarray) -> np.ndarray:
    """
    Spearman rho computed as Pearson correlation of ranks.
    """
    r = _rank_ordinal(scores)
    r = r - r.mean(axis=1, keepdims=True)
    num = r @ r.T
    den = np.sqrt(np.sum(r * r, axis=1, keepdims=True))
    den = den @ den.T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def _topk_mask(scores: np.ndarray, *, frac: float) -> np.ndarray:
    """
    Boolean mask of top-k by score per model: [M, N]
    """
    M, N = scores.shape
    k = int(np.ceil(float(frac) * N))
    k = max(1, min(N, k))
    mask = np.zeros((M, N), dtype=bool)
    for i in range(M):
        idx = np.argpartition(scores[i], -k)[-k:]
        mask[i, idx] = True
    return mask


def _jaccard_iou_matrix(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int16)
    inter = (m @ m.T).astype(np.float32)
    k = np.diag(inter).reshape(-1, 1)
    union = k + k.T - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = inter / union
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def _count_inversions_int(arr: np.ndarray) -> int:
    """
    Count inversions in an integer array using merge-sort. O(n log n).
    """
    a = np.asarray(arr, dtype=np.int64)
    tmp = np.empty_like(a)

    def _sort(lo: int, hi: int) -> int:
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        inv = _sort(lo, mid) + _sort(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if a[i] <= a[j]:
                tmp[k] = a[i]
                i += 1
            else:
                tmp[k] = a[j]
                j += 1
                inv += mid - i
            k += 1
        while i < mid:
            tmp[k] = a[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = a[j]
            j += 1
            k += 1
        a[lo:hi] = tmp[lo:hi]
        return inv

    return int(_sort(0, int(len(a))))


def _kendall_tau_a(x: np.ndarray, y: np.ndarray) -> float:
    """
    Kendall's tau-a (ties ignored). For continuous logits, ties are rare.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 3:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    y_ord = y[order]
    # Convert to ordinal ranks to count inversions (discordant pairs).
    y_rank = np.empty(n, dtype=np.int64)
    y_rank[np.argsort(y_ord, kind="mergesort")] = np.arange(n, dtype=np.int64)
    discordant = _count_inversions_int(y_rank)
    total = n * (n - 1) // 2
    if total <= 0:
        return float("nan")
    concordant = total - discordant
    return float((concordant - discordant) / float(total))


def _kendall_topk_matrix(
    scores: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kendall tau-a on intersection of top-k sets.

    Returns:
      tau[M,M], n_intersection[M,M]
    """
    M, _N = scores.shape
    tau = np.full((M, M), np.nan, dtype=np.float32)
    n_int = np.zeros((M, M), dtype=np.int32)
    for i in range(M):
        tau[i, i] = 1.0
        n_int[i, i] = int(mask[i].sum())
    for i in range(M):
        for j in range(i + 1, M):
            inter = mask[i] & mask[j]
            nij = int(inter.sum())
            n_int[i, j] = nij
            n_int[j, i] = nij
            if nij < 3:
                continue
            xi = scores[i, inter]
            xj = scores[j, inter]
            t = _kendall_tau_a(xi, xj)
            tau[i, j] = float(t)
            tau[j, i] = float(t)
    return tau, n_int


def _write_matrix_csv(path: Path, run_ids: list[str], mat: np.ndarray) -> None:
    df = pd.DataFrame(mat, index=run_ids, columns=run_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def _build_rra_lists(
    scores: np.ndarray,
    run_ids: list[str],
    item_ids: np.ndarray,
    *,
    top_frac: float,
    query_id: str = "1",
    list_name: str = "ECG",
) -> pd.DataFrame:
    """
    Build a long-form ranking table for RRA aggregation.

    Columns (current convention):
      - list_id: run_id for each model (list)
      - item_id: unique sample id (row_idx)
      - rank: 1 = best (highest logit), increasing
      - score: raw logit (kept for debugging)

    NOTE: PyFLAGR's expected schema may differ. See TODO in main() to adjust.
    """
    M, N = scores.shape
    k = int(np.ceil(float(top_frac) * N))
    k = max(1, min(N, k))
    rows: list[dict[str, Any]] = []
    for i, rid in enumerate(tqdm(run_ids, desc="Build RRA lists", unit="run")):
        s = scores[i]
        idx = np.argpartition(s, -k)[-k:]
        # Sort top-k by score descending for proper rank order
        idx = idx[np.argsort(s[idx])[::-1]]
        for r, j in enumerate(idx, start=1):
            item_tag = f"Q{query_id}-E{int(item_ids[j]) + 1}"
            rows.append(
                {
                    "Query": f"Q{query_id}",
                    "Voter": f"V-{int(i)}",
                    "ItemID": item_tag,
                    "Rank": int(r),
                    "Score": float(s[j]),
                    "List": list_name,
                }
            )
    return pd.DataFrame(
        rows, columns=["Query", "Voter", "ItemID", "Rank", "Score", "List"]
    )


def _infer_rra_columns(df_out: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Best-effort guessing of item_id and score columns from PyFLAGR output.
    Returns (item_col, score_col).
    """
    item_candidates = [
        "ItemID",
        "item_id",
        "item",
        "docno",
        "document",
        "object",
        "element",
        "id",
    ]
    score_candidates = [
        "Score",
        "score",
        "rra_score",
        "pvalue",
        "p_value",
        "p",
        "agg_score",
    ]
    item_col = next((c for c in item_candidates if c in df_out.columns), None)
    score_col = next((c for c in score_candidates if c in df_out.columns), None)
    return item_col, score_col


def _rank_from_score(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    order = np.argsort(values)
    if higher_is_better:
        order = order[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1, dtype=int)
    return ranks


def _parse_rra_output(df_out: pd.DataFrame, *, n: int) -> pd.DataFrame:
    item_col, score_col = _infer_rra_columns(df_out)
    if item_col is None or score_col is None:
        raise ValueError("Could not infer item/score columns from RRA output.")
    rra_df = df_out[[item_col, score_col]].copy()
    rra_df = rra_df.rename(columns={item_col: "row_idx", score_col: "rra_score"})
    if rra_df["row_idx"].dtype == object:
        rra_df["rra_item_id"] = rra_df["row_idx"].astype(str)
        rra_df["row_idx"] = (
            rra_df["row_idx"]
            .astype(str)
            .str.extract(r"E(\d+)", expand=False)
            .astype(float)
        )
    if rra_df["row_idx"].isna().all():
        raise ValueError("Failed to parse ItemID into row_idx (all NaN).")
    if rra_df["row_idx"].min() >= 1 and rra_df["row_idx"].max() == n:
        rra_df["row_idx"] = rra_df["row_idx"].astype(int) - 1
    if "Rank" in df_out.columns:
        rra_df["rra_rank"] = pd.to_numeric(df_out["Rank"], errors="coerce")
    else:
        lower_better = "p" in score_col.lower()
        rra_df["rra_rank"] = _rank_from_score(
            rra_df["rra_score"].to_numpy(dtype=float),
            higher_is_better=not lower_better,
        )
    return rra_df


def _write_run_id_list(path: Path, run_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(run_ids) + "\n", encoding="utf-8")


def main() -> None:
    _add_src_to_path()

    from ecg_chagas_embeddings.analysis.embeddings_probe import DEFAULT_OUTPUT_DIR
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs

    parser = argparse.ArgumentParser(
        description="Model-vs-model ranking agreement on a fixed set (Spearman rho, top-k IoU, top-k Kendall tau) + per-sample top-k consensus."
    )
    parser.add_argument(
        "--run_specs",
        type=Path,
        default=Path("configs/analysis/embeddings_probe_runs.toml"),
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--set",
        type=str,
        choices=["test", "probe"],
        default="test",
        help="Which frozen index to use: test_index.csv or probe_index.csv.",
    )
    parser.add_argument("--top_frac", type=float, default=0.05)
    parser.add_argument(
        "--rra",
        action="store_true",
        help="Compute Robust Rank Aggregation (PyFLAGR) on top-k lists and write per-sample RRA scores.",
    )
    parser.add_argument(
        "--rra_top_frac",
        type=float,
        default=1.0,
        help="Top fraction to include in RRA lists (default: 1.0 = full ranking).",
    )
    parser.add_argument("--rra_eval_pts", type=int, default=7)
    parser.add_argument("--rra_exact", action="store_true")
    parser.add_argument(
        "--rra_subprocess",
        action="store_true",
        help="Run PyFLAGR RRA in a subprocess to isolate potential segfaults.",
    )
    parser.add_argument(
        "--rra_fill_missing",
        action="store_true",
        help="Fill missing RRA entries with worst score/rank (score=1.0, rank=N).",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip runs that are missing logits memmaps instead of failing.",
    )
    parser.add_argument(
        "--write_membership_csv",
        action="store_true",
        help="Also write a wide CSV with per-sample membership (can be large on the full test set).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    index_df = _load_index(out_dir, set_name=str(args.set))
    N = int(len(index_df))

    _global_cfg, runs = load_run_specs(args.run_specs)
    run_ids = [r.run_id for r in runs]

    kept_ids, scores = _load_logits(
        out_dir=out_dir, run_ids=run_ids, n=N, skip_missing=bool(args.skip_missing)
    )
    M = int(len(kept_ids))
    print(f"Loaded logits for {M} runs on {args.set} set N={N}")

    # Dense aggregation: median rank across all models (full ordering)
    median_rank_full = np.median(_rank_ordinal(scores), axis=0).astype(np.float32) + 1.0

    out_base = out_dir / "ranking_agreement" / str(args.set)
    out_base.mkdir(parents=True, exist_ok=True)

    # (1) Spearman rho over all probe predictions (global agreement)
    rho = _spearman_rho_matrix(scores)
    _write_matrix_csv(out_base / "spearman_rho.csv", kept_ids, rho)
    print(f"Wrote {out_base / 'spearman_rho.csv'}")

    # (2) Top-k overlap (Jaccard / IoU)
    mask = _topk_mask(scores, frac=float(args.top_frac))
    iou = _jaccard_iou_matrix(mask)
    _write_matrix_csv(out_base / "top5_iou.csv", kept_ids, iou)
    print(f"Wrote {out_base / 'top5_iou.csv'}")

    # (3) Within-screening ranking agreement (Kendall tau on intersection)
    tau, n_int = _kendall_topk_matrix(scores, mask)
    _write_matrix_csv(out_base / "top5_kendall_tau.csv", kept_ids, tau)
    _write_matrix_csv(
        out_base / "top5_intersection_n.csv", kept_ids, n_int.astype(np.float32)
    )
    print(f"Wrote {out_base / 'top5_kendall_tau.csv'}")
    print(f"Wrote {out_base / 'top5_intersection_n.csv'}")

    # (4) Per-sample consensus for top-k (Step 2 in your screenshot)
    consensus_count = mask.sum(axis=0).astype(np.int32)  # [N]
    consensus_frac = (consensus_count.astype(np.float32) / float(M)).astype(np.float32)
    keep_cols = [
        c
        for c in ["row_idx", "exam_id", "dataset_source", "chagas"]
        if c in index_df.columns
    ]
    consensus_df = index_df[keep_cols].copy()
    if "chagas" in consensus_df.columns:
        consensus_df["chagas"] = consensus_df["chagas"].astype(int)
    consensus_df["top5_count_models"] = consensus_count
    consensus_df["top5_frac_models"] = consensus_frac
    consensus_df.to_csv(out_base / "sample_top5_consensus.csv", index=False)
    print(f"Wrote {out_base / 'sample_top5_consensus.csv'}")

    # (5) Persist full membership matrix (fast to load later; avoids huge CSVs).
    membership_mm_path = out_base / f"top5_membership__N{N}__M{M}.u8.mmap"
    mm = np.memmap(membership_mm_path, mode="w+", dtype="uint8", shape=(N, M))
    mm[:] = mask.T.astype(np.uint8)
    mm.flush()
    _write_run_id_list(out_base / "top5_membership_run_ids.txt", kept_ids)
    print(f"Wrote {membership_mm_path}")
    print(f"Wrote {out_base / 'top5_membership_run_ids.txt'}")

    if args.write_membership_csv:
        membership = pd.DataFrame(mask.T.astype(np.int8), columns=kept_ids)
        membership.insert(0, "row_idx", index_df["row_idx"].to_numpy(dtype=int))
        membership.insert(1, "exam_id", index_df["exam_id"].astype(str).to_numpy())
        membership.to_csv(out_base / "top5_membership.csv", index=False)
        print(f"Wrote {out_base / 'top5_membership.csv'}")

    if args.rra:
        # (6) Robust Rank Aggregation (PyFLAGR)
        def _run_rra(lists_path: Path, *, tag: str) -> pd.DataFrame:
            print(f"Running RRA aggregation ({tag})...")
            if args.rra_subprocess:
                code = (
                    "import pandas as pd\n"
                    "import pyflagr.RRA as RRA\n"
                    "from pathlib import Path\n"
                    f'lists_path = Path(r"{lists_path}")\n'
                    f'out_base = Path(r"{out_base}")\n'
                    f"rra = RRA.RRA(eval_pts={int(args.rra_eval_pts)}, exact={bool(args.rra_exact)})\n"
                    "df_out, df_eval = rra.aggregate(input_file=str(lists_path), out_dir=str(out_base))\n"
                    f'df_out.to_csv(out_base / "rra_output_{tag}.csv", index=False)\n'
                    f'df_eval.to_csv(out_base / "rra_eval_{tag}.csv", index=False)\n'
                )
                log_path = out_base / f"rra_subprocess_{tag}.log"
                with log_path.open("w", encoding="utf-8") as log_f:
                    log_f.write(f"RRA subprocess starting ({tag})...\n")
                    log_f.flush()
                    proc = subprocess.run(
                        [sys.executable, "-X", "faulthandler", "-c", code],
                        check=False,
                        stdout=log_f,
                        stderr=log_f,
                    )
                    log_f.write(f"\nRRA subprocess returncode={proc.returncode}\n")
                if proc.returncode != 0:
                    if proc.returncode < 0:
                        print(
                            "WARNING: RRA subprocess terminated by signal "
                            f"{-proc.returncode}. This can indicate a segfault or OOM."
                        )
                    print(
                        "WARNING: RRA subprocess failed (non-zero exit). "
                        f"See {log_path} for details."
                    )
                    raise RuntimeError("RRA subprocess failed")
                out_path = out_base / f"rra_output_{tag}.csv"
                eval_path = out_base / f"rra_eval_{tag}.csv"
                print(f"Wrote {out_path}")
                print(f"Wrote {eval_path}")
                return pd.read_csv(out_path)
            try:
                import pyflagr.RRA as RRA  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(f"Failed to import pyflagr: {exc}") from exc
            rra = RRA.RRA(eval_pts=int(args.rra_eval_pts), exact=bool(args.rra_exact))
            df_out, df_eval = rra.aggregate(
                input_file=str(lists_path), out_dir=str(out_base)
            )
            out_path = out_base / f"rra_output_{tag}.csv"
            eval_path = out_base / f"rra_eval_{tag}.csv"
            df_out.to_csv(out_path, index=False)
            df_eval.to_csv(eval_path, index=False)
            print(f"Wrote {out_path}")
            print(f"Wrote {eval_path}")
            return df_out

        lists_path = out_base / "rra_lists.csv"
        lists_debug_path = out_base / "rra_lists_debug.csv"
        voter_map_path = out_base / "rra_voter_map.csv"
        item_map_path = out_base / "rra_item_map.csv"
        if not lists_path.exists():
            item_ids = index_df["row_idx"].to_numpy(dtype=int)
            lists_df = _build_rra_lists(
                scores, kept_ids, item_ids, top_frac=float(args.rra_top_frac)
            )
            # PyFLAGR expects header-less CSV: Query, Voter, ItemID, Rank, Score, List
            lists_df.to_csv(lists_path, index=False, header=False)
            lists_df.to_csv(lists_debug_path, index=False)
            pd.DataFrame(
                {"voter_id": np.arange(len(kept_ids)), "run_id": kept_ids}
            ).to_csv(voter_map_path, index=False)
            pd.DataFrame(
                {
                    "item_id": [f"Q1-E{int(i) + 1}" for i in item_ids],
                    "row_idx": item_ids,
                }
            ).to_csv(item_map_path, index=False)
            print(f"Wrote {lists_path}")
            print(f"Wrote {lists_debug_path}")
            print(f"Wrote {voter_map_path}")
            print(f"Wrote {item_map_path}")
        else:
            print(f"Using existing {lists_path}")

        lists_inv_path = out_base / "rra_lists_inverted.csv"
        lists_inv_debug_path = out_base / "rra_lists_inverted_debug.csv"
        if not lists_inv_path.exists():
            item_ids = index_df["row_idx"].to_numpy(dtype=int)
            lists_inv_df = _build_rra_lists(
                -scores, kept_ids, item_ids, top_frac=float(args.rra_top_frac)
            )
            lists_inv_df.to_csv(lists_inv_path, index=False, header=False)
            lists_inv_df.to_csv(lists_inv_debug_path, index=False)
            print(f"Wrote {lists_inv_path}")
            print(f"Wrote {lists_inv_debug_path}")
        else:
            print(f"Using existing {lists_inv_path}")

        df_out_pos = _run_rra(lists_path, tag="pos")
        df_out_neg = _run_rra(lists_inv_path, tag="neg")

        try:
            rra_pos = _parse_rra_output(df_out_pos, n=N).rename(
                columns={"rra_rank": "rra_rank_pos"}
            )
        except Exception as exc:
            print(f"WARNING: Failed to parse RRA positive output: {exc}")
            return
        try:
            rra_neg = _parse_rra_output(df_out_neg, n=N).rename(
                columns={"rra_rank": "rra_rank_neg"}
            )
        except Exception as exc:
            print(f"WARNING: Failed to parse RRA negative output: {exc}")
            return
        # Invert negative ranks back to original direction
        rra_neg["rra_rank_neg"] = float(N) - rra_neg["rra_rank_neg"] + 1.0

        merged = index_df.merge(
            rra_pos[["row_idx", "rra_rank_pos", "rra_item_id"]],
            on="row_idx",
            how="left",
        )
        merged = merged.merge(rra_neg[["row_idx", "rra_rank_neg"]], on="row_idx", how="left")
        keep_cols = [
            c
            for c in ["row_idx", "exam_id", "dataset_source", "chagas"]
            if c in merged.columns
        ]
        merged["median_rank_full"] = median_rank_full
        if args.rra_fill_missing:
            merged["rra_rank_pos"] = merged["rra_rank_pos"].fillna(float(N))
            merged["rra_rank_neg"] = merged["rra_rank_neg"].fillna(float(N))
        # Convert filled values back to NaN so only robust hits remain.
        merged.loc[merged["rra_rank_pos"] >= float(N), "rra_rank_pos"] = np.nan
        merged.loc[merged["rra_rank_neg"] >= float(N), "rra_rank_neg"] = np.nan
        # Ensure at most one of rra_rank_pos / rra_rank_neg is set.
        both = merged["rra_rank_pos"].notna() & merged["rra_rank_neg"].notna()
        if both.any():
            pos_rank = merged.loc[both, "rra_rank_pos"].to_numpy(dtype=float)
            neg_raw_rank = float(N) - merged.loc[both, "rra_rank_neg"].to_numpy(dtype=float) + 1.0
            keep_pos = pos_rank <= neg_raw_rank
            drop_pos_idx = merged.loc[both].index[~keep_pos]
            drop_neg_idx = merged.loc[both].index[keep_pos]
            merged.loc[drop_pos_idx, "rra_rank_pos"] = np.nan
            merged.loc[drop_neg_idx, "rra_rank_neg"] = np.nan
        # Only keep rank (score is not meaningful at this scale)
        merged = merged[
            keep_cols
            + ["rra_item_id", "rra_rank_pos", "rra_rank_neg", "median_rank_full"]
        ]
        merged.to_csv(out_base / "sample_rra_consensus.csv", index=False)
        print(f"Wrote {out_base / 'sample_rra_consensus.csv'}")


if __name__ == "__main__":
    main()
