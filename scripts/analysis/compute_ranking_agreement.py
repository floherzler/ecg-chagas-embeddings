#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def _load_logits(*, out_dir: Path, run_ids: list[str], n: int, skip_missing: bool) -> tuple[list[str], np.ndarray]:
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


def _kendall_topk_matrix(scores: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    _write_matrix_csv(out_base / "top5_intersection_n.csv", kept_ids, n_int.astype(np.float32))
    print(f"Wrote {out_base / 'top5_kendall_tau.csv'}")
    print(f"Wrote {out_base / 'top5_intersection_n.csv'}")

    # (4) Per-sample consensus for top-k (Step 2 in your screenshot)
    consensus_count = mask.sum(axis=0).astype(np.int32)  # [N]
    consensus_frac = (consensus_count.astype(np.float32) / float(M)).astype(np.float32)
    keep_cols = [c for c in ["row_idx", "exam_id", "dataset_source", "chagas"] if c in index_df.columns]
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


if __name__ == "__main__":
    main()
