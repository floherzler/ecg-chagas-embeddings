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


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _as_binary(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float).to_numpy(dtype=float)
    s_num = pd.to_numeric(series, errors="coerce")
    out = np.full(len(series), np.nan, dtype=float)
    if s_num.notna().any():
        v = s_num.to_numpy(dtype=float)
        out[v == 0] = 0.0
        out[v == 1] = 1.0
        return out
    s_str = series.astype("string[python]").str.strip().str.lower()
    out[s_str.isin(["1", "true", "t", "yes", "y"])] = 1.0
    out[s_str.isin(["0", "false", "f", "no", "n"])] = 0.0
    return out


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    if float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_pca_coords(*, out_dir: Path, run_id: str, space: str) -> pd.DataFrame:
    # New layout: per-run coords.
    p = out_dir / "runs" / run_id / "coords" / f"{run_id}__{space}__pca.csv"
    if not p.exists():
        # Legacy fallback.
        p = out_dir / "coords" / f"{run_id}__{space}__pca.csv"
    return pd.read_csv(p, usecols=["row_idx", "x", "y"])


def _upsert_into_scores(*, scores_path: Path, corr_df: pd.DataFrame) -> None:
    if not scores_path.exists():
        return
    scores = pd.read_csv(scores_path)
    # Upsert correlation columns into the scores table without failing on overlaps.
    keep_cols = [c for c in corr_df.columns if c not in {"space"}]
    corr_df = corr_df[keep_cols].copy()

    merged = scores.merge(corr_df, on="run_id", how="left", suffixes=("", "__new"))
    for col in corr_df.columns:
        if col == "run_id":
            continue
        new_col = f"{col}__new"
        if new_col not in merged.columns:
            continue
        merged[col] = merged[new_col]
        merged = merged.drop(columns=[new_col])

    # Drop any leftover __new columns (defensive).
    leftovers = [c for c in merged.columns if c.endswith("__new")]
    if leftovers:
        merged = merged.drop(columns=leftovers)

    merged.to_csv(scores_path, index=False)


def main() -> None:
    _add_src_to_path()

    from ecg_chagas_embeddings.analysis.embeddings_probe import DEFAULT_OUTPUT_DIR
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs

    parser = argparse.ArgumentParser(
        description="Compute correlations of metadata variables with PCA1/PCA2 on the probe set."
    )
    parser.add_argument(
        "--run_specs",
        type=Path,
        default=Path("configs/analysis/embeddings_probe_runs.toml"),
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--space",
        type=str,
        default="enc",
        choices=["enc", "proj"],
        help="Which embedding space PCA coords were computed for.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite pca_correlations.csv (otherwise upserts per run_id).",
    )
    parser.add_argument(
        "--write_into_test_scores",
        action="store_true",
        help="Also write correlation columns into <out_dir>/test_scores.csv.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    meta_path = out_dir / "probe_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = pd.read_csv(meta_path)
    if "chagas" not in meta.columns and "y_true" in meta.columns:
        meta = meta.rename(columns={"y_true": "chagas"})

    _global_cfg, runs = load_run_specs(args.run_specs)

    rows: list[dict[str, Any]] = []
    for run in tqdm(runs, desc="Runs", unit="run"):
        coords = _load_pca_coords(out_dir=out_dir, run_id=run.run_id, space=args.space)
        df = coords.merge(meta, on="row_idx", how="left")

        pc1 = _as_float(df["x"])
        pc2 = _as_float(df["y"])

        out: dict[str, Any] = {"run_id": run.run_id, "space": args.space}

        if "chagas" in df.columns:
            chagas = _as_binary(df["chagas"])
            out["corr_chagas_pc1"] = _pearsonr(chagas, pc1)
            out["corr_chagas_pc2"] = _pearsonr(chagas, pc2)
            out["n_chagas"] = int(np.isfinite(chagas).sum())

        if "age" in df.columns:
            age = _as_float(df["age"])
            out["corr_age_pc1"] = _pearsonr(age, pc1)
            out["corr_age_pc2"] = _pearsonr(age, pc2)
            out["n_age"] = int(np.isfinite(age).sum())

        # RBBB correlation on CODE15 only.
        if "RBBB" in df.columns and "dataset_source" in df.columns:
            is_code15 = df["dataset_source"].astype(str).eq("CODE15").to_numpy()
            rbbb = _as_binary(df["RBBB"])
            rbbb_c = np.where(is_code15, rbbb, np.nan)
            out["corr_rbbb_code15_pc1"] = _pearsonr(rbbb_c, pc1)
            out["corr_rbbb_code15_pc2"] = _pearsonr(rbbb_c, pc2)
            out["n_rbbb_code15"] = int(np.isfinite(rbbb_c).sum())

        rows.append(out)

    out_path = out_dir / "pca_correlations.csv"
    df_new = pd.DataFrame(rows)
    if out_path.exists() and not args.overwrite:
        df_old = pd.read_csv(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["run_id", "space"], keep="last")
    else:
        df = df_new
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    if args.write_into_test_scores:
        _upsert_into_scores(scores_path=out_dir / "test_scores.csv", corr_df=df_new)
        print(f"Updated {out_dir / 'test_scores.csv'}")


if __name__ == "__main__":
    main()
