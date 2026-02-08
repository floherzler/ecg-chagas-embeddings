#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
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


_ENC_RE = re.compile(r"__enc__N(?P<N>\d+)__D(?P<D>\d+)\.fp32\.mmap$")
_PROJ_RE = re.compile(r"__proj__N(?P<N>\d+)__D(?P<D>\d+)\.fp32\.mmap$")
_ENC_VIEW1_RE = re.compile(r"__enc_view1__N(?P<N>\d+)__D(?P<D>\d+)\.fp32\.mmap$")


def _find_memmap(path: Path, *, run_id: str, space: str, n_expected: int) -> tuple[Path, int]:
    if space == "enc":
        patt = f"{run_id}__enc__N{n_expected}__D*.fp32.mmap"
        matches = sorted(path.glob(patt))
        if not matches:
            raise FileNotFoundError(f"Missing enc memmap for {run_id}: {patt}")
        m = _ENC_RE.search(matches[0].name)
        if not m:
            raise ValueError(f"Unexpected enc memmap filename: {matches[0].name}")
        return matches[0], int(m.group("D"))
    if space == "proj":
        patt = f"{run_id}__proj__N{n_expected}__D*.fp32.mmap"
        matches = sorted(path.glob(patt))
        if not matches:
            raise FileNotFoundError(f"Missing proj memmap for {run_id}: {patt}")
        m = _PROJ_RE.search(matches[0].name)
        if not m:
            raise ValueError(f"Unexpected proj memmap filename: {matches[0].name}")
        return matches[0], int(m.group("D"))
    raise ValueError(f"Unsupported space: {space}")


def _find_enc_view1_memmap(path: Path, *, run_id: str, n_expected: int) -> tuple[Path, int] | None:
    patt = f"{run_id}__enc_view1__N{n_expected}__D*.fp32.mmap"
    matches = sorted(path.glob(patt))
    if not matches:
        return None
    m = _ENC_VIEW1_RE.search(matches[0].name)
    if not m:
        raise ValueError(f"Unexpected enc_view1 memmap filename: {matches[0].name}")
    return matches[0], int(m.group("D"))


def _saa_from_two_views(
    x0_u: np.ndarray, x1_u: np.ndarray, y: np.ndarray, *, block: int = 256
) -> dict[str, float]:
    """
    Sample Alignment Accuracy (SAA) between two views, probe-only.

    For each sample i, find the nearest neighbor of view0[i] in view1 (cosine similarity on L2-normalized embeddings).
    It's a correct match if argmax_j sim(view0[i], view1[j]) == i.
    Compute the same in the other direction and average.

    Returns SAA_0 and SAA_1 as percentages (0..100) for class 0/1.
    """
    x0_u = np.asarray(x0_u, dtype=np.float32)
    x1_u = np.asarray(x1_u, dtype=np.float32)
    y = np.asarray(y, dtype=int).reshape(-1)
    if x0_u.shape != x1_u.shape:
        raise ValueError(f"Expected x0_u and x1_u same shape, got {x0_u.shape} vs {x1_u.shape}")
    N = int(x0_u.shape[0])
    if N != int(y.size):
        raise ValueError(f"Expected y size N={N}, got {y.size}")
    if N < 1:
        return {"SAA_0": float("nan"), "SAA_1": float("nan")}

    def _nn_indices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        nn = np.empty((N,), dtype=np.int32)
        BT = B.T
        for i0 in range(0, N, int(block)):
            i1 = min(N, i0 + int(block))
            sim = A[i0:i1] @ BT  # [b, N]
            nn[i0:i1] = np.argmax(sim, axis=1).astype(np.int32)
        return nn

    nn01 = _nn_indices(x0_u, x1_u)
    nn10 = _nn_indices(x1_u, x0_u)
    correct01 = nn01 == np.arange(N, dtype=np.int32)
    correct10 = nn10 == np.arange(N, dtype=np.int32)
    correct = 0.5 * (correct01.astype(np.float32) + correct10.astype(np.float32))

    out: dict[str, float] = {}
    for cls in (0, 1):
        m = y == cls
        if not np.any(m):
            out[f"SAA_{cls}"] = float("nan")
        else:
            out[f"SAA_{cls}"] = float(correct[m].mean() * 100.0)
    return out


def _effective_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    # Singular values are sqrt of eigenvalues (up to scaling) for covariance.
    # Use economy SVD.
    try:
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    lam = (s**2) / float(max(1, x.shape[0]))
    num = float(lam.sum()) ** 2
    den = float((lam**2).sum())
    if den <= 0:
        return float("nan")
    return float(num / den)


def _compute_ttc_metrics(x_u: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from ecg_chagas_embeddings.metrics.ttc_metrics import (
        calculate_class_alignment_consistency,
        calculate_class_alignment_distance,
        calculate_gaussian_potential_uniformity,
    )

    x_u = np.asarray(x_u, dtype=np.float32)
    y = np.asarray(y, dtype=int).reshape(-1)
    if x_u.ndim != 2:
        raise ValueError(f"Expected x_u [N,D], got {tuple(x_u.shape)}")

    dot = np.clip(x_u @ x_u.T, -1.0, 1.0)
    dist_sq = np.clip(2.0 - 2.0 * dot, 0.0, None)
    dist = np.sqrt(dist_sq, out=dist_sq)  # reuse buffer

    cad0, cad1 = calculate_class_alignment_distance(dist, x_u, y)
    cac0, cac1 = calculate_class_alignment_consistency(dist, x_u, y)
    gpu0, gpu1 = calculate_gaussian_potential_uniformity(x_u, y)

    return {
        "CAD_0": float(np.mean(cad0)) if getattr(cad0, "size", 0) else float("nan"),
        "CAD_1": float(np.mean(cad1)) if getattr(cad1, "size", 0) else float("nan"),
        "CAC_0": float(np.mean(cac0)) if getattr(cac0, "size", 0) else float("nan"),
        "CAC_1": float(np.mean(cac1)) if getattr(cac1, "size", 0) else float("nan"),
        "GPU_0": float(gpu0),
        "GPU_1": float(gpu1),
    }


def main() -> None:
    _add_src_to_path()

    from sklearn.decomposition import PCA

    from ecg_chagas_embeddings.analysis.embeddings_probe import (
        DEFAULT_OUTPUT_DIR,
        legacy_memmap_dir,
        l2_normalize_np,
        run_memmap_dir,
    )
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs

    parser = argparse.ArgumentParser(
        description="Compute TTC embedding metrics + collapse diagnostics from persisted probe memmaps."
    )
    parser.add_argument(
        "--run_specs",
        type=Path,
        default=Path("configs/analysis/embeddings_probe_runs.toml"),
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--probe_index",
        type=Path,
        default=None,
        help="Path to probe_index.csv (defaults to <out_dir>/probe_index.csv).",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 = use all samples")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--compute_saa",
        action="store_true",
        help="Compute probe-only SAA (requires `__enc_view1__...` memmaps; encoder space only).",
    )
    parser.add_argument(
        "--group_metrics",
        action="store_true",
        help="Also compute probe-only metrics on dataset splits: verified=(PTBXL+SAMITROP) vs CODE15.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    legacy_memmaps = legacy_memmap_dir(out_dir)
    probe_index_path = args.probe_index or (out_dir / "probe_index.csv")
    probe_index = pd.read_csv(probe_index_path).sort_values("row_idx").reset_index(drop=True)
    if "chagas" not in probe_index.columns and "y_true" in probe_index.columns:
        probe_index = probe_index.rename(columns={"y_true": "chagas"})
    if "chagas" not in probe_index.columns:
        raise ValueError(f"{probe_index_path} must include label column 'chagas'")
    if "dataset_source" not in probe_index.columns:
        raise ValueError(f"{probe_index_path} must include column 'dataset_source' for group splits")
    y = probe_index["chagas"].to_numpy(dtype=int)
    ds = probe_index["dataset_source"].astype(str).to_numpy(dtype=object)
    N = int(len(y))

    global_cfg, runs = load_run_specs(args.run_specs)
    _ = global_cfg

    out_path = out_dir / "embedding_metrics.csv"
    done: set[tuple[str, str]] = set()
    if out_path.exists() and not args.overwrite:
        try:
            df_done = pd.read_csv(out_path, usecols=["run_id", "space"])
            done = set(
                zip(df_done["run_id"].astype(str).tolist(), df_done["space"].astype(str).tolist())
            )
        except Exception:
            done = set()

    rng = np.random.default_rng(int(args.seed))
    if args.max_samples and int(args.max_samples) > 0 and N > int(args.max_samples):
        idx = rng.choice(N, size=int(args.max_samples), replace=False)
        idx = np.sort(idx)
    else:
        idx = None

    rows: list[dict[str, Any]] = []
    for run in tqdm(runs, desc="Runs", unit="run"):
        memmap_dir = run_memmap_dir(out_dir, run.run_id)
        for space in tqdm(("enc", "proj"), desc="Spaces", unit="space", leave=False):
            if (run.run_id, space) in done:
                continue
            try:
                mmap_path, D = _find_memmap(memmap_dir, run_id=run.run_id, space=space, n_expected=N)
            except FileNotFoundError:
                # Backwards compatibility: read from legacy <out_dir>/memmap if present.
                try:
                    mmap_path, D = _find_memmap(
                        legacy_memmaps, run_id=run.run_id, space=space, n_expected=N
                    )
                except FileNotFoundError:
                    if space == "proj":
                        continue
                    raise

            x = np.memmap(mmap_path, mode="r", dtype="float32", shape=(N, D))
            if idx is not None:
                x_use = np.asarray(x[idx], dtype=np.float32)
                y_use = y[idx]
            else:
                x_use = np.asarray(x, dtype=np.float32)
                y_use = y

            norms = np.linalg.norm(x_use, axis=1)
            x_u = l2_normalize_np(x_use, axis=1, eps=1e-12)

            # TTC metrics on normalized tensors.
            ttc = _compute_ttc_metrics(x_u, y_use)

            # Collapse diagnostics (raw space).
            pca_k = min(10, x_use.shape[1], x_use.shape[0])
            pca = PCA(n_components=pca_k, svd_solver="randomized", random_state=int(args.seed))
            pca.fit(x_use)
            evr = pca.explained_variance_ratio_
            diag = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "cov_trace": float(np.var(x_use, axis=0).sum()),
                "eff_rank": float(_effective_rank(x_use)),
                "pca_var_top1": float(evr[0]) if evr.size >= 1 else float("nan"),
                "pca_var_top2": float(evr[:2].sum()) if evr.size >= 2 else float("nan"),
                "pca_var_top5": float(evr[:5].sum()) if evr.size >= 5 else float("nan"),
                "pca_var_top10": float(evr[:10].sum()) if evr.size >= 10 else float("nan"),
            }

            row: dict[str, Any] = {
                "run_id": run.run_id,
                "track": run.track,
                "preprocessing": run.preprocessing,
                "space": space,
                "N": int(x_use.shape[0]),
                "D": int(x_use.shape[1]),
            }
            row.update({k: v for k, v in run.meta.items()})
            row.update(ttc)
            row.update(diag)

            # Optional: load view1 embeddings once (used for SAA and/or group SAA).
            x1_u: np.ndarray | None = None
            if (bool(args.compute_saa) or bool(args.group_metrics)) and space == "enc" and idx is None:
                enc1 = _find_enc_view1_memmap(memmap_dir, run_id=run.run_id, n_expected=N)
                if enc1 is None:
                    enc1 = _find_enc_view1_memmap(legacy_memmaps, run_id=run.run_id, n_expected=N)
                if enc1 is not None:
                    mmap_path1, D1 = enc1
                    if int(D1) == int(D):
                        x1 = np.memmap(mmap_path1, mode="r", dtype="float32", shape=(N, D1))
                        x1_u = l2_normalize_np(np.asarray(x1, dtype=np.float32), axis=1, eps=1e-12)

            # SAA on probe only, encoder space only.
            if bool(args.compute_saa) and space == "enc" and idx is None and x1_u is not None:
                row.update(_saa_from_two_views(x_u, x1_u, y_use))

            if bool(args.group_metrics) and space == "enc" and idx is None:
                # Dataset splits on the probe set:
                # - verified: PTBXL + SAMITROP
                # - self-reported/mixed: CODE15
                is_verified = np.isin(ds, np.array(["PTBXL", "SAMITROP"], dtype=object))
                is_code15 = ds == "CODE15"

                def _add_group(prefix: str, m: np.ndarray) -> None:
                    m = np.asarray(m, dtype=bool)
                    if not np.any(m):
                        row[f"N_{prefix}"] = 0
                        for key in ("CAC_0", "CAC_1", "GPU_0", "GPU_1"):
                            row[f"{key}_{prefix}"] = float("nan")
                        if x1_u is not None:
                            row[f"SAA_0_{prefix}"] = float("nan")
                            row[f"SAA_1_{prefix}"] = float("nan")
                        return
                    row[f"N_{prefix}"] = int(m.sum())
                    ttc_g = _compute_ttc_metrics(x_u[m], y_use[m])
                    for key in ("CAC_0", "CAC_1", "GPU_0", "GPU_1"):
                        row[f"{key}_{prefix}"] = float(ttc_g.get(key, float("nan")))
                    if x1_u is not None:
                        saa_g = _saa_from_two_views(x_u[m], x1_u[m], y_use[m])
                        row[f"SAA_0_{prefix}"] = float(saa_g.get("SAA_0", float("nan")))
                        row[f"SAA_1_{prefix}"] = float(saa_g.get("SAA_1", float("nan")))

                _add_group("verified", is_verified)
                _add_group("code15", is_code15)

            rows.append(row)
            print(f"{run.run_id} {space}: CAC_1={row.get('CAC_1', float('nan')):.3f}")

    df_new = pd.DataFrame(rows)
    if out_path.exists() and not args.overwrite:
        df_old = pd.read_csv(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        if {"run_id", "space"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["run_id", "space"], keep="last")
    else:
        df = df_new
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
