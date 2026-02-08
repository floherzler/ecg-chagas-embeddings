#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

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


def main() -> None:
    _add_src_to_path()

    from sklearn.decomposition import PCA

    from ecg_chagas_embeddings.analysis.embeddings_probe import (
        DEFAULT_OUTPUT_DIR,
        legacy_coords_dir,
        legacy_memmap_dir,
        l2_normalize_np,
        run_coords_dir,
        run_memmap_dir,
    )
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs

    import umap  # type: ignore

    parser = argparse.ArgumentParser(
        description="Compute PCA/UMAP 2D projections from persisted probe embeddings."
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
    parser.add_argument("--spaces", nargs="+", default=["enc", "proj"], choices=["enc", "proj"])
    parser.add_argument("--normalize", action="store_true", help="L2-normalize before projection.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    legacy_memmaps = legacy_memmap_dir(out_dir)
    legacy_coords = legacy_coords_dir(out_dir)
    legacy_coords.mkdir(parents=True, exist_ok=True)

    probe_index_path = args.probe_index or (out_dir / "probe_index.csv")
    probe_index = pd.read_csv(probe_index_path).sort_values("row_idx").reset_index(drop=True)
    row_idx = probe_index["row_idx"].to_numpy(dtype=int)
    N = int(len(row_idx))

    _global_cfg, runs = load_run_specs(args.run_specs)

    for run in tqdm(runs, desc="Runs", unit="run"):
        memmap_dir = run_memmap_dir(out_dir, run.run_id)
        coords_dir = run_coords_dir(out_dir, run.run_id)
        coords_dir.mkdir(parents=True, exist_ok=True)
        for space in tqdm(args.spaces, desc="Spaces", unit="space", leave=False):
            try:
                mmap_path, D = _find_memmap(
                    memmap_dir, run_id=run.run_id, space=space, n_expected=N
                )
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
            x_use = np.asarray(x, dtype=np.float32)
            if args.normalize:
                x_use = l2_normalize_np(x_use, axis=1, eps=1e-12)

            # PCA
            pca_out = coords_dir / f"{run.run_id}__{space}__pca.csv"
            if args.overwrite or not pca_out.exists():
                pca = PCA(n_components=2, svd_solver="randomized", random_state=int(args.seed))
                xy = pca.fit_transform(x_use)
                pd.DataFrame({"row_idx": row_idx, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(
                    pca_out, index=False
                )
                print(f"Wrote {pca_out}")

            # UMAP
            umap_out = coords_dir / f"{run.run_id}__{space}__umap.csv"
            if args.overwrite or not umap_out.exists():
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=int(args.umap_neighbors),
                    min_dist=float(args.umap_min_dist),
                    metric="euclidean",
                    random_state=int(args.seed),
                )
                xy = reducer.fit_transform(x_use)
                pd.DataFrame({"row_idx": row_idx, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(
                    umap_out, index=False
                )
                print(f"Wrote {umap_out}")


if __name__ == "__main__":
    main()
