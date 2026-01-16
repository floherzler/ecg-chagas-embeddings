#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def main() -> None:
    _add_src_to_path()

    from ecg_chagas_embeddings.analysis.embeddings_probe import (
        DEFAULT_CODE15_EXAMS_PATH,
        DEFAULT_MASTER_META_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PTBXL_DB_PATH,
        DEFAULT_SAMITROP_EXAMS_PATH,
        build_probe_index,
        build_probe_metadata,
        build_stratification_frame,
        ensure_dir,
        load_code15_exams,
        load_master_quality,
        load_ptbxl_database,
        load_samitrop_exams,
        load_test_master_table,
    )

    parser = argparse.ArgumentParser(
        description="Build a fixed fold4 probe subset + probe metadata CSV."
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--meta_path", type=Path, default=DEFAULT_MASTER_META_PATH)
    parser.add_argument("--test_fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--neg_multiplier", type=int, default=2)
    parser.add_argument(
        "--neg_frac_code15",
        type=float,
        default=0.5,
        help="Fraction of sampled negatives from CODE15 (renormalized across available sources).",
    )
    parser.add_argument(
        "--neg_frac_ptbxl",
        type=float,
        default=0.5,
        help="Fraction of sampled negatives from PTBXL (renormalized across available sources).",
    )
    parser.add_argument(
        "--neg_frac_samitrop",
        type=float,
        default=0.0,
        help="Fraction of sampled negatives from SAMITROP (often 0 because fold4 has no SaMi-Trop negatives).",
    )
    parser.add_argument("--code15_exams", type=Path, default=DEFAULT_CODE15_EXAMS_PATH)
    parser.add_argument("--samitrop_exams", type=Path, default=DEFAULT_SAMITROP_EXAMS_PATH)
    parser.add_argument("--ptbxl_db", type=Path, default=DEFAULT_PTBXL_DB_PATH)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    df_test = load_test_master_table(meta_path=args.meta_path, test_fold=args.test_fold)
    master_quality = load_master_quality(meta_path=args.meta_path)
    code15 = load_code15_exams(args.code15_exams)
    samitrop = load_samitrop_exams(args.samitrop_exams)
    ptbxl = load_ptbxl_database(args.ptbxl_db)

    df_strat = build_stratification_frame(
        df_test,
        code15=code15,
        samitrop=samitrop,
        ptbxl=ptbxl,
    )

    probe_index = build_probe_index(
        df_strat,
        seed=args.seed,
        neg_multiplier=args.neg_multiplier,
        neg_source_fracs={
            "CODE15": float(args.neg_frac_code15),
            "PTBXL": float(args.neg_frac_ptbxl),
            "SAMITROP": float(args.neg_frac_samitrop),
        },
    )
    probe_index_path = out_dir / "probe_index.csv"
    probe_index.to_csv(probe_index_path, index=False)

    probe_meta = build_probe_metadata(
        probe_index,
        code15=code15,
        samitrop=samitrop,
        ptbxl=ptbxl,
        master_quality=master_quality,
    )
    probe_meta_path = out_dir / "probe_metadata.csv"
    probe_meta.to_csv(probe_meta_path, index=False)

    n_pos = int((probe_index["chagas"] == 1).sum())
    n_neg = int((probe_index["chagas"] == 0).sum())
    print(f"Wrote {probe_index_path} (N={len(probe_index)}; pos={n_pos}; neg={n_neg})")
    print(f"Wrote {probe_meta_path} (rows={len(probe_meta)})")


if __name__ == "__main__":
    main()
