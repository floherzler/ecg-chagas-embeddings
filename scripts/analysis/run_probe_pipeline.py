#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


def _read_run_ids_from_specs(path: Path) -> list[str]:
    import tomllib

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    runs = raw.get("runs", [])
    if not isinstance(runs, list):
        raise TypeError(f"'runs' must be a list in {path}")
    out: list[str] = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        rid = str(entry.get("run_id", "")).strip()
        if rid:
            out.append(rid)
    return out


def _write_single_run_specs(
    *,
    path: Path,
    processed_root: str,
    meta_path: str,
    run_id: str,
    track: str,
    preprocessing: str,
    checkpoint_path: str,
    has_projection: bool | None,
    extra_meta: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append(f'processed_root = "{processed_root}"')
    lines.append(f'meta_path = "{meta_path}"')
    lines.append("")
    lines.append("[[runs]]")
    lines.append(f'run_id = "{run_id}"')
    lines.append(f'track = "{track}"')
    lines.append(f'preprocessing = "{preprocessing}"')
    lines.append(f'checkpoint_path = "{checkpoint_path}"')
    if has_projection is not None:
        lines.append(f"has_projection = {'true' if has_projection else 'false'}")
    for k, v in extra_meta.items():
        key = str(k).strip()
        if not key:
            continue
        # Minimal TOML scalar support (string/bool/int/float), fall back to string.
        if isinstance(v, bool):
            val = "true" if v else "false"
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            val = str(v)
        else:
            val = f'"{str(v)}"'
        lines.append(f"{key} = {val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def _has_probe_files(out_dir: Path) -> bool:
    return (
        (out_dir / "probe_index.csv").exists()
        and (out_dir / "probe_metadata.csv").exists()
        and (out_dir / "test_index.csv").exists()
    )


def _has_all_scores(out_dir: Path, run_ids: list[str]) -> bool:
    p = out_dir / "test_scores.csv"
    if not p.exists():
        return False
    try:
        import pandas as pd

        # Require the new columns as well (prevents "skipping" after adding new metrics).
        # We only check for column presence (not non-null), because some splits may
        # legitimately produce NaNs (e.g. AUROC on all-negative PTB-XL).
        df_head = pd.read_csv(p, nrows=1)
        required_cols = {
            "run_id",
            # Global
            "auroc",
            "ap",
            "pauc_fpr0.05",
            "tpr_top0.05",
            "tpr_top0.10",
            # Verified vs self-reported/mixed splits
            "N_verified",
            "N_code15",
            "auroc_verified",
            "ap_verified",
            "pauc_fpr0.05_verified",
            "tpr_top0.05_verified",
            "tpr_top0.10_verified",
            "auroc_code15",
            "ap_code15",
            "pauc_fpr0.05_code15",
            "tpr_top0.05_code15",
            "tpr_top0.10_code15",
        }
        if not required_cols.issubset(set(df_head.columns)):
            return False

        df = pd.read_csv(p, usecols=["run_id"])
        have = set(df["run_id"].astype(str).tolist())
        return all(rid in have for rid in run_ids)
    except Exception:
        return False


def _has_all_test_logits(out_dir: Path, run_ids: list[str]) -> bool:
    test_index = out_dir / "test_index.csv"
    if not test_index.exists():
        return False
    try:
        import pandas as pd

        N = int(len(pd.read_csv(test_index, usecols=["row_idx"])))
    except Exception:
        return False

    for rid in run_ids:
        memmap_dir = out_dir / "runs" / rid / "memmap"
        legacy_dir = out_dir / "memmap"
        p = memmap_dir / f"{rid}__logits__N{N}.fp32.mmap"
        if p.exists():
            continue
        p = legacy_dir / f"{rid}__logits__N{N}.fp32.mmap"
        if not p.exists():
            return False
    return True


def _has_all_memmaps(out_dir: Path, run_ids: list[str]) -> bool:
    probe_index = out_dir / "probe_index.csv"
    if not probe_index.exists():
        return False
    try:
        import pandas as pd

        N = int(len(pd.read_csv(probe_index, usecols=["row_idx"])))
    except Exception:
        return False

    for rid in run_ids:
        # New layout: <out_dir>/runs/<run_id>/memmap/
        memmap_dir = out_dir / "runs" / rid / "memmap"
        legacy_dir = out_dir / "memmap"
        enc = list(memmap_dir.glob(f"{rid}__enc__N{N}__D*.fp32.mmap"))
        enc1 = list(memmap_dir.glob(f"{rid}__enc_view1__N{N}__D*.fp32.mmap"))
        logits = memmap_dir / f"{rid}__logits__N{N}.fp32.mmap"
        if enc and enc1 and logits.exists():
            continue
        # Backwards compatibility: legacy <out_dir>/memmap/
        enc_legacy = list(legacy_dir.glob(f"{rid}__enc__N{N}__D*.fp32.mmap"))
        enc1_legacy = list(legacy_dir.glob(f"{rid}__enc_view1__N{N}__D*.fp32.mmap"))
        logits_legacy = legacy_dir / f"{rid}__logits__N{N}.fp32.mmap"
        if not enc_legacy or not enc1_legacy or not logits_legacy.exists():
            return False
    return True


def _has_all_embedding_metrics(out_dir: Path, run_ids: list[str]) -> bool:
    p = out_dir / "embedding_metrics.csv"
    if not p.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(p)
        if "SAA_0" not in df.columns or "SAA_1" not in df.columns:
            return False
        if "CAC_1_verified" not in df.columns or "SAA_1_verified" not in df.columns:
            return False
        df = df[["run_id", "space"]]
        have = set(zip(df["run_id"].astype(str), df["space"].astype(str)))
        return all((rid, "enc") in have for rid in run_ids)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full embeddings-probe pipeline end-to-end (build probe, eval, memmaps, metrics, projections)."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("analysis/embeddings_probe"),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run_specs",
        type=Path,
        help="Existing TOML run registry (runs will be processed in order).",
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Single checkpoint path or W&B artifact URI; a temporary run_specs file is created.",
    )

    # Only used with --checkpoint
    parser.add_argument("--run_id", type=str, default="single-run")
    parser.add_argument("--track", type=str, choices=["t1", "t2", "t3"], default="t1")
    parser.add_argument("--preprocessing", type=str, default="bp")
    parser.add_argument(
        "--processed_root",
        type=str,
        default="/home/flo178/projects/master-thesis/datasets/physionet2025/processedMaster",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="/home/flo178/projects/master-thesis/datasets/physionet2025/processedMaster/metadata.csv",
    )
    parser.add_argument(
        "--has_projection",
        type=str,
        default="auto",
        help="auto|true|false (only used with --checkpoint)",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        help="Extra per-run metadata as key=value (repeatable).",
    )

    # Probe sampling knobs
    parser.add_argument("--test_fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--neg_multiplier", type=int, default=2)
    parser.add_argument("--neg_frac_code15", type=float, default=0.5)
    parser.add_argument("--neg_frac_ptbxl", type=float, default=0.5)
    parser.add_argument("--neg_frac_samitrop", type=float, default=0.0)

    # Common runtime knobs
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=2500)
    parser.add_argument("--augmentation_base_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    # Plotting (hex-tiling multipanels only)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate hex-tiling multipanel figures (main/conduction/outcome + small multiples) after projections.",
    )
    parser.add_argument(
        "--stdftlrp",
        action="store_true",
        help="Compute STDFT-LRP beat-level aggregates per run over the selected index set.",
    )
    parser.add_argument(
        "--stdftlrp_exam_ids",
        choices=["probe", "test"],
        default="probe",
        help="Use probe_index.csv or test_index.csv as the sample list for STDFT-LRP aggregation.",
    )
    parser.add_argument("--stdftlrp_lead_index", type=int, default=1)
    parser.add_argument("--stdftlrp_all_leads", action="store_true")
    parser.add_argument("--stdftlrp_write_per_lead", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_specs:
        run_specs = args.run_specs
    else:
        meta_kv: dict[str, Any] = {}
        for item in args.meta:
            if "=" not in item:
                raise ValueError(f"--meta must be key=value, got {item!r}")
            k, v = item.split("=", 1)
            meta_kv[k.strip()] = v.strip()

        if args.has_projection not in ("auto", "true", "false"):
            raise ValueError("--has_projection must be auto|true|false")
        has_proj: bool | None
        if args.has_projection == "auto":
            has_proj = None
        else:
            has_proj = args.has_projection == "true"

        run_specs = out_dir / "run_specs.single.toml"
        _write_single_run_specs(
            path=run_specs,
            processed_root=args.processed_root,
            meta_path=args.meta_path,
            run_id=args.run_id,
            track=args.track,
            preprocessing=args.preprocessing,
            checkpoint_path=args.checkpoint,
            has_projection=has_proj,
            extra_meta=meta_kv,
        )

    py = sys.executable
    overwrite_flag = ["--overwrite"] if args.overwrite else []
    run_ids = _read_run_ids_from_specs(run_specs)

    if args.overwrite or not _has_probe_files(out_dir):
        _run(
            [
                py,
                "scripts/analysis/build_probe_set.py",
                "--out_dir",
                str(out_dir),
                "--test_fold",
                str(args.test_fold),
                "--seed",
                str(args.seed),
                "--neg_multiplier",
                str(args.neg_multiplier),
                "--neg_frac_code15",
                str(args.neg_frac_code15),
                "--neg_frac_ptbxl",
                str(args.neg_frac_ptbxl),
                "--neg_frac_samitrop",
                str(args.neg_frac_samitrop),
            ]
        )
    else:
        print("Skipping build_probe_set: probe files already exist.")

    if args.overwrite or not _has_all_scores(out_dir, run_ids) or not _has_all_test_logits(out_dir, run_ids):
        _run(
            [
                py,
                "scripts/analysis/evaluate_test_models.py",
                "--run_specs",
                str(run_specs),
                "--out_dir",
                str(out_dir),
                "--test_fold",
                str(args.test_fold),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--device",
                str(args.device),
                "--crop_size",
                str(args.crop_size),
                "--augmentation_base_seed",
                str(args.augmentation_base_seed),
                "--save_logits",
                *overwrite_flag,
            ]
        )
    else:
        print("Skipping evaluate_test_models: scores already exist for all runs.")

    if args.overwrite or not _has_all_memmaps(out_dir, run_ids):
        _run(
            [
                py,
                "scripts/analysis/extract_probe_embeddings.py",
                "--run_specs",
                str(run_specs),
                "--out_dir",
                str(out_dir),
                "--test_fold",
                str(args.test_fold),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--device",
                str(args.device),
                "--crop_size",
                str(args.crop_size),
                "--augmentation_base_seed",
                str(args.augmentation_base_seed),
                *overwrite_flag,
            ]
        )
    else:
        print("Skipping extract_probe_embeddings: memmaps already exist for all runs.")

    if args.overwrite or not _has_all_embedding_metrics(out_dir, run_ids):
        _run(
            [
                py,
                "scripts/analysis/compute_embedding_metrics.py",
                "--run_specs",
                str(run_specs),
                "--out_dir",
                str(out_dir),
                "--compute_saa",
                "--group_metrics",
                *overwrite_flag,
            ]
        )
    else:
        print("Skipping compute_embedding_metrics: metrics already exist for all runs.")

    _run(
        [
            py,
            "scripts/analysis/compute_ranking_agreement.py",
            "--run_specs",
            str(run_specs),
            "--out_dir",
            str(out_dir),
            "--set",
            "test",
            "--rra",
            "--rra_subprocess",
        ]
    )

    _run(
        [
            py,
            "scripts/analysis/compute_projections.py",
            "--run_specs",
            str(run_specs),
            "--out_dir",
            str(out_dir),
            "--normalize",
            *overwrite_flag,
        ]
    )

    _run(
        [
            py,
            "scripts/analysis/compute_pca_correlations.py",
            "--run_specs",
            str(run_specs),
            "--out_dir",
            str(out_dir),
            "--space",
            "enc",
            "--write_into_test_scores",
        ]
    )

    if args.plots:
        _run(
            [
                py,
                "scripts/analysis/plot_probe_hex_panels.py",
                "--run_specs",
                str(run_specs),
                "--out_dir",
                str(out_dir),
            ]
        )

    if args.stdftlrp:
        _run(
            [
                py,
                "scripts/analysis/run_stdftlrp_pipeline.py",
                "--run_specs",
                str(run_specs),
                "--out_dir",
                str(out_dir),
                "--processed_root",
                str(args.processed_root),
                "--meta_path",
                str(args.meta_path),
                "--test_fold",
                str(args.test_fold),
                "--stdftlrp_exam_ids",
                str(args.stdftlrp_exam_ids),
                "--stdftlrp_lead_index",
                str(args.stdftlrp_lead_index),
                "--stdftlrp_crop_size",
                str(args.crop_size),
                *(
                    ["--stdftlrp_all_leads"]
                    if args.stdftlrp_all_leads
                    else []
                ),
                *(
                    ["--stdftlrp_write_per_lead"]
                    if args.stdftlrp_write_per_lead
                    else []
                ),
                *(
                    ["--overwrite"]
                    if args.overwrite
                    else []
                ),
            ]
        )


if __name__ == "__main__":
    main()
