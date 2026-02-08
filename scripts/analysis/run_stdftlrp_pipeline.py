#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from subprocess import run as run_subprocess

from tqdm import tqdm


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _run(cmd: list[str]) -> None:
    run_subprocess(cmd, check=True)


def _resolve_checkpoint(path: str, *, download_dir: Path) -> Path:
    p = Path(path)
    is_wandb_uri = False
    uri = path
    if path.startswith("wandb:"):
        is_wandb_uri = True
        uri = path[len("wandb:") :]
    else:
        parts = path.split("/")
        if len(parts) == 3 and ":" in parts[-1] and not p.exists():
            is_wandb_uri = True

    if is_wandb_uri:
        safe_name = (
            uri.replace("/", "__")
            .replace(":", "__")
            .replace("\\", "__")
        )
        cached_root = download_dir / safe_name
        if cached_root.exists():
            for fname in ("model.ckpt", "best.ckpt"):
                candidate = cached_root / fname
                if candidate.exists():
                    return candidate
            ckpts = sorted(cached_root.rglob("*.ckpt"))
            if ckpts:
                return ckpts[0]

        import wandb  # type: ignore

        api = wandb.Api()
        art = api.artifact(uri, type="model")
        try:
            local_dir = Path(art.download(root=str(cached_root)))
        except TypeError:
            local_dir = Path(art.download())
        for fname in ("model.ckpt", "best.ckpt"):
            candidate = local_dir / fname
            if candidate.exists():
                return candidate
        ckpts = sorted(local_dir.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found in wandb artifact {uri}")
        return ckpts[0]

    if p.is_dir():
        for fname in ("model.ckpt", "best.ckpt"):
            candidate = p / fname
            if candidate.exists():
                return candidate
        ckpts = sorted(p.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found under {p}")
        return ckpts[0]

    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_specs", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--processed_root", type=Path, default=Path("."))
    parser.add_argument("--meta_path", type=Path, default=Path("meta.json"))
    parser.add_argument("--test_fold", type=int, default=4)
    parser.add_argument("--run_id", type=str, default=None, help="Only run a single run_id")
    parser.add_argument("--stdftlrp_exam_ids", choices=["probe", "test"], default="probe")
    parser.add_argument("--stdftlrp_lead_index", type=int, default=1)
    parser.add_argument("--stdftlrp_all_leads", action="store_true")
    parser.add_argument("--stdftlrp_write_per_lead", action="store_true")
    parser.add_argument("--stdftlrp_crop_size", type=int, default=2500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _add_src_to_path()
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs, resolve_data_dir

    global_cfg, runs = load_run_specs(args.run_specs)
    if args.run_id:
        runs = [r for r in runs if r.run_id == args.run_id]
        if not runs:
            raise ValueError(f"--run_id '{args.run_id}' not found in {args.run_specs}")
    processed_root_raw = str(global_cfg.get("processed_root", "")).strip()
    processed_root = Path(processed_root_raw) if processed_root_raw else args.processed_root
    meta_path_raw = str(global_cfg.get("meta_path", "")).strip()
    meta_path = Path(meta_path_raw) if meta_path_raw else args.meta_path

    exam_ids_csv = args.out_dir / (
        "probe_index.csv" if args.stdftlrp_exam_ids == "probe" else "test_index.csv"
    )
    if not exam_ids_csv.exists():
        raise FileNotFoundError(f"Missing exam_ids_csv: {exam_ids_csv}")

    py = sys.executable
    download_dir = args.out_dir / "wandb_downloads"
    runs_iter = tqdm(runs, desc="STDFT-LRP runs", unit="run")
    for run in runs_iter:
        lead_tag = f"lead_{args.stdftlrp_lead_index}" if args.stdftlrp_lead_index is not None else "lead_all"
        out_root = args.out_dir / "runs" / run.run_id / "xai" / lead_tag
        out_path = out_root / "stdftlrp_beat_agg.csv"
        if out_path.exists() and not args.overwrite:
            runs_iter.set_postfix_str(f"skip {run.run_id}")
            continue

        ckpt_path = _resolve_checkpoint(str(run.checkpoint_path), download_dir=download_dir)
        data_dir = resolve_data_dir(run, processed_root=processed_root)
        cmd = [
            py,
            "scripts/analysis/compute_stdftlrp_beat_aggregates.py",
            "--checkpoint",
            str(ckpt_path),
            "--run_id",
            str(run.run_id),
            "--meta_path",
            str(meta_path),
            "--data_dir",
            str(data_dir),
            "--exam_ids_csv",
            str(exam_ids_csv),
            "--out_dir",
            str(args.out_dir / "runs"),
            "--fold",
            str(args.test_fold),
            "--lead_index",
            str(args.stdftlrp_lead_index),
            "--crop_size",
            str(args.stdftlrp_crop_size),
        ]
        if args.stdftlrp_all_leads:
            cmd.append("--all_leads")
        if args.stdftlrp_write_per_lead:
            cmd.append("--write_per_lead")

        _run(cmd)


if __name__ == "__main__":
    main()
