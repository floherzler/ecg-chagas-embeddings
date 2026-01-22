#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _resolve_checkpoint(path: str, *, download_dir: Path) -> Path:
    p = Path(path)
    is_wandb_uri = False
    uri = path
    if path.startswith("wandb:"):
        is_wandb_uri = True
        uri = path[len("wandb:") :]
    else:
        # Accept bare wandb artifact URIs like:
        #   "entity/project/model-xxxx:v12"
        # to avoid forcing a "wandb:" prefix in config files.
        parts = path.split("/")
        if len(parts) == 3 and ":" in parts[-1] and not p.exists():
            is_wandb_uri = True

    if is_wandb_uri:
        import wandb  # type: ignore

        api = wandb.Api()
        art = api.artifact(uri, type="model")
        try:
            local_dir = Path(art.download(root=str(download_dir)))
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


def _append_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        if "run_id" in df.columns:
            df = df.drop_duplicates(subset=["run_id"], keep="last")
    else:
        df = df_new
    df.to_csv(path, index=False)

def _track3_overrides(run_meta: dict[str, Any]) -> dict[str, Any]:
    """
    Track-3 (linear probe) checkpoints may require a `pretrained_encoder_path` to be present in hparams.

    For offline evaluation we normally don't call `setup()`, so encoder weights should already be in the
    checkpoint state_dict. Still, providing the path as an override makes track3 checkpoints robust and
    allows future code paths that call `setup()` to work.
    """
    out: dict[str, Any] = {}
    if not run_meta:
        return out
    # Preferred key
    pep = run_meta.get("pretrained_encoder_path", None)
    # Backward/alternate naming
    if pep is None:
        pep = run_meta.get("encoder_checkpoint_path", None)
    if pep is None:
        pep = run_meta.get("encoder_path", None)
    if pep is not None:
        out["pretrained_encoder_path"] = str(pep)
    return out


def main() -> None:
    _add_src_to_path()

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader

    from ecg_chagas_embeddings.analysis.embeddings_probe import (
        DEFAULT_OUTPUT_DIR,
        compute_binary_pauc,
        compute_tpr_at_top_fraction,
        run_memmap_dir,
    )
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs, resolve_data_dir
    from ecg_chagas_embeddings.data.augmentation import ECGAugmentation
    from ecg_chagas_embeddings.data.dataset import TorchDataset, collate_dict_batch
    from ecg_chagas_embeddings.models.resnet18_ecg_flex import LitResNet18

    parser = argparse.ArgumentParser(
        description="Evaluate each run on the full held-out test fold (fold4 by default)."
    )
    parser.add_argument(
        "--run_specs",
        type=Path,
        default=Path("configs/analysis/embeddings_probe_runs.toml"),
        help="TOML file listing runs (run_id/track/preprocessing/checkpoint_path...).",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test_fold", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="'auto'|'cpu'|'cuda'")
    parser.add_argument("--crop_size", type=int, default=2500)
    parser.add_argument("--augmentation_base_seed", type=int, default=42)
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="Also persist full-test logits per run as a float32 memmap (for ranking-agreement analysis).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    global_cfg, runs = load_run_specs(args.run_specs)
    meta_path_raw = str(global_cfg.get("meta_path", "")).strip()
    if not meta_path_raw:
        raise ValueError("run_specs missing global 'meta_path'")
    meta_path = Path(meta_path_raw)

    processed_root_raw = str(global_cfg.get("processed_root", "")).strip()
    processed_root = Path(processed_root_raw) if processed_root_raw else meta_path.parent

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "test_scores.csv"
    if args.overwrite and scores_path.exists():
        scores_path.unlink()
    existing_run_ids: set[str] = set()
    if scores_path.exists() and not args.overwrite:
        try:
            import pandas as pd

            df_existing = pd.read_csv(scores_path, usecols=["run_id"])
            existing_run_ids = set(df_existing["run_id"].astype(str).tolist())
        except Exception:
            existing_run_ids = set()

    download_dir = out_dir / "wandb_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run in tqdm(runs, desc="Runs", unit="run"):
        if not run.checkpoint_path:
            print(f"Skipping {run.run_id}: empty checkpoint_path")
            continue

        # If --save_logits is enabled we may need to run inference even when scores exist.
        need_scores = bool(args.overwrite) or (run.run_id not in existing_run_ids)
        memmap_dir = run_memmap_dir(out_dir, run.run_id)
        logits_path = memmap_dir / f"{run.run_id}__logits__N{{N}}.fp32.mmap"  # placeholder for messages
        need_logits = bool(args.save_logits)

        ckpt = _resolve_checkpoint(run.checkpoint_path, download_dir=download_dir)
        if not ckpt.exists():
            print(f"Skipping {run.run_id}: checkpoint not found: {ckpt}")
            continue

        data_dir = resolve_data_dir(run, processed_root=processed_root)
        if not data_dir.exists():
            print(f"Skipping {run.run_id}: data_dir not found: {data_dir}")
            continue

        transform = ECGAugmentation(
            crop_size=int(args.crop_size),
            n_views=2,
            mode="val",
            base_seed=int(args.augmentation_base_seed),
            val_anchor_clean=True,
        )
        dataset = TorchDataset(
            meta_path=meta_path,
            data_dir=data_dir,
            transforms=transform,
            folds=[int(args.test_fold)],
            return_age_and_sex=True,
            use_code15=True,
            use_ptb_xl=True,
            use_sami_trop=True,
            is_submission=False,
            use_sup_con_views=2,
        )
        N_total = int(len(dataset))
        if need_logits:
            memmap_dir.mkdir(parents=True, exist_ok=True)
            logits_path = memmap_dir / f"{run.run_id}__logits__N{N_total}.fp32.mmap"
            if logits_path.exists() and not args.overwrite:
                need_logits = False
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_dict_batch,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        overrides: dict[str, Any] = {}
        if str(run.track) == "t3":
            overrides = _track3_overrides(run.meta)

        try:
            model = LitResNet18.load_from_checkpoint(
                str(ckpt), map_location="cpu", strict=True, **overrides
            )
        except RuntimeError as exc:
            # Some historical checkpoints include extra keys (e.g. nested criterion params).
            # For offline analysis we can safely ignore unexpected keys as long as core weights load.
            msg = str(exc)
            if "Unexpected key(s) in state_dict" not in msg:
                raise
            print(f"Warning: strict checkpoint load failed for {run.run_id}; retrying strict=False.")
            model = LitResNet18.load_from_checkpoint(
                str(ckpt), map_location="cpu", strict=False, **overrides
            )
        model.eval()
        model.to(device)

        if not need_scores and not need_logits:
            tqdm.write(
                f"Skipping {run.run_id}: scores present and logits memmap already exists "
                f"({scores_path.name}; {logits_path.name})"
            )
            continue

        ys: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        logits_mm: np.memmap | None = None
        offset = 0
        if need_logits:
            logits_mm = np.memmap(
                logits_path,
                mode="w+",
                dtype="float32",
                shape=(N_total,),
            )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc=f"Infer {run.run_id}",
                unit="batch",
                leave=False,
            ):
                y = batch["chagas"].view(-1).detach().cpu().numpy().astype(int)
                x = batch["ecg_views"][:, 0].to(device, non_blocking=True)
                _feats, _proj, logits = model(x)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                logits_cpu = logits.detach().cpu().to(torch.float32).numpy()
                probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
                ys.append(y)
                ps.append(probs.reshape(-1))
                if logits_mm is not None:
                    b = int(logits_cpu.shape[0])
                    logits_mm[offset : offset + b] = logits_cpu.reshape(-1)
                    offset += b

        if logits_mm is not None:
            if offset != N_total:
                raise RuntimeError(
                    f"Internal error: wrote {offset} logits but dataset has N={N_total} for {run.run_id}"
                )
            logits_mm.flush()

        y_true = np.concatenate(ys, axis=0)
        y_score = np.concatenate(ps, axis=0)

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "track": run.track,
            "preprocessing": run.preprocessing,
            "checkpoint_path": str(run.checkpoint_path),
            "N": int(y_true.size),
            "N_pos": n_pos,
            "N_neg": n_neg,
        }
        row.update({k: v for k, v in run.meta.items()})

        m = np.isfinite(y_score)
        y_true_m = y_true[m]
        y_score_m = y_score[m]
        try:
            row["auroc"] = float(roc_auc_score(y_true_m, y_score_m))
        except Exception:
            row["auroc"] = float("nan")
        try:
            row["ap"] = float(average_precision_score(y_true_m, y_score_m))
        except Exception:
            row["ap"] = float("nan")

        row["pauc_fpr0.05"] = float(compute_binary_pauc(y_true_m, y_score_m, max_fpr=0.05))
        row["tpr_top0.05"] = float(compute_tpr_at_top_fraction(y_true_m, y_score_m, fraction=0.05))

        if need_scores:
            rows.append(row)
        print(
            f"{run.run_id}: auroc={row['auroc']:.4f} ap={row['ap']:.4f} "
            f"tpr_top0.05={row['tpr_top0.05']:.4f}"
        )

    if rows:
        _append_rows_csv(scores_path, rows)
        print(f"Wrote {scores_path}")


if __name__ == "__main__":
    main()
