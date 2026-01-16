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


def main() -> None:
    _add_src_to_path()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from ecg_chagas_embeddings.analysis.embeddings_probe import (
        DEFAULT_OUTPUT_DIR,
        ensure_dir,
        legacy_memmap_dir,
        run_memmap_dir,
    )
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs, resolve_data_dir
    from ecg_chagas_embeddings.data.augmentation import ECGAugmentation
    from ecg_chagas_embeddings.data.dataset import TorchDataset, collate_dict_batch
    from ecg_chagas_embeddings.models.resnet18_ecg_flex import LitResNet18

    parser = argparse.ArgumentParser(
        description="Extract encoder/projection embeddings for a fixed probe set and store as memmaps (Pattern A)."
    )
    parser.add_argument(
        "--run_specs",
        type=Path,
        default=Path("configs/analysis/embeddings_probe_runs.toml"),
        help="TOML file listing runs (run_id/track/preprocessing/checkpoint_path...).",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--probe_index",
        type=Path,
        default=None,
        help="Path to probe_index.csv (defaults to <out_dir>/probe_index.csv).",
    )
    parser.add_argument("--test_fold", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="'auto'|'cpu'|'cuda'")
    parser.add_argument("--crop_size", type=int, default=2500)
    parser.add_argument("--augmentation_base_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    global_cfg, runs = load_run_specs(args.run_specs)
    meta_path_raw = str(global_cfg.get("meta_path", "")).strip()
    if not meta_path_raw:
        raise ValueError("run_specs missing global 'meta_path'")
    meta_path = Path(meta_path_raw)
    processed_root_raw = str(global_cfg.get("processed_root", "")).strip()
    processed_root = Path(processed_root_raw) if processed_root_raw else meta_path.parent

    out_dir = ensure_dir(args.out_dir)
    legacy_dir = ensure_dir(legacy_memmap_dir(out_dir))
    download_dir = ensure_dir(out_dir / "wandb_downloads")

    probe_index_path = args.probe_index or (out_dir / "probe_index.csv")
    probe_index = pd.read_csv(probe_index_path)
    required = {"row_idx", "exam_id", "dataset_source"}
    if not required.issubset(probe_index.columns):
        raise ValueError(
            f"{probe_index_path} must include columns {', '.join(sorted(required))}"
        )
    if "chagas" not in probe_index.columns and "y_true" in probe_index.columns:
        probe_index = probe_index.rename(columns={"y_true": "chagas"})
    if "chagas" not in probe_index.columns:
        raise ValueError(f"{probe_index_path} must include label column 'chagas'")
    probe_index = probe_index.sort_values("row_idx").reset_index(drop=True)
    exam_ids = probe_index["exam_id"].astype(str).tolist()
    N = int(len(exam_ids))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    transform = ECGAugmentation(
        crop_size=int(args.crop_size),
        n_views=2,
        mode="val",
        base_seed=int(args.augmentation_base_seed),
        val_anchor_clean=True,
    )

    rows_written: list[dict[str, Any]] = []

    for run in tqdm(runs, desc="Runs", unit="run"):
        if not run.checkpoint_path:
            print(f"Skipping {run.run_id}: empty checkpoint_path")
            continue

        memmap_dir = ensure_dir(run_memmap_dir(out_dir, run.run_id))

        # Fast-path skip if memmaps already exist (avoid loading data/model).
        if not args.overwrite:
            enc_matches = sorted(memmap_dir.glob(f"{run.run_id}__enc__N{N}__D*.fp32.mmap"))
            logits_path = memmap_dir / f"{run.run_id}__logits__N{N}.fp32.mmap"
            if (enc_matches and logits_path.exists()) or (
                sorted(legacy_dir.glob(f"{run.run_id}__enc__N{N}__D*.fp32.mmap"))
                and (legacy_dir / f"{run.run_id}__logits__N{N}.fp32.mmap").exists()
            ):
                tqdm.write(
                    f"Skipping {run.run_id}: memmaps already exist (use --overwrite)"
                )
                continue

        ckpt = _resolve_checkpoint(run.checkpoint_path, download_dir=download_dir)
        if not ckpt.exists():
            print(f"Skipping {run.run_id}: checkpoint not found: {ckpt}")
            continue

        data_dir = resolve_data_dir(run, processed_root=processed_root)
        if not data_dir.exists():
            print(f"Skipping {run.run_id}: data_dir not found: {data_dir}")
            continue

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

        id_to_index: dict[str, int] = {}
        meta = getattr(dataset, "metadata", None)
        if meta is not None and "exam_id" in meta:
            for i, v in enumerate(meta["exam_id"].tolist()):
                sid = str(v)
                if sid not in id_to_index:
                    id_to_index[sid] = int(i)

        missing = [sid for sid in exam_ids if sid not in id_to_index]
        if missing:
            raise KeyError(
                f"{run.run_id}: {len(missing)}/{N} probe exam_ids missing from dataset fold={args.test_fold}; "
                f"first few: {missing[:5]}"
            )

        indices = [id_to_index[sid] for sid in exam_ids]
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_dict_batch,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        try:
            model = LitResNet18.load_from_checkpoint(
                str(ckpt), map_location="cpu", strict=True
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "Unexpected key(s) in state_dict" not in msg:
                raise
            print(f"Warning: strict checkpoint load failed for {run.run_id}; retrying strict=False.")
            model = LitResNet18.load_from_checkpoint(
                str(ckpt), map_location="cpu", strict=False
            )
        model.eval()
        model.to(device)

        # Determine whether to store projection embeddings.
        has_projection = run.has_projection
        if has_projection is None:
            has_projection = not isinstance(getattr(model, "projection_head", nn.Identity()), nn.Identity)
        elif has_projection and isinstance(getattr(model, "projection_head", nn.Identity()), nn.Identity):
            print(f"Warning: {run.run_id} has_projection=true but model.projection_head is Identity; skipping proj.")
            has_projection = False

        # Allocate memmaps after first forward to know dimensions.
        enc_mmap = None
        proj_mmap = None
        logits_mmap = None

        offset = 0
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc=f"Embed {run.run_id}",
                unit="batch",
                leave=False,
            ):
                x = batch["ecg_views"][:, 0].to(device, non_blocking=True)
                feats, proj, logits = model(x)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]

                feats_np = feats.detach().cpu().to(torch.float32).numpy()
                proj_np = proj.detach().cpu().to(torch.float32).numpy()
                logits_np = logits.detach().cpu().to(torch.float32).numpy().reshape(-1)

                if enc_mmap is None:
                    D = int(feats_np.shape[1])
                    enc_path = memmap_dir / f"{run.run_id}__enc__N{N}__D{D}.fp32.mmap"
                    logits_path = memmap_dir / f"{run.run_id}__logits__N{N}.fp32.mmap"
                    if (enc_path.exists() or logits_path.exists()) and not args.overwrite:
                        print(f"Skipping {run.run_id}: memmaps already exist (use --overwrite)")
                        break
                    enc_mmap = np.memmap(enc_path, mode="w+", dtype="float32", shape=(N, D))
                    logits_mmap = np.memmap(logits_path, mode="w+", dtype="float32", shape=(N,))
                    if has_projection:
                        d_proj = int(proj_np.shape[1])
                        proj_path = memmap_dir / f"{run.run_id}__proj__N{N}__D{d_proj}.fp32.mmap"
                        proj_mmap = np.memmap(
                            proj_path, mode="w+", dtype="float32", shape=(N, d_proj)
                        )

                bsz = int(feats_np.shape[0])
                i0, i1 = offset, offset + bsz
                if enc_mmap is None or logits_mmap is None:
                    raise RuntimeError("Internal error: memmaps not allocated")
                enc_mmap[i0:i1] = feats_np.astype(np.float32, copy=False)
                logits_mmap[i0:i1] = logits_np.astype(np.float32, copy=False)
                if has_projection and proj_mmap is not None:
                    proj_mmap[i0:i1] = proj_np.astype(np.float32, copy=False)

                offset = i1

        if enc_mmap is None:
            continue
        if offset != N:
            raise RuntimeError(f"{run.run_id}: wrote {offset} rows, expected {N}")
        enc_mmap.flush()
        if proj_mmap is not None:
            proj_mmap.flush()
        if logits_mmap is not None:
            logits_mmap.flush()

        rows_written.append(
            {
                "run_id": run.run_id,
                "track": run.track,
                "preprocessing": run.preprocessing,
                "checkpoint_path": run.checkpoint_path,
                "N": N,
                "D_enc": int(enc_mmap.shape[1]),
                "has_projection": bool(has_projection),
            }
        )
        print(f"Wrote memmaps for {run.run_id} (N={N})")

    if rows_written:
        df = pd.DataFrame(rows_written)
        df.to_csv(out_dir / "memmap_index.csv", index=False)
        print(f"Wrote {out_dir / 'memmap_index.csv'}")


if __name__ == "__main__":
    main()
