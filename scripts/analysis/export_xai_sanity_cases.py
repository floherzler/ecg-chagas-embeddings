#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from subprocess import run as run_subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _center_crop_or_pad(signal: torch.Tensor, target_length: int) -> torch.Tensor:
    if signal.ndim != 2:
        raise ValueError(f"Expected [C,T], got {tuple(signal.shape)}")
    if target_length <= 0:
        return signal
    t = int(signal.shape[-1])
    if t == target_length:
        return signal
    if t > target_length:
        start = max(0, (t - target_length) // 2)
        end = start + target_length
        return signal[:, start:end]
    pad = target_length - t
    left = pad // 2
    right = pad - left
    import torch.nn.functional as f

    return f.pad(signal, (left, right))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _normalize_source(s: object) -> str:
    x = str(s).strip().upper().replace("-", "").replace("_", "").replace("%", "")
    if "CODE" in x:
        return "CODE15"
    if "SAMITROP" in x:
        return "SAMITROP"
    if "PTB" in x:
        return "PTBXL"
    return x


def _fmt_meta(v: object, *, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "nan":
            return "-"
        return s
    try:
        if pd.isna(v):
            return "-"
    except Exception:
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


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
        safe_name = uri.replace("/", "__").replace(":", "__").replace("\\", "__")
        cached_root = download_dir / safe_name
        if cached_root.exists():
            for fname in ("model.ckpt", "best.ckpt"):
                c = cached_root / fname
                if c.exists():
                    return c
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
            c = local_dir / fname
            if c.exists():
                return c
        ckpts = sorted(local_dir.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found in wandb artifact {uri}")
        return ckpts[0]

    if p.is_dir():
        for fname in ("model.ckpt", "best.ckpt"):
            c = p / fname
            if c.exists():
                return c
        ckpts = sorted(p.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found under {p}")
        return ckpts[0]
    return p


def _extract_pos_logit(model_out: object) -> torch.Tensor:
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        model_out = model_out[-1]
    if isinstance(model_out, dict):
        model_out = model_out["logits"]
    if not torch.is_tensor(model_out):
        raise TypeError(f"Unsupported model output type: {type(model_out)}")
    logits = model_out
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:, 1]
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def _choose_window_width(signal_length: int, preferred: int) -> int:
    cands = [w for w in range(16, signal_length + 1) if signal_length % w == 0]
    if not cands:
        return signal_length
    return min(cands, key=lambda w: abs(w - int(preferred)))


def _robust_signed(x: np.ndarray, q: float = 0.99, eps: float = 1e-12) -> np.ndarray:
    s = np.quantile(np.abs(x), q) + eps
    return np.clip(x / s, -1.0, 1.0)


def _load_tensor(path: Path) -> torch.Tensor:
    x0 = torch.load(path, map_location="cpu")
    if isinstance(x0, dict):
        for k in ("ecg", "signal", "x"):
            if k in x0:
                x0 = x0[k]
                break
    if torch.is_tensor(x0) and x0.ndim == 3:
        x0 = x0[0]
    if not torch.is_tensor(x0) or x0.ndim != 2:
        raise ValueError(f"Unsupported tensor payload at {path}")
    return x0.to(torch.float32)


def _plot_composite(
    *,
    signal_time: np.ndarray,
    relevance_time: np.ndarray,
    relevance_freq: np.ndarray,
    signal_timefreq: np.ndarray,
    relevance_timefreq: np.ndarray,
    fs_hz: float,
    freq_max_hz: float,
    window_width: int,
    window_shift_factor: int,
    title: str,
    out_path: Path,
) -> None:
    t = np.arange(signal_time.shape[-1], dtype=np.float32) / float(fs_hz)
    r_t = _robust_signed(relevance_time)

    n_f = min(relevance_freq.shape[-1], relevance_timefreq.shape[-1])
    freqs = np.linspace(0.0, float(fs_hz) / 2.0, int(n_f), dtype=np.float32)
    k_max = int(np.searchsorted(freqs, float(freq_max_hz), side="right"))
    k_max = max(1, min(k_max, n_f))
    freqs = freqs[:k_max]

    amp_f = np.abs(relevance_freq[:k_max]).astype(np.float32)
    r_f = _robust_signed(relevance_freq[:k_max])

    tf_sig = np.abs(signal_timefreq[:, :k_max]).T.astype(np.float32)  # [F,W]
    tf_rel = relevance_timefreq[:, :k_max].T.astype(np.float32)  # [F,W]
    tf_rel_n = _robust_signed(tf_rel, q=0.97)

    n_windows = tf_sig.shape[1]
    step = max(1, int(window_width // max(1, window_shift_factor)))
    t0 = 0.0
    t1 = float(((n_windows - 1) * step + window_width) / float(fs_hz))
    extent = [t0, t1, float(freqs[0]), float(freqs[-1])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    ax = axes[0]
    ax.plot(t, signal_time, color="black", lw=1.0, alpha=0.8)
    ax.fill_between(t, 0.0, r_t, color="tab:red", alpha=0.25)
    ax.set_title("Time signal + signed relevance overlay")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / rel.")

    ax = axes[1]
    ax.plot(freqs, amp_f, color="tab:blue", lw=1.0, alpha=0.9)
    ax.fill_between(freqs, 0.0, r_f, color="tab:red", alpha=0.25)
    ax.set_title("Frequency relevance (DFT-LRP)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|signal| / rel.")

    ax = axes[2]
    im0 = ax.imshow(
        tf_sig,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys",
        alpha=0.85,
    )
    _ = im0
    im1 = ax.imshow(
        tf_rel_n,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        alpha=0.45,
    )
    plt.colorbar(im1, ax=ax, pad=0.01, fraction=0.02, label="Normalized signed relevance")
    ax.set_title("Time-frequency relevance (ST-DFT-LRP)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    fig.suptitle(title, fontsize=13, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_delineation_overlap(
    *,
    ecg_signal: np.ndarray,
    ecg_cleaned: np.ndarray,
    waves: dict,
    segments: dict,
    fs_hz: float,
    title: str,
    out_path: Path,
    max_beats: int = 10,
) -> None:
    t = np.arange(ecg_signal.shape[-1], dtype=np.float32) / float(fs_hz)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    ax = axes[0]
    ax.plot(t, ecg_cleaned, color="black", lw=0.8, alpha=0.9, label="Clean ECG")
    marker_map = {
        "ECG_P_Onsets": ("tab:orange", "P_on"),
        "ECG_P_Offsets": ("tab:orange", "P_off"),
        "ECG_R_Onsets": ("tab:red", "R_on"),
        "ECG_R_Offsets": ("tab:red", "R_off"),
        "ECG_T_Onsets": ("tab:green", "T_on"),
        "ECG_T_Offsets": ("tab:green", "T_off"),
    }
    for key, (c, lbl) in marker_map.items():
        vals = np.asarray(waves.get(key, []), dtype=float)
        vals = vals[np.isfinite(vals)].astype(int)
        vals = vals[(vals >= 0) & (vals < ecg_cleaned.shape[0])]
        if vals.size == 0:
            continue
        ax.scatter(vals / float(fs_hz), ecg_cleaned[vals], s=8, c=c, label=lbl, alpha=0.8)
    ax.set_title("NeuroKit delineation onsets/offsets")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG")
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    ax = axes[1]
    keys = list(segments.keys())[: int(max_beats)]
    for i, k in enumerate(keys):
        df = segments[k]
        if "Signal" not in df.columns or "Label" not in df.columns:
            continue
        x = df["Label"].to_numpy(dtype=float)
        y = df["Signal"].to_numpy(dtype=float)
        if x.size == 0:
            continue
        x0 = x - np.nanmean(x)
        ax.plot(x0 / float(fs_hz), y, lw=0.9, alpha=0.55)
    ax.set_title("Overlapping heartbeat segments (ecg_segment)")
    ax.set_xlabel("Relative time (s)")
    ax.set_ylabel("ECG")

    fig.suptitle(title, fontsize=13, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch export notebook-style XAI sanity figures.")
    parser.add_argument("--run_specs", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("analysis/embeddings_probe"))
    parser.add_argument("--candidates_csv", type=Path, required=True)
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help="Comma-separated run_ids to export.",
    )
    parser.add_argument("--lead_index", type=int, default=7)
    parser.add_argument("--crop_size", type=int, default=2500)
    parser.add_argument("--window_width", type=int, default=125)
    parser.add_argument("--window_shift_factor", type=int, default=1)
    parser.add_argument("--window_shape", type=str, default="rectangle")
    parser.add_argument("--fs_hz", type=float, default=400.0)
    parser.add_argument("--freq_max_hz", type=float, default=45.0)
    parser.add_argument("--max_cases", type=int, default=0, help="0 => all")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    _add_src_to_path()
    from ecg_chagas_embeddings.analysis.run_specs import load_run_specs, resolve_data_dir
    from ecg_chagas_embeddings.models.resnet18_ecg_flex import LitResNet18
    from ecg_chagas_embeddings.callbacks.xai_probe import compute_lrp_relevance_time, _import_dft_lrp

    dft_lrp = _import_dft_lrp()

    global_cfg, runs = load_run_specs(args.run_specs)
    by_id = {r.run_id: r for r in runs}
    run_ids = [x.strip() for x in str(args.run_ids).split(",") if x.strip()]
    missing = [r for r in run_ids if r not in by_id]
    if missing:
        raise ValueError(f"Unknown run_ids in --run_ids: {missing}")

    cand = pd.read_csv(args.candidates_csv, low_memory=False).copy()
    if "exam_id" not in cand.columns:
        raise ValueError("candidates CSV must contain exam_id")
    cand["exam_id"] = cand["exam_id"].astype(str)
    if "dataset_source" in cand.columns:
        cand["dataset_source"] = cand["dataset_source"].map(_normalize_source)
    if int(args.max_cases) > 0:
        cand = cand.head(int(args.max_cases)).copy()

    # Optional metadata enrichment for titles (used when candidate CSV is sparse,
    # e.g., raw-score extremes selected from full-fold predictions).
    probe_meta_path = args.out_dir / "probe_metadata.csv"
    probe_meta = None
    if probe_meta_path.exists():
        probe_meta = pd.read_csv(probe_meta_path, low_memory=False).copy()
        probe_meta["exam_id"] = probe_meta["exam_id"].astype(str)
        keep_cols = [
            "exam_id",
            "qc_zhao2018_bp",
            "qc_templatematch_bp",
            "normal_ecg",
            "RBBB",
            "ptb_any_rbbb",
            "ptb_crbbb",
        ]
        keep_cols = [c for c in keep_cols if c in probe_meta.columns]
        probe_meta = probe_meta[keep_cols].drop_duplicates(subset=["exam_id"])
        probe_meta = probe_meta.set_index("exam_id", drop=True)

    processed_root = Path(str(global_cfg.get("processed_root", "")).strip())
    if not processed_root:
        raise ValueError("processed_root not found in run_specs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_dir = args.out_dir / "wandb_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    export_root = args.out_dir / "xai_summary" / f"lead_{int(args.lead_index)}" / "sanity_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for rid in run_ids:
        run = by_id[rid]
        ckpt_path = _resolve_checkpoint(str(run.checkpoint_path), download_dir=download_dir)
        data_dir = resolve_data_dir(run, processed_root=processed_root)
        model = LitResNet18.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", log_umap=False, strict=False
        )
        model.to(device).eval()

        dft_tf_obj = None
        dft_f_obj = None
        cached_t = None
        cached_h = None

        run_out = export_root / rid
        run_out.mkdir(parents=True, exist_ok=True)

        for i, row in cand.iterrows():
            exam_id = str(row["exam_id"])
            pt_path = data_dir / f"{exam_id}.pt"
            if not pt_path.exists():
                continue

            out_base = run_out / f"{i:03d}_{exam_id}"
            out_comp = out_base.with_name(out_base.name + "__composite.png")
            out_seg = out_base.with_name(out_base.name + "__segments.png")
            if args.skip_existing and out_comp.exists() and out_seg.exists():
                manifest_rows.append(
                    {
                        "run_id": rid,
                        "exam_id": exam_id,
                        "composite_png": str(out_comp),
                        "segments_png": str(out_seg),
                        "status": "skipped_exists",
                    }
                )
                continue

            try:
                x0 = _load_tensor(pt_path)
                x0 = _center_crop_or_pad(x0, int(args.crop_size))
                t_len = int(x0.shape[-1])
                if cached_t is None:
                    cached_t = t_len
                    cached_h = _choose_window_width(t_len, int(args.window_width))
                    dft_tf_obj = dft_lrp.DFTLRP(
                        t_len,
                        leverage_symmetry=True,
                        precision=32,
                        cuda=(device.type == "cuda"),
                        window_shift=int(args.window_shift_factor),
                        window_width=int(cached_h),
                        window_shape=str(args.window_shape),
                        create_dft=False,
                        create_inverse=False,
                    )
                    dft_f_obj = dft_lrp.DFTLRP(
                        t_len,
                        leverage_symmetry=True,
                        precision=32,
                        cuda=(device.type == "cuda"),
                        create_stdft=False,
                        create_inverse=False,
                    )
                if t_len != int(cached_t):
                    raise ValueError(f"Signal length changed in one run ({cached_t} -> {t_len})")
                assert dft_tf_obj is not None
                assert dft_f_obj is not None
                assert cached_h is not None

                x = x0.unsqueeze(0).to(device=device)
                with torch.no_grad():
                    out = model(x)
                    logit = _extract_pos_logit(out).detach().cpu().numpy()
                    p_chagas = float(_sigmoid(logit)[0])

                with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
                    device_type=device.type, enabled=False
                ):
                    rel_time = compute_lrp_relevance_time(pl_module=model, x=x)

                lead = int(args.lead_index)
                sig_1d = x0[lead].detach().cpu().numpy().astype(np.float64, copy=False)
                rel_1d = (
                    rel_time[0, lead].detach().cpu().numpy().astype(np.float32, copy=False)
                )

                sig_tf, rel_tf = dft_tf_obj.dft_lrp(
                    rel_1d[None, :],
                    sig_1d[None, :],
                    real=False,
                    short_time=True,
                    epsilon=1e-6,
                )
                _, rel_f = dft_f_obj.dft_lrp(
                    rel_1d[None, :],
                    sig_1d[None, :],
                    real=False,
                    short_time=False,
                    epsilon=1e-6,
                )
                signal_timefreq = np.asarray(sig_tf)[0].astype(np.float32)  # [W,F]
                relevance_timefreq = np.asarray(rel_tf)[0].astype(np.float32)  # [W,F]
                relevance_freq = np.asarray(rel_f)[0].astype(np.float32)  # [F]

                ecg_cleaned = nk.ecg_clean(sig_1d, sampling_rate=int(args.fs_hz), method="neurokit")
                _, info_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(args.fs_hz))
                rpeaks = info_peaks["ECG_R_Peaks"]
                segments = nk.ecg_segment(
                    ecg_cleaned, rpeaks, sampling_rate=int(args.fs_hz), show=False
                )
                _, waves = nk.ecg_delineate(
                    ecg_cleaned,
                    rpeaks,
                    sampling_rate=int(args.fs_hz),
                    method="dwt",
                    show=False,
                )

                ds = row.get("dataset_source", "")
                qz = row.get("qc_zhao2018_bp", "")
                rset = row.get("robust_set", "")
                ch = row.get("chagas", "")
                rank = row.get("rra_rank", np.nan)
                qt = row.get("qc_templatematch_bp", np.nan)
                normal = row.get("normal_ecg", np.nan)
                rbbb = row.get("rbbb_equiv", np.nan)
                if probe_meta is not None and exam_id in probe_meta.index:
                    pm = probe_meta.loc[exam_id]
                    if _fmt_meta(qz) == "-":
                        qz = pm.get("qc_zhao2018_bp", qz)
                    if _fmt_meta(qt) == "-":
                        qt = pm.get("qc_templatematch_bp", qt)
                    if _fmt_meta(normal) == "-":
                        normal = pm.get("normal_ecg", normal)
                    if _fmt_meta(rbbb) == "-":
                        rbbb = pm.get("RBBB", pm.get("ptb_any_rbbb", pm.get("ptb_crbbb", rbbb)))
                title = (
                    f"{rid} | exam={exam_id} | ds={ds} | robust={rset} | ch={ch} | "
                    f"qz={_fmt_meta(qz)} | qt={_fmt_meta(qt)} | normal={_fmt_meta(normal, digits=0)} | "
                    f"rbbb={_fmt_meta(rbbb, digits=0)} | rank={_fmt_meta(rank, digits=0)} | "
                    f"p={p_chagas:.3f} | lead={lead}"
                )

                _plot_composite(
                    signal_time=sig_1d.astype(np.float32),
                    relevance_time=rel_1d,
                    relevance_freq=relevance_freq,
                    signal_timefreq=signal_timefreq,
                    relevance_timefreq=relevance_timefreq,
                    fs_hz=float(args.fs_hz),
                    freq_max_hz=float(args.freq_max_hz),
                    window_width=int(cached_h),
                    window_shift_factor=int(args.window_shift_factor),
                    title=title,
                    out_path=out_comp,
                )
                _plot_delineation_overlap(
                    ecg_signal=sig_1d.astype(np.float32),
                    ecg_cleaned=ecg_cleaned.astype(np.float32),
                    waves=waves,
                    segments=segments,
                    fs_hz=float(args.fs_hz),
                    title=title,
                    out_path=out_seg,
                )

                manifest_rows.append(
                    {
                        "run_id": rid,
                        "exam_id": exam_id,
                        "dataset_source": ds,
                        "robust_set": rset,
                        "chagas": ch,
                        "qc_zhao2018_bp": qz,
                        "rra_rank": rank,
                        "p_chagas": p_chagas,
                        "composite_png": str(out_comp),
                        "segments_png": str(out_seg),
                        "status": "ok",
                    }
                )
                print(f"[ok] {rid} exam={exam_id}")
            except Exception as exc:
                manifest_rows.append(
                    {
                        "run_id": rid,
                        "exam_id": exam_id,
                        "composite_png": str(out_comp),
                        "segments_png": str(out_seg),
                        "status": f"error: {exc}",
                    }
                )
                print(f"[error] {rid} exam={exam_id}: {exc}")

    manifest = pd.DataFrame(manifest_rows)
    out_manifest = export_root / "manifest.csv"
    manifest.to_csv(out_manifest, index=False)
    print(f"Wrote manifest: {out_manifest}")


if __name__ == "__main__":
    main()
