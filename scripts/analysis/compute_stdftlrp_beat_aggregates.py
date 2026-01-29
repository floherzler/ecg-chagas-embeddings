#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _choose_window_width(*, signal_length: int, preferred: int) -> int:
    candidates = [w for w in range(16, signal_length + 1) if signal_length % w == 0]
    if not candidates:
        return signal_length
    return min(candidates, key=lambda w: abs(w - int(preferred)))


def _overlap_weights_for_segment(
    *,
    t_on: int,
    t_off: int,
    window_width: int,
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (window_indices, weights) for a segment [t_on, t_off] (inclusive indices).

    We use overlap / window_width, where each window is [m*H, (m+1)*H) (half-open).
    """
    if t_on is None or t_off is None:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    t_on_i = int(t_on)
    t_off_i = int(t_off)
    if not np.isfinite(t_on_i) or not np.isfinite(t_off_i):
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    if t_off_i < t_on_i:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    seg_start = max(0, t_on_i)
    seg_end = t_off_i + 1  # make inclusive -> half-open
    if seg_end <= seg_start:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    m0 = seg_start // window_width
    m1 = (seg_end - 1) // window_width
    m0 = max(0, int(m0))
    m1 = min(int(m1), int(n_windows) - 1)
    if m1 < m0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    ms = np.arange(m0, m1 + 1, dtype=np.int64)
    w_start = ms * window_width
    w_end = (ms + 1) * window_width
    overlap = np.maximum(0, np.minimum(w_end, seg_end) - np.maximum(w_start, seg_start)).astype(
        np.float32
    )
    weights = overlap / float(window_width)
    keep = weights > 0
    return ms[keep], weights[keep]


def _band_masks(
    *, signal_length: int, fs_hz: float, freq_bands_hz: dict[str, tuple[float, float]], freq_max_hz: float
) -> tuple[np.ndarray, list[str]]:
    freqs = np.fft.rfftfreq(int(signal_length), d=1.0 / float(fs_hz)).astype(np.float32)
    names: list[str] = []
    masks: list[np.ndarray] = []
    for name, (lo, hi) in freq_bands_hz.items():
        lo_f = float(lo)
        hi_f = float(hi)
        names.append(name)
        masks.append((freqs >= lo_f) & (freqs < min(hi_f, float(freq_max_hz))))
    return np.stack(masks, axis=0), names  # [B,F]


@dataclass(frozen=True)
class BeatAggResult:
    beat_mats: np.ndarray  # [n_beats, 4, n_bands]
    sample_mat: np.ndarray  # [4, n_bands]
    n_beats: int


def _compute_beat_level_matrix(
    *,
    relevance_timefreq: np.ndarray,  # [n_windows, F] (signed)
    waves_df: pd.DataFrame,
    window_width: int,
    band_masks: np.ndarray,  # [n_bands, F]
) -> BeatAggResult:
    n_windows, F = relevance_timefreq.shape
    n_bands = int(band_masks.shape[0])

    band_mass = np.zeros((n_windows, n_bands), dtype=np.float32)
    for bi in range(n_bands):
        mask = band_masks[bi]
        if mask.shape[0] != F:
            raise ValueError(f"Band mask length {mask.shape[0]} != F {F}")
        # signed mass per window per band
        band_mass[:, bi] = relevance_timefreq[:, mask].sum(axis=1)

    required_cols = [
        "ECG_P_Onsets",
        "ECG_P_Offsets",
        "ECG_R_Onsets",
        "ECG_R_Offsets",
        "ECG_T_Onsets",
        "ECG_T_Offsets",
    ]
    missing = [c for c in required_cols if c not in waves_df.columns]
    if missing:
        return BeatAggResult(
            beat_mats=np.zeros((0, 4, n_bands), dtype=np.float32),
            sample_mat=np.full((4, n_bands), np.nan, dtype=np.float32),
            n_beats=0,
        )

    def _mid(a: int, b: int) -> int:
        return int((int(a) + int(b)) // 2)

    def _overlap_weights_for_segment(t_on: int, t_off: int) -> tuple[np.ndarray, np.ndarray]:
        seg_start = max(0, int(t_on))
        seg_end = int(t_off) + 1  # inclusive -> half-open
        if seg_end <= seg_start:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        m0 = seg_start // window_width
        m1 = (seg_end - 1) // window_width
        m0 = max(0, int(m0))
        m1 = min(int(m1), int(n_windows) - 1)
        if m1 < m0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        ms = np.arange(m0, m1 + 1, dtype=np.int64)
        w_start = ms * window_width
        w_end = (ms + 1) * window_width
        overlap = np.maximum(0, np.minimum(w_end, seg_end) - np.maximum(w_start, seg_start)).astype(np.float32)
        weights = overlap / float(window_width)
        keep = weights > 0
        return ms[keep], weights[keep]

    seg_names = ["P", "QRS", "T", "Between"]

    n_beats_guess = int(max((len(waves_df.get(col, [])) for col in required_cols), default=0))
    beat_mats: list[np.ndarray] = []
    for b in range(n_beats_guess):
        vals = {c: waves_df.iloc[b][c] for c in required_cols}
        if any(pd.isna(v) for v in vals.values()):
            continue
        p_on = int(vals["ECG_P_Onsets"])
        p_off = int(vals["ECG_P_Offsets"])
        r_on = int(vals["ECG_R_Onsets"])
        r_off = int(vals["ECG_R_Offsets"])
        t_on = int(vals["ECG_T_Onsets"])
        t_off = int(vals["ECG_T_Offsets"])

        m1 = _mid(p_off, r_on)
        m2 = _mid(r_off, t_on)

        segs = [
            ("P", p_on, m1),
            ("QRS", m1, m2),
            ("T", m2, t_off),
        ]
        if b + 1 < n_beats_guess:
            next_p_on = waves_df.iloc[b + 1]["ECG_P_Onsets"]
            if not pd.isna(next_p_on):
                segs.append(("Between", t_off, int(next_p_on)))

        mat = np.zeros((4, n_bands), dtype=np.float32)
        any_seg = False
        for si, (_sname, t_on, t_off) in enumerate(segs):
            ms, ws = _overlap_weights_for_segment(int(t_on), int(t_off))
            if ms.size == 0:
                continue
            any_seg = True
            mat[si, :] = (band_mass[ms, :] * ws[:, None]).sum(axis=0)

        if not any_seg:
            continue

        denom = float(np.abs(mat).sum())
        if denom <= 0:
            continue
        beat_mats.append(mat / denom)

    if not beat_mats:
        return BeatAggResult(
            beat_mats=np.zeros((0, 4, n_bands), dtype=np.float32),
            sample_mat=np.full((4, n_bands), np.nan, dtype=np.float32),
            n_beats=0,
        )

    beat_arr = np.stack(beat_mats, axis=0)
    sample_mat = beat_arr.mean(axis=0)
    return BeatAggResult(beat_mats=beat_arr, sample_mat=sample_mat, n_beats=int(beat_arr.shape[0]))


def _load_exam_ids(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "exam_id" not in df.columns:
        raise ValueError(f"Missing exam_id column in {path}")
    return df["exam_id"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute beat-normalized STDFT-LRP relevance aggregates.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or wandb artifact id.")
    parser.add_argument("--run_id", default="run", help="Used only for naming outputs.")
    parser.add_argument(
        "--meta_path",
        type=Path,
        default=Path("/home/flo178/projects/master-thesis/datasets/physionet2025/processedMaster/metadata.csv"),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/flo178/projects/master-thesis/datasets/physionet2025/processedMaster/bp_sc_norm"),
        help="Folder with preprocessed tensors (*.pt) matching the model preprocessing.",
    )
    parser.add_argument("--exam_ids_csv", type=Path, required=True, help="CSV with exam_id column.")
    parser.add_argument("--out_dir", type=Path, default=Path("./analysis/xai"))
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--lead_index", type=int, default=1, help="Lead used when --all_leads is not set.")
    parser.add_argument(
        "--all_leads",
        action="store_true",
        help="Compute per-lead matrices for all leads and aggregate across leads.",
    )
    parser.add_argument("--fs_hz", type=float, default=400.0)
    parser.add_argument("--freq_max_hz", type=float, default=45.0)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--window_width", type=int, default=125, help="STDFT window width H (will be snapped to a divisor of T).")
    parser.add_argument(
        "--window_shift_factor",
        type=int,
        default=1,
        help="Upstream dft-lrp uses a shift factor; effective hop D = H // window_shift_factor. Use 1 for D=H (no overlap).",
    )
    parser.add_argument("--window_shape", type=str, default="rectangle", choices=["rectangle", "halfsine"])
    parser.add_argument("--write_per_beat", action="store_true", help="Also write a per-beat long table.")
    parser.add_argument(
        "--write_per_lead",
        action="store_true",
        help="Also write a per-lead long table (one row per lead per sample).",
    )
    args = parser.parse_args()

    _add_src_to_path()
    import torch
    import neurokit2 as nk

    from ecg_chagas_embeddings.models.resnet18_ecg_flex import LitResNet18
    from ecg_chagas_embeddings.callbacks.xai_probe import compute_lrp_relevance_time, _import_dft_lrp

    dft_lrp = _import_dft_lrp()

    meta = pd.read_csv(args.meta_path, low_memory=False)
    meta = meta.copy()
    meta["exam_id"] = meta["exam_id"].astype(str)
    if "fold" in meta.columns:
        meta = meta[pd.to_numeric(meta["fold"], errors="coerce").fillna(-1).astype(int) == int(args.fold)]
    meta = meta.drop_duplicates(subset=["exam_id"], keep="first").reset_index(drop=True)
    meta_idx = meta.set_index("exam_id", drop=False)

    exam_ids = _load_exam_ids(args.exam_ids_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitResNet18.load_from_checkpoint(str(args.checkpoint), map_location="cpu", log_umap=False)
    model.to(device).eval()

    out_root = args.out_dir / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    freq_bands_hz = {
        "low": (0.67, 4.0),
        "mid": (4.0, 12.0),
        "high": (12.0, 25.0),
        "vhigh": (25.0, 45.0),
    }

    # Lazy init: all samples in this pipeline are typically fixed-length (e.g., 2500 @ 400 Hz).
    # Creating the upstream STDFT weight matrices is expensive; reuse them across samples.
    cached_T: int | None = None
    cached_H: int | None = None
    cached_band_masks: np.ndarray | None = None
    cached_band_names: list[str] | None = None
    dftlrp_tf_obj = None
    dftlrp_full_obj = None

    rows: list[dict[str, object]] = []
    beat_rows: list[dict[str, object]] = []
    lead_rows: list[dict[str, object]] = []

    for exam_id in tqdm(exam_ids, desc="Samples"):
        pt_path = args.data_dir / f"{exam_id}.pt"
        if not pt_path.exists():
            continue

        row_meta = meta_idx.loc[exam_id].to_dict() if exam_id in meta_idx.index else {}

        x0 = torch.load(pt_path, map_location="cpu")
        if isinstance(x0, dict):
            for k in ("ecg", "signal", "x"):
                if k in x0:
                    x0 = x0[k]
                    break
        if torch.is_tensor(x0) and x0.ndim == 3:
            x0 = x0[0]
        if not torch.is_tensor(x0) or x0.ndim != 2:
            continue
        x0 = x0.to(torch.float32)
        T = int(x0.shape[-1])

        if cached_T is None:
            cached_T = T
            cached_H = _choose_window_width(signal_length=T, preferred=int(args.window_width))
            if int(cached_H) <= 0:
                raise ValueError(f"Invalid window width selected: {cached_H}")
            cached_band_masks, cached_band_names = _band_masks(
                signal_length=T,
                fs_hz=args.fs_hz,
                freq_bands_hz=freq_bands_hz,
                freq_max_hz=args.freq_max_hz,
            )
            dftlrp_tf_obj = dft_lrp.DFTLRP(
                T,
                leverage_symmetry=True,
                precision=int(args.precision),
                cuda=(device.type == "cuda"),
                window_shift=int(args.window_shift_factor),
                window_width=int(cached_H),
                window_shape=str(args.window_shape),
                create_dft=False,
                create_inverse=False,
            )
            dftlrp_full_obj = dft_lrp.DFTLRP(
                T,
                leverage_symmetry=True,
                precision=int(args.precision),
                cuda=(device.type == "cuda"),
                create_stdft=False,
                create_inverse=False,
            )
        else:
            if T != int(cached_T):
                raise ValueError(
                    f"Encountered varying signal lengths in one run: first T={cached_T}, now T={T} for exam_id={exam_id}"
                )

        assert cached_H is not None
        assert cached_band_masks is not None
        assert cached_band_names is not None
        assert dftlrp_tf_obj is not None
        assert dftlrp_full_obj is not None
        H = int(cached_H)
        band_masks = cached_band_masks
        band_names = cached_band_names

        x = x0.unsqueeze(0).to(device=device)  # [1,12,T]
        with torch.no_grad():
            out = model(x)
            # reuse the callback's logic to extract logits
            from ecg_chagas_embeddings.callbacks.xai_probe import extract_pos_logit

            logit = extract_pos_logit(out).detach().cpu().numpy()
            prob = float(_sigmoid(logit)[0])

        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            rel_time = compute_lrp_relevance_time(pl_module=model, x=x)  # [1,12,T]

        rel_time_np = rel_time[0].detach().cpu().numpy().astype(np.float32, copy=False)  # [L,T]
        lead_mass = np.sum(np.abs(rel_time_np), axis=1)
        lead_mass_sum = float(np.sum(lead_mass))
        p_lead = lead_mass / (lead_mass_sum + 1e-12)
        lead_entropy = float(
            -np.sum(p_lead * np.log(p_lead + 1e-12)) / np.log(len(p_lead) + 1e-12)
        )
        top1_lead = int(np.argmax(p_lead))
        top3_leads = np.argsort(p_lead)[-3:][::-1].astype(int).tolist()

        seg_names = ["P", "QRS", "T", "Between"]
        n_leads = int(x0.shape[0])
        lead_indices = list(range(n_leads)) if args.all_leads else [int(args.lead_index)]

        per_lead_mats: list[np.ndarray] = []
        per_lead_abs_mass_tf: list[float] = []
        per_lead_abs_mass_freq: list[float] = []
        per_lead_freq_mats: list[np.ndarray] = []
        per_lead_n_beats: list[int] = []

        for lead in lead_indices:
            sig_1d = x0[lead].detach().cpu().numpy().astype(np.float64, copy=False)
            rel_1d = rel_time_np[lead].astype(np.float32, copy=False)

            try:
                ecg_cleaned = nk.ecg_clean(sig_1d, sampling_rate=int(args.fs_hz), method="neurokit")
                _signals_peaks, info_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(args.fs_hz))
                rpeaks = info_peaks["ECG_R_Peaks"]
                _sig_dwt, waves_dwt = nk.ecg_delineate(
                    ecg_cleaned,
                    rpeaks,
                    sampling_rate=int(args.fs_hz),
                    method="dwt",
                    show=False,
                )
                waves_df = pd.DataFrame({k: pd.Series(v) for k, v in waves_dwt.items()})
            except Exception:
                continue

            try:
                _sig_tf, rel_tf = dftlrp_tf_obj.dft_lrp(
                    rel_1d[None, :],
                    sig_1d[None, :],
                    real=False,
                    short_time=True,
                    epsilon=1e-6,
                )
                rel_tf = np.asarray(rel_tf)[0].astype(np.float32)  # [n_windows, F]
            except Exception:
                continue

            try:
                _sig_f, rel_f = dftlrp_full_obj.dft_lrp(
                    rel_1d[None, :],
                    sig_1d[None, :],
                    real=False,
                    short_time=False,
                    epsilon=1e-6,
                )
                rel_f = np.asarray(rel_f)[0].astype(np.float32)  # [F]
            except Exception:
                continue

            agg = _compute_beat_level_matrix(
                relevance_timefreq=rel_tf,
                waves_df=waves_df,
                window_width=int(H),
                band_masks=band_masks,
            )

            per_lead_mats.append(agg.sample_mat.astype(np.float32))
            per_lead_abs_mass_tf.append(float(np.sum(np.abs(rel_tf))))
            per_lead_abs_mass_freq.append(float(np.sum(np.abs(rel_f))))
            freq_vec = np.zeros((len(band_names),), dtype=np.float32)
            freq_denom = float(np.sum(np.abs(rel_f)))
            if freq_denom > 0:
                for bi, mask in enumerate(band_masks):
                    freq_vec[bi] = float(rel_f[mask].sum()) / freq_denom
            per_lead_freq_mats.append(freq_vec)
            per_lead_n_beats.append(int(agg.n_beats))

            if args.write_per_beat and agg.n_beats > 0:
                for b in range(agg.beat_mats.shape[0]):
                    br = {
                        "exam_id": exam_id,
                        "lead_index": int(lead),
                        "beat_idx": int(b),
                    }
                    for si, sname in enumerate(seg_names):
                        for bi, bname in enumerate(band_names):
                            br[f"rel_{sname}_{bname}"] = float(agg.beat_mats[b, si, bi])
                    beat_rows.append(br)

            if args.write_per_lead:
                lr = {
                    "exam_id": exam_id,
                    "lead_index": int(lead),
                    "abs_mass_tf": float(np.sum(np.abs(rel_tf))),
                    "abs_mass_freq": float(np.sum(np.abs(rel_f))),
                    "n_beats": int(agg.n_beats),
                }
                for si, sname in enumerate(seg_names):
                    for bi, bname in enumerate(band_names):
                        lr[f"rel_{sname}_{bname}"] = float(agg.sample_mat[si, bi])
                for bi, bname in enumerate(band_names):
                    lr[f"freq_rel_{bname}"] = float(freq_vec[bi])
                lead_rows.append(lr)

        if not per_lead_mats:
            continue

        lead_mats = np.stack(per_lead_mats, axis=0)  # [L', 4, 4]
        lead_freq = np.stack(per_lead_freq_mats, axis=0)  # [L', B]
        m_mean = lead_mats.mean(axis=0)
        lead_w = np.asarray(per_lead_abs_mass_tf, dtype=np.float32)
        if float(lead_w.sum()) <= 0.0:
            m_weighted = m_mean
        else:
            m_weighted = np.average(lead_mats, axis=0, weights=lead_w)
        lead_w_freq = np.asarray(per_lead_abs_mass_freq, dtype=np.float32)
        if float(lead_w_freq.sum()) <= 0.0:
            freq_mean = lead_freq.mean(axis=0)
            freq_weighted = freq_mean
        else:
            freq_mean = lead_freq.mean(axis=0)
            freq_weighted = np.average(lead_freq, axis=0, weights=lead_w_freq)

        out_row: dict[str, object] = {
            "exam_id": exam_id,
            "patient_id": row_meta.get("patient_id", None),
            "source": row_meta.get("source", None),
            "chagas": row_meta.get("chagas", None),
            "qc_zhao2018_bp": row_meta.get("qc_zhao2018_bp", None),
            "qc_templatematch_bp": row_meta.get("qc_templatematch_bp", None),
            "T": int(T),
            "window_width": int(H),
            "n_windows": int(T // H),
            "n_beats_mean": float(np.mean(per_lead_n_beats)),
            "n_valid_leads": int(len(per_lead_mats)),
            "p_chagas": float(prob),
            "lead_entropy": float(lead_entropy),
            "top1_lead": int(top1_lead),
            "top3_leads": ",".join(str(i) for i in top3_leads),
            "lead_mass_sum": float(lead_mass_sum),
        }
        for li, p in enumerate(p_lead):
            out_row[f"p_lead_{li}"] = float(p)
            out_row[f"lead_mass_{li}"] = float(lead_mass[li])
        for si, sname in enumerate(seg_names):
            for bi, bname in enumerate(band_names):
                out_row[f"rel_mean_{sname}_{bname}"] = float(m_mean[si, bi])
                out_row[f"rel_weighted_{sname}_{bname}"] = float(m_weighted[si, bi])
        for bi, bname in enumerate(band_names):
            out_row[f"freq_rel_mean_{bname}"] = float(freq_mean[bi])
            out_row[f"freq_rel_weighted_{bname}"] = float(freq_weighted[bi])

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_path = out_root / "stdftlrp_beat_agg.csv"
    out_df.to_csv(out_path, index=False)

    if args.write_per_beat and beat_rows:
        beat_df = pd.DataFrame(beat_rows)
        beat_path = out_root / "stdftlrp_beat_per_beat.csv"
        beat_df.to_csv(beat_path, index=False)
    if args.write_per_lead and lead_rows:
        lead_df = pd.DataFrame(lead_rows)
        lead_path = out_root / "stdftlrp_beat_per_lead.csv"
        lead_df.to_csv(lead_path, index=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
