import argparse
import os
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, resample, resample_poly, sosfiltfilt, firwin, filtfilt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from tqdm import tqdm

TOTAL_EXPECTED_FILES = 430_766


_SOFTCLIP_LOWER: np.ndarray | None = None
_SOFTCLIP_UPPER: np.ndarray | None = None
_SOFTCLIP_C: np.ndarray | None = None
_SOFTCLIP_SKIP_EXISTING: bool = False

_NORM_SKIP_EXISTING: bool = False


def _init_softclip_worker(
    lower: np.ndarray, upper: np.ndarray, c: np.ndarray, skip_existing: bool
) -> None:
    global _SOFTCLIP_LOWER, _SOFTCLIP_UPPER, _SOFTCLIP_C, _SOFTCLIP_SKIP_EXISTING
    _SOFTCLIP_LOWER = lower
    _SOFTCLIP_UPPER = upper
    _SOFTCLIP_C = c
    _SOFTCLIP_SKIP_EXISTING = skip_existing


def _softclip_worker(task: tuple[str, str]) -> None:
    in_path, out_path = task
    assert _SOFTCLIP_LOWER is not None
    assert _SOFTCLIP_UPPER is not None
    assert _SOFTCLIP_C is not None
    _softclip_to_file(
        in_path,
        out_path,
        _SOFTCLIP_LOWER,
        _SOFTCLIP_UPPER,
        _SOFTCLIP_C,
        _SOFTCLIP_SKIP_EXISTING,
    )


def _init_norm_worker(skip_existing: bool) -> None:
    global _NORM_SKIP_EXISTING
    _NORM_SKIP_EXISTING = skip_existing


def _norm_worker(task: tuple[str, str]) -> None:
    in_path, out_path = task
    _norm_to_file(in_path, out_path, _NORM_SKIP_EXISTING)


def _is_hpc_environment() -> bool:
    return any(
        k in os.environ
        for k in (
            "SLURM_JOB_ID",
            "SLURM_CLUSTER_NAME",
            "PBS_JOBID",
            "LSB_JOBID",
            "SGE_JOB_ID",
        )
    )


def _compute_disable_tqdm(tqdm_mode: str) -> bool:
    if tqdm_mode == "always":
        return False
    if tqdm_mode == "never":
        return True
    if _is_hpc_environment():
        return True
    return not sys.stdout.isatty()


def _parse_dev_folds(s: str) -> set[int]:
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


disable_tqdm = True


def poly_resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    gcd = np.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    return resample_poly(ecg, up=up, down=down, axis=-1)


def fft_resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    ecg_length_in_s = ecg.shape[1] / sample_rate
    num = np.round(ecg_length_in_s * target_sample_rate)
    actual_sample_rate = num / ecg_length_in_s
    error_in_sample_rate = abs(actual_sample_rate - target_sample_rate)

    assert error_in_sample_rate < 0.5, (
        f"Actual sample rate {actual_sample_rate} is not within 0.5 Hz of target sample rate {target_sample_rate}."
    )
    return resample(ecg, num=int(num), axis=-1)


def resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> Tuple[np.ndarray, str]:
    if sample_rate == target_sample_rate:
        return ecg, "None"
    if sample_rate % target_sample_rate == 0 or target_sample_rate % sample_rate == 0:
        return poly_resample_ecg(ecg, sample_rate, target_sample_rate), "Polyphase"
    return fft_resample_ecg(ecg, sample_rate, target_sample_rate), "FFT"


def butter_filter(
    ecg: np.ndarray,
    sample_rate: float,
    lower_freq: float = 0.67,
    upper_freq: float = 45.0,
    order: float = 1.5,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to ECG signal. Cutoffs chosen to match biosppy in neurokit2"""
    # sos = butter(
    #     N=order,
    #     Wn=[lower_freq, upper_freq],
    #     fs=sample_rate,
    #     btype="bandpass",
    #     output="sos",
    # )

    n = int(ecg.shape[-1])

    # BioSPPy/NeuroKit2-style FIR length is ~1.5 * fs. For short signals, SciPy's filtfilt
    # requires len(x) > padlen, where padlen defaults to 3 * (numtaps - 1). Adapt the tap
    # count downward to avoid crashing on short records.
    desired_taps = int(order * sample_rate)
    if desired_taps % 2 == 0:
        desired_taps += 1  # enforce odd

    # SciPy's default filtfilt padlen is 3 * max(len(a), len(b)) for FIR/IIR,
    # so we need n > 3 * taps  => taps <= floor((n - 1) / 3).
    max_taps = max((n - 1) // 3, 1)
    taps = min(desired_taps, max_taps)
    if taps < 3:
        raise ValueError(
            f"Signal too short for filtfilt FIR filtering (n={n}, taps={taps})."
        )
    if taps % 2 == 0:
        taps -= 1
        if taps < 3:
            raise ValueError(
                f"Signal too short for filtfilt FIR filtering (n={n}, taps={taps})."
            )

    # -> filter_signal()
    frequency = [lower_freq, upper_freq]

    #   -> get_filter()
    #     -> _norm_freq()
    frequency = (
        2 * np.array(frequency) / sample_rate
    )  # Normalize frequency to Nyquist Frequency (Fs/2).

    #     -> get coeffs
    a = np.array([1])
    b = firwin(numtaps=taps, cutoff=frequency, pass_zero=False)

    # _filter_signal()
    filtered = filtfilt(b, a, ecg, axis=-1)

    # DC offset
    # filtered -= np.mean(filtered) # removed because comes later in pipeline

    return filtered


def extract_metadata(record_path):
    """Extract metadata from WFDB record header comments."""
    record = wfdb.rdheader(record_path)
    exam_id = record_path.stem
    age, sex, chagas, source = None, None, None, None

    for comment in record.comments:
        if "Age" in comment:
            match = re.search(r"Age:\s*(\d+)", comment)
            if match:
                age = int(match.group(1))
        elif "Sex" in comment:
            match = re.search(r"Sex:\s*(\w+)", comment)
            if match:
                sex = match.group(1)
        elif "Chagas label" in comment:
            match = re.search(r"Chagas label:\s*(\w+)", comment)
            if match:
                chagas = 1 if match.group(1).lower() == "true" else 0
        elif "Source" in comment:
            match = re.search(r"Source:\s*(\S+)", comment)
            if match:
                source = match.group(1)

    return {
        "exam_id": exam_id,
        "age": age,
        "sex": sex,
        "chagas": chagas,
        "source": source,
        "path": str(record_path),
    }


def subtract_channel_medians(ecg: np.ndarray) -> np.ndarray:
    medians = np.median(ecg, axis=1, keepdims=True)
    return ecg - medians


def normalize_per_lead(ecg: np.ndarray) -> np.ndarray:
    """Per-sample, per-lead robust normalization: (x - median) / (IQR + eps)."""
    medians = np.median(ecg, axis=1, keepdims=True)
    iqrs = np.subtract(*np.percentile(ecg, [75, 25], axis=1, keepdims=True))
    iqrs = np.maximum(iqrs, 1e-6)  # guard against flat leads
    return (ecg - medians) / iqrs


def softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def softminus(x):
    return -softplus(-x)


_c_tanh = 2 / (np.e * np.e + 1)
_c_softclip = np.log(2) / _c_tanh


def softclip(x, a=None, b=None, c=_c_softclip):
    if a is not None and b is not None:
        c /= (b - a) / 2

    v = x
    if a is not None:
        v = v - softminus(c * (x - a)) / c
    if b is not None:
        v = v - softplus(c * (x - b)) / c
    return v


def softclip_scale_ecg(
    ecg: np.ndarray, a: float | np.ndarray, b: float | np.ndarray, c: float | np.ndarray
) -> np.ndarray:
    ecg = np.asarray(ecg)
    n_leads = ecg.shape[0]

    a_arr = np.full(n_leads, a) if np.ndim(a) == 0 else np.asarray(a)
    b_arr = np.full(n_leads, b) if np.ndim(b) == 0 else np.asarray(b)
    c_arr = np.full(n_leads, c) if np.ndim(c) == 0 else np.asarray(c)

    clipped = np.vstack(
        [softclip(ecg[i], a=a_arr[i], b=b_arr[i], c=c_arr[i]) for i in range(n_leads)]
    )

    scale = np.maximum(np.abs(a_arr), np.abs(b_arr))
    scale[scale == 0] = 1.0
    return clipped / scale[:, None]


def preprocess_ecg_safe(path, target_sample_rate=400):
    try:
        if isinstance(path, str) and "gbm-radiomics" in path:
            path = path.replace("gbm-radiomics", "dh-face")

        record = wfdb.rdrecord(path)
        sample_rate = wfdb.rdheader(path).fs
        ecg = record.p_signal.T  # (12, T)

        ecg = butter_filter(ecg, sample_rate=sample_rate)
        ecg, method = resample_ecg(
            ecg, sample_rate=sample_rate, target_sample_rate=target_sample_rate
        )
        ecg = subtract_channel_medians(ecg)

        return torch.from_numpy(ecg).float(), method
    except Exception as e:
        print(f"[ERROR] ECG preprocessing failed for {path}: {e}")
        return None


def save_bp_tensor(
    record_path: Path,
    target_sample_rate: int,
    bp_dir: Path,
    skip_existing: bool,
) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """Create and save band passed tensor. Returns (resample_method, bp_path, p1, p99)."""
    bp_path = bp_dir / f"{record_path.stem}.pt"

    if skip_existing and bp_path.is_file():
        x = torch.load(bp_path).float()
        ecg = x.numpy()
        p1 = np.percentile(ecg, 1, axis=1)
        p99 = np.percentile(ecg, 99, axis=1)
        return "Cached", str(bp_path), p1, p99

    out = preprocess_ecg_safe(record_path, target_sample_rate=target_sample_rate)
    if out is None:
        raise RuntimeError(f"Preprocess returned None for {record_path}")

    ecg_t, method = out
    ecg = ecg_t.numpy()

    p1 = np.percentile(ecg, 1, axis=1)  # (12,)
    p99 = np.percentile(ecg, 99, axis=1)  # (12,)

    torch.save(ecg_t, bp_path)

    return str(method), str(bp_path), p1, p99


def extract_meta_and_process_bp(
    record_path: Path,
    target_sample_rate: int,
    bp_dir: Path,
    skip_existing: bool,
    save_meta_only: bool = False,
):
    metadata = extract_metadata(record_path)
    if save_meta_only:
        return metadata

    resample_method, bp_path, p1, p99 = save_bp_tensor(
        record_path, target_sample_rate, bp_dir, skip_existing
    )
    metadata["resample_method"] = resample_method
    metadata["path_bp"] = bp_path
    metadata["p1"] = p1
    metadata["p99"] = p99
    return metadata


def prepare_data_bp(
    records: List[Path],
    bp_dir: Path,
    processes: int = 0,
    target_sample_rate: int = 400,
    skip_existing: bool = False,
    save_meta_only: bool = False,
) -> pd.DataFrame:
    preprocessor = partial(
        extract_meta_and_process_bp,
        target_sample_rate=target_sample_rate,
        bp_dir=bp_dir,
        skip_existing=skip_existing,
        save_meta_only=save_meta_only,
    )

    metadata_list = []

    if processes <= 1:
        for r in tqdm(
            records,
            desc="Process records (bp)",
            total=len(records),
            disable=disable_tqdm,
        ):
            metadata_list.append(preprocessor(r))
    else:
        with Pool(processes) as pool:
            for meta in tqdm(
                pool.imap(preprocessor, records, chunksize=1),
                desc="Process records (bp)",
                total=len(records),
                disable=disable_tqdm,
            ):
                metadata_list.append(meta)

    return pd.DataFrame(metadata_list)


def find_official_records(data_dir: Path, allowed_keywords: List[str]) -> List[Path]:
    records = []
    for f in tqdm(
        data_dir.rglob("*.dat"),
        desc="Scanning for .dat files",
        disable=disable_tqdm,
    ):
        full_path = f.as_posix()
        if any(kw in full_path for kw in allowed_keywords):
            records.append(f.with_suffix(""))
    return records


def add_fold_column(
    df: pd.DataFrame,
    nsplits: int,
    label_col: str = "chagas",
    dataset_col: str = "source",
    group_col: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df.copy()
    y = df[label_col].copy()
    y = y.fillna(-1).astype(int)

    src_codes = df[dataset_col].astype("category").cat.codes.astype(int)
    strat_str = y.astype(str) + "_" + src_codes.astype(str)
    strat_labels, _ = pd.factorize(strat_str)

    if group_col is not None:
        groups = df[group_col].astype(str).fillna("nan_group").values
        splitter = StratifiedGroupKFold(
            n_splits=nsplits,
            shuffle=True,
            random_state=random_state,
        )
        split_iterator = splitter.split(X=df, y=strat_labels, groups=groups)
    else:
        splitter = StratifiedKFold(
            n_splits=nsplits,
            shuffle=True,
            random_state=random_state,
        )
        split_iterator = splitter.split(X=df, y=strat_labels)

    df["fold"] = -1
    for fold_idx, (_, valid_idx) in enumerate(split_iterator):
        df.iloc[valid_idx, df.columns.get_loc("fold")] = fold_idx

    df["fold"] = df["fold"].astype(int)
    return df


def update_ptb_meta(ptb_df, meta_df):
    _df = meta_df.copy()

    _df["path_source"] = _df.path.apply(lambda x: Path(x).parents[2].name)
    ptb_mask = _df["path_source"].eq("ptb-xl")
    if "source" in _df.columns:
        ptb_mask = ptb_mask | _df["source"].astype(str).eq("PTB-XL")
    ptb_paths = _df.loc[ptb_mask, "path"].map(Path)

    def _path_to_record_name(p: Path) -> str:
        rel = p.relative_to(p.parents[2])
        parts = list(rel.parts)
        if parts:
            if parts[0].startswith("processedOfficial500"):
                parts[0] = "records500"
            elif parts[0].startswith("processedOfficial100") or parts[0].startswith(
                "processedOfficial"
            ):
                parts[0] = "records100"
            elif parts[0].startswith("processed"):
                parts[0] = parts[0].replace("processed", "records")
        return str(Path(*parts))

    record_names = ptb_paths.map(_path_to_record_name)
    ptb_info = pd.DataFrame(
        {"path": _df.loc[ptb_mask, "path"], "record_name": record_names}
    )

    melted_ptb_df = pd.melt(
        ptb_df,
        id_vars=["patient_id"],
        value_vars=["filename_hr", "filename_lr"],
        var_name="filename_type",
        value_name="record_name",
    ).drop(columns="filename_type")

    merged = ptb_info.merge(melted_ptb_df, on="record_name", how="left")

    if "patient_id" in merged.columns:
        _df.loc[ptb_mask, "patient_id"] = merged["patient_id"]

    _df.loc[ptb_mask, "chagas"] = 0.0
    return _df


def update_code_meta(code_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    _df = meta_df.copy()
    if "source" not in _df.columns:
        return _df

    code_mask = _df["source"].astype(str).isin({"CODE-15%", "CODE-15"})
    if not code_mask.any():
        return _df

    code_cols = [
        "exam_id",
        "patient_id",
        "RBBB",
        "LBBB",
        "1dAVb",
        "SB",
        "AF",
        "ST",
        "normal_ecg",
        "death",
        "timey",
        "nn_predicted_age",
    ]
    available_cols = [c for c in code_cols if c in code_df.columns]
    code_subset = code_df[available_cols].copy()
    code_subset["exam_id"] = code_subset["exam_id"].astype(str)

    code_side = _df.loc[code_mask].copy()
    code_side["exam_id"] = code_side["exam_id"].astype(str)

    merged = code_side.merge(
        code_subset, on="exam_id", how="left", suffixes=("", "_code")
    )

    pid_col = "patient_id_code" if "patient_id_code" in merged else "patient_id"
    if pid_col in merged.columns:
        _df.loc[code_mask, "patient_id"] = merged[pid_col]

    secondary_cols = [
        "RBBB",
        "LBBB",
        "1dAVb",
        "SB",
        "AF",
        "ST",
        "normal_ecg",
        "death",
        "timey",
        "nn_predicted_age",
    ]
    for col in secondary_cols:
        if col in merged.columns:
            _df.loc[code_mask, col] = merged[col]

    return _df


def update_sami_meta(sami_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    _df = meta_df.copy()
    if "source" not in _df.columns:
        return _df

    sami_mask = _df["source"].astype(str).eq("SaMi-Trop")
    if not sami_mask.any():
        return _df

    sami_cols = ["exam_id", "normal_ecg", "death", "timey", "nn_predicted_age"]
    available_cols = [c for c in sami_cols if c in sami_df.columns]
    sami_subset = sami_df[available_cols].copy()
    sami_subset["exam_id"] = sami_subset["exam_id"].astype(str)

    sami_side = _df.loc[sami_mask].copy()
    sami_side["exam_id"] = sami_side["exam_id"].astype(str)

    merged = sami_side.merge(
        sami_subset, on="exam_id", how="left", suffixes=("", "_sami")
    )

    for col in ["normal_ecg", "death", "timey", "nn_predicted_age"]:
        if col in merged.columns:
            _df.loc[sami_mask, col] = merged[col]

    return _df


def compute_softclip_bounds_from_df(
    df: pd.DataFrame, train_folds: set[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute global per-lead softclip bounds from p1/p99 using ONLY train folds."""
    train = df[df["fold"].isin(train_folds)].copy()
    if train.empty:
        raise ValueError("No rows found for train folds when computing softclip stats.")

    all_p1 = np.vstack(train["p1"].values)
    all_p99 = np.vstack(train["p99"].values)

    global_lower = np.percentile(all_p1, 5, axis=0)
    global_upper = np.percentile(all_p99, 95, axis=0)

    T = np.max([np.abs(global_lower), global_upper], axis=0)
    global_lower = -T
    global_upper = T

    c = (global_upper - global_lower) / 2.0
    return (
        global_lower.astype(np.float32),
        global_upper.astype(np.float32),
        c.astype(np.float32),
    )


def _softclip_to_file(
    in_path: str,
    out_path: str,
    lower: np.ndarray,
    upper: np.ndarray,
    c: np.ndarray,
    skip_existing: bool,
) -> None:
    out_p = Path(out_path)
    if skip_existing and out_p.is_file():
        return
    x = torch.load(in_path)
    y = softclip_scale_ecg(x.numpy(), lower, upper, c)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(y).float(), out_path)


def _norm_to_file(in_path: str, out_path: str, skip_existing: bool) -> None:
    out_p = Path(out_path)
    if skip_existing and out_p.is_file():
        return
    x = torch.load(in_path).numpy()
    y = normalize_per_lead(x)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(y).float(), out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dataset with multiple preprocessing regimes."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ptb_meta_csv", type=Path, default=None)
    parser.add_argument("--code_meta_csv", type=Path, default=None)
    parser.add_argument("--sami_meta_csv", type=Path, default=None)
    parser.add_argument("--save_meta_only", action="store_true")
    parser.add_argument("--output_file", type=str, default="metadata.csv")
    parser.add_argument(
        "--dev_folds",
        type=str,
        default="0,1,2,3",
        help="Comma-separated fold indices to use for dev-only preprocessing stats (e.g. softclip). "
        "Fold(s) not listed (e.g. 4) are treated as held-out/test for stats computation.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, do not rewrite tensors that already exist on disk (bp/bp_sc/bp_sc_norm).",
    )
    parser.add_argument(
        "--tqdm",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help="Progress bar mode. auto=enabled locally, disabled on HPC; always=force on; never=force off.",
    )
    cpu_count = os.cpu_count() or 1
    parser.add_argument("--processes", type=int, default=(cpu_count // 2))
    parser.add_argument("--sample_rate", type=int, default=400)
    parser.add_argument("--splits", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    disable_tqdm = _compute_disable_tqdm(args.tqdm)

    print(f"Using {args.processes} processes.")
    print(f"Only saving metadata: {args.save_meta_only}")

    allowed_keywords = [
        "code15/processed/exams_part",
        "sami-trop/processedOfficial",
        "ptb-xl/processedOfficial500",
    ]
    records = find_official_records(args.data_dir, allowed_keywords)

    if len(records) != TOTAL_EXPECTED_FILES:
        print(
            f"WARNING! Found {len(records)} records. Expected {TOTAL_EXPECTED_FILES}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- regime output dirs ---
    bp_dir = args.output_dir / "bp"
    bp_sc_dir = args.output_dir / "bp_sc"
    bp_sc_norm_dir = args.output_dir / "bp_sc_norm"
    bp_dir.mkdir(parents=True, exist_ok=True)
    bp_sc_dir.mkdir(parents=True, exist_ok=True)
    bp_sc_norm_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: create bp tensors + base metadata ---
    df = prepare_data_bp(
        records=records,
        bp_dir=bp_dir,
        processes=args.processes,
        target_sample_rate=args.sample_rate,
        skip_existing=args.skip_existing,
        save_meta_only=args.save_meta_only,
    )

    print(df.head())

    if args.save_meta_only:
        df.to_csv(args.output_dir / args.output_file, index=False)
        print(f"Metadata saved to {args.output_dir / args.output_file}.")
        sys.exit(0)

    # --- Enrich metadata per dataset BEFORE creating folds ---
    if args.ptb_meta_csv is not None and Path(args.ptb_meta_csv).is_file():
        ptb_df = pd.read_csv(args.ptb_meta_csv)
        df = update_ptb_meta(ptb_df=ptb_df, meta_df=df)

    if args.code_meta_csv is not None and Path(args.code_meta_csv).is_file():
        code_df = pd.read_csv(args.code_meta_csv)
        df = update_code_meta(code_df=code_df, meta_df=df)

    if args.sami_meta_csv is not None and Path(args.sami_meta_csv).is_file():
        sami_df = pd.read_csv(args.sami_meta_csv)
        df = update_sami_meta(sami_df=sami_df, meta_df=df)

    # Ensure patient_id exists
    if "patient_id" not in df.columns:
        df["patient_id"] = df["exam_id"]
    else:
        df["patient_id"] = df["patient_id"].fillna(df["exam_id"])

    # --- Phase 2: compute folds ---
    df = add_fold_column(
        df=df,
        nsplits=args.splits,
        label_col="chagas",
        dataset_col="source",
        group_col="patient_id",
    )

    # --- Phase 3: compute softclip bounds from folds 0-3 only ---
    train_folds = _parse_dev_folds(args.dev_folds)
    lower, upper, c = compute_softclip_bounds_from_df(df, train_folds=train_folds)
    print(f"Softclip bounds (train folds {train_folds}):")
    print(f"lower={lower}")
    print(f"upper={upper}")
    print(f"c={c}")

    bounds_path = args.output_dir / "softclip_bounds.npz"
    np.savez(
        bounds_path,
        lower=lower,
        upper=upper,
        c=c,
        train_folds=np.array(sorted(train_folds), dtype=np.int32),
        sample_rate=np.int32(args.sample_rate),
    )
    print(f"Saved softclip bounds to {bounds_path}")

    # p1/p99 are only needed to compute the dataset-level softclip bounds
    df = df.drop(columns=["p1", "p99"], errors="ignore")

    # --- Phase 4: generate bp_sc and bp_sc_norm tensors ---
    # Create output path columns
    df["path_bp_sc"] = df["path_bp"].apply(
        lambda p: str(bp_sc_dir / (Path(p).stem + ".pt"))
    )
    df["path_bp_sc_norm"] = df["path_bp"].apply(
        lambda p: str(bp_sc_norm_dir / (Path(p).stem + ".pt"))
    )

    # Softclip+scale all samples (including fold 4, using train-only stats)
    tasks_sc = list(zip(df["path_bp"].tolist(), df["path_bp_sc"].tolist()))
    if args.processes <= 1:
        for in_p, out_p in tqdm(
            tasks_sc, desc="Write bp_sc", total=len(tasks_sc), disable=disable_tqdm
        ):
            _softclip_to_file(in_p, out_p, lower, upper, c, args.skip_existing)
    else:
        with Pool(
            args.processes,
            initializer=_init_softclip_worker,
            initargs=(lower, upper, c, args.skip_existing),
        ) as pool:
            for _ in tqdm(
                pool.imap(_softclip_worker, tasks_sc, chunksize=64),
                desc="Write bp_sc",
                total=len(tasks_sc),
                disable=disable_tqdm,
            ):
                pass

    # Robust normalize from bp_sc -> bp_sc_norm
    tasks_norm = list(zip(df["path_bp_sc"].tolist(), df["path_bp_sc_norm"].tolist()))
    if args.processes <= 1:
        for in_p, out_p in tqdm(
            tasks_norm,
            desc="Write bp_sc_norm",
            total=len(tasks_norm),
            disable=disable_tqdm,
        ):
            _norm_to_file(in_p, out_p, args.skip_existing)
    else:
        with Pool(
            args.processes,
            initializer=_init_norm_worker,
            initargs=(args.skip_existing,),
        ) as pool:
            for _ in tqdm(
                pool.imap(_norm_worker, tasks_norm, chunksize=64),
                desc="Write bp_sc_norm",
                total=len(tasks_norm),
                disable=disable_tqdm,
            ):
                pass

    # Save final metadata to CSV
    # Keep a compact reference to processed tensors without storing 3 redundant absolute paths
    if "path_bp" in df.columns:
        df["proc_stem"] = df["path_bp"].apply(lambda p: Path(p).stem)
    df["processed_root"] = str(args.output_dir)

    df_out = df.drop(
        columns=["path_bp", "path_bp_sc", "path_bp_sc_norm"],
        errors="ignore",
    )
    df_out.to_csv(args.output_dir / args.output_file, index=False)
    print(f"Metadata saved to {args.output_dir / args.output_file}.")
