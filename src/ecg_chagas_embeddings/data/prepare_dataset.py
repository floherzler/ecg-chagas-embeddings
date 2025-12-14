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
from scipy.signal import butter, resample, resample_poly, sosfiltfilt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from tqdm import tqdm

# from .stats import MEAN, STD

TOTAL_EXPECTED_FILES = 430_766

disable_tqdm = not sys.stdout.isatty()


def poly_resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    gcd = np.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd

    resampled_ecg = resample_poly(ecg, up=up, down=down, axis=-1)
    return resampled_ecg


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

    resampled_ecg = resample(ecg, num=int(num), axis=-1)
    return resampled_ecg


def resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> Tuple[np.ndarray, str]:
    """Resample an ECG. We use polymorphic resampling if the original sampling rate and target sampling
    rate are integer multiples of each other. Otherwise, FFT resampling is used"""

    if sample_rate == target_sample_rate:
        return ecg, "None"

    if sample_rate % target_sample_rate == 0 or target_sample_rate % sample_rate == 0:
        return poly_resample_ecg(ecg, sample_rate, target_sample_rate), "Polyphase"

    return fft_resample_ecg(ecg, sample_rate, target_sample_rate), "FFT"


def butter_filter(
    ecg: np.ndarray,
    sample_rate: float,
    lower_freq: float = 1,
    upper_freq: float = 47,
    order: int = 3,
) -> np.ndarray:
    sos = butter(
        N=order,
        Wn=[lower_freq, upper_freq],
        fs=sample_rate,
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(sos, ecg)


def extract_metadata(record_path):
    """Extract metadata from .hea file."""
    record = wfdb.rdheader(record_path)  # Read header file
    exam_id = record_path.stem  # Filename without extension
    age, sex, chagas, source = None, None, None, None  # Default values

    for comment in record.comments:
        # Use regular expressions to find and extract the required information. Faster then split
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


def save_resampled_normalized_ecg(
    record_path: Path, target_sample_rate: int, output_dir: Path
) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """Save resampled and normalized ECG to .pt file."""

    ecg, method = preprocess_ecg_safe(
        record_path, target_sample_rate=target_sample_rate
    )

    p1 = np.percentile(ecg, 1, axis=1)  # shape (12,)
    p99 = np.percentile(ecg, 99, axis=1)

    # Save resampled ECG to .pt file
    output_path = output_dir / f"{record_path.stem}.pt"
    torch.save(torch.tensor(ecg), output_path)

    return str(method), str(output_path), p1, p99


def _softclip_save(
    pt_path: Path, lower: np.ndarray, upper: np.ndarray, c: np.ndarray
) -> None:
    """Apply softclip scaling to a saved tensor and overwrite in place."""
    torch.save(
        torch.from_numpy(
            softclip_scale_ecg(torch.load(pt_path).numpy(), lower, upper, c)
        ).float(),
        pt_path,
    )


def extract_meta_and_process(
    record_path: Path,
    target_sample_rate: int,
    output_dir: Path,
    save_meta_only: bool = False,
):
    """Extract metadata and process ECG."""
    metadata = extract_metadata(record_path)
    if save_meta_only:
        return metadata
    else:
        resample_method, path, p1, p99 = save_resampled_normalized_ecg(
            record_path, target_sample_rate, output_dir
        )
        metadata["resample_method"] = resample_method
        metadata["path"] = path
        metadata["p1"] = p1
        metadata["p99"] = p99
        return metadata


def prepare_data(
    records: List[Path],
    output_dir: Path,
    processes: int = 0,
    target_sample_rate: int = 400,
    save_meta_only: bool = False,
):
    """Create metadata CSV file."""
    preprocessor = partial(
        extract_meta_and_process,
        target_sample_rate=target_sample_rate,
        output_dir=output_dir,
        save_meta_only=save_meta_only,
    )

    # total = len(records)
    metadata_list = []

    if processes <= 1:
        for r in tqdm(
            records,
            desc="Process records",
            total=len(records),
            disable=disable_tqdm,
        ):
            metadata_list.append(preprocessor(r))
    else:
        with Pool(processes) as pool:
            # imap preserves ordering, chunksize=1 streams as soon as ready
            for meta in tqdm(
                pool.imap(preprocessor, records, chunksize=1),
                desc="Process records",
                total=len(records),
                disable=disable_tqdm,
            ):
                metadata_list.append(meta)

    if save_meta_only:
        df = pd.DataFrame(metadata_list)
        return df

    all_p1 = np.vstack([m["p1"] for m in metadata_list])
    all_p99 = np.vstack([m["p99"] for m in metadata_list])

    for lead in range(12):
        col_p1 = all_p1[:, lead]
        col_p99 = all_p99[:, lead]
        stats = {
            "1pct of 1pct": np.percentile(col_p1, 1),
            "5pct of 1pct": np.percentile(col_p1, 5),
            "10pct of 1pct": np.percentile(col_p1, 10),
            "90pct of 99pct": np.percentile(col_p99, 90),
            "95pct of 99pct": np.percentile(col_p99, 95),
            "99pct of 99pct": np.percentile(col_p99, 99),
        }
        print(
            f"Lead {lead:2d} → " + ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
        )
    global_lower = np.percentile(
        all_p1, 5, axis=0
    )  # e.g. 5th‐percentile of chunk‐level 1%’s
    global_upper = np.percentile(all_p99, 95, axis=0)

    T = np.max([np.abs(global_lower), global_upper], axis=0)
    global_lower = -T
    global_upper = T

    for lead in range(12):
        print(
            f"Lead {lead:2d} → "
            + f"Global lower={global_lower[lead]:.4f}, "
            + f"Global upper={global_upper[lead]:.4f}"
        )

    c = (global_upper - global_lower) / 2.0

    print(f"Global lower bound: {global_lower}")
    print(f"Global upper bound: {global_upper}")
    print(f"Softclip corner sharpness: {c}")

    pt_files = list(output_dir.glob("*.pt"))

    if processes <= 1:
        for ptf in tqdm(
            pt_files,
            desc="Softclip scale",
            total=len(pt_files),
            disable=disable_tqdm,
        ):
            _softclip_save(ptf, global_lower, global_upper, c)
    else:
        with Pool(processes) as pool:
            list(
                tqdm(
                    pool.imap(
                        partial(
                            _softclip_save,
                            lower=global_lower,
                            upper=global_upper,
                            c=c,
                        ),
                        pt_files,
                    ),
                    desc="Softclip scale",
                    total=len(pt_files),
                    disable=disable_tqdm,
                )
            )

    # @Flo commented this out because it was not working on the cluster
    # if processes == 0:
    #     metadata_list = [preprocessor(r) for r in tqdm(records, disable=disable_tqdm)]
    # else:
    #     with Pool(processes) as p:
    #         # imap results in better progress bar. map would be faster.
    #         metadata_list = p.imap(
    #             preprocessor, tqdm(records, desc="Process all .dat files.", disable=disable_tqdm)
    #         )

    df = pd.DataFrame(metadata_list)

    return df


def find_all_records(data_dir: Path) -> List[Path]:
    return [
        f.with_suffix("")
        for f in tqdm(
            data_dir.rglob("*.dat"),
            desc="Find all .dat files.",
            total=TOTAL_EXPECTED_FILES,
            disable=disable_tqdm,
        )
    ]


def find_official_records(data_dir: Path, allowed_keywords: List[str]) -> List[Path]:
    """
    Find .dat files in subdirectories that include any of the allowed keywords in their full path.
    This matches folders like 'processedOfficial500' or 'exams_part0'.

    Args:
        data_dir (Path): Root search directory.
        allowed_keywords (List[str]): Keywords for subdirectory filtering.

    Returns:
        List[Path]: .dat file paths (without suffix), filtered by allowed keywords.
    """
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
    """
    Add a 'fold' column with stratified (and optionally grouped) K-fold assignments.

    - Stratifies on a composite of (label, dataset_col) to balance Chagas condition per source.
    - Optionally groups by group_col (e.g. patient_id or exam_id) to avoid leakage.
    """

    df = df.copy()

    # --- 1. Build stratification label: (label, source) ---
    # Handle missing labels safely
    y = df[label_col].copy()
    y = y.fillna(-1).astype(int)

    # Encode dataset/source as integers
    src_codes = df[dataset_col].astype("category").cat.codes.astype(int)

    # Composite strat label: e.g. "0_0", "1_1", ...
    strat_str = y.astype(str) + "_" + src_codes.astype(str)
    # Factorize to integers because StratifiedKFold wants a 1D label array
    strat_labels, _ = pd.factorize(strat_str)

    # --- 2. Choose stratified splitter (with or without groups) ---
    if group_col is not None:
        # Ensure groups are a single dtype to avoid mixed type sorting errors
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

    # --- 3. Assign fold indices ---
    df["fold"] = -1  # init

    for fold_idx, (_, valid_idx) in enumerate(split_iterator):
        df.iloc[valid_idx, df.columns.get_loc("fold")] = fold_idx

    df["fold"] = df["fold"].astype(int)
    return df


def update_ptb_meta(ptb_df, meta_df):
    """Update PTB metadata with patient_id information from ptb_df.

    Args:
        ptb_df (pd.DataFrame): DataFrame containing PTB metadata with columns 'filename_hr', 'filename_lr', 'patient_id'.
        meta_df (pd.DataFrame): DataFrame containing metadata with column 'path'.

    Returns:
        pd.DataFrame: Updated metadata DataFrame with patient_id information.
    """
    _df = meta_df.copy()

    # Only process rows from ptb-xl
    _df["path_source"] = _df.path.apply(lambda x: Path(x).parents[2].name)
    ptb_mask = _df["path_source"].eq("ptb-xl")
    if "source" in _df.columns:
        ptb_mask = ptb_mask | _df["source"].astype(str).eq("PTB-XL")
    ptb_paths = _df.loc[ptb_mask, "path"].map(Path)

    def _path_to_record_name(p: Path) -> str:
        """Map processed path back to the original PTB-XL record name (records500/… or records100/…)."""
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

    # Extract record_name from relative path
    record_names = ptb_paths.map(_path_to_record_name)

    # Create a temporary DataFrame for merging
    ptb_info = pd.DataFrame(
        {"path": _df.loc[ptb_mask, "path"], "record_name": record_names}
    )

    # Melt ptb_df to unify filename_hr and filename_lr into a single column
    melted_ptb_df = pd.melt(
        ptb_df,
        id_vars=["patient_id"],
        value_vars=["filename_hr", "filename_lr"],
        var_name="filename_type",
        value_name="record_name",
    ).drop(columns="filename_type")

    # Merge on record_name
    merged = ptb_info.merge(melted_ptb_df, on="record_name", how="left")

    # Update patient_id in _df using index alignment
    if "patient_id" in merged.columns:
        _df.loc[ptb_mask, "patient_id"] = merged["patient_id"]

    # PTB has no chagas cases!
    _df.loc[ptb_mask, "chagas"] = 0.0

    return _df


def update_code_meta(code_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Merge CODE-15 metadata (patient_id and optional secondary labels) into meta_df.

    Note: age/sex are taken from the WFDB .hea headers and are not overwritten from the CSV.
    """
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
    if "exam_id" in code_subset.columns:
        code_subset["exam_id"] = code_subset["exam_id"].astype(str)

    code_side = _df.loc[code_mask].copy()
    if "exam_id" in code_side.columns:
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
    """Merge SaMi-Trop metadata (optional secondary labels) into meta_df.

    Note: age/sex are taken from the WFDB .hea headers and are not overwritten from the CSV.
    """
    _df = meta_df.copy()

    if "source" not in _df.columns:
        return _df

    sami_mask = _df["source"].astype(str).eq("SaMi-Trop")
    if not sami_mask.any():
        return _df

    sami_cols = [
        "exam_id",
        "normal_ecg",
        "death",
        "timey",
        "nn_predicted_age",
    ]
    available_cols = [c for c in sami_cols if c in sami_df.columns]
    sami_subset = sami_df[available_cols].copy()
    if "exam_id" in sami_subset.columns:
        sami_subset["exam_id"] = sami_subset["exam_id"].astype(str)

    sami_side = _df.loc[sami_mask].copy()
    if "exam_id" in sami_side.columns:
        sami_side["exam_id"] = sami_side["exam_id"].astype(str)

    merged = sami_side.merge(
        sami_subset, on="exam_id", how="left", suffixes=("", "_sami")
    )

    for col in ["normal_ecg", "death", "timey", "nn_predicted_age"]:
        if col in merged.columns:
            _df.loc[sami_mask, col] = merged[col]

    return _df


def normalize_per_lead(ecg):
    medians = np.median(ecg, axis=1, keepdims=True)
    iqrs = np.subtract(*np.percentile(ecg, [75, 25], axis=1, keepdims=True))
    return (ecg - medians) / (iqrs + 1e-6)


def subtract_channel_means(ecg):
    medians = np.median(ecg, axis=1, keepdims=True)
    return ecg - medians


def softplus(x):
    """numerically stable calculation for log(1 + exp(x))"""
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def softminus(x):
    return -softplus(-x)


# default scaling constants to match tanh corner shape
_c_tanh = 2 / (np.e * np.e + 1)  # == 1 - np.tanh(1) ~= 0.24
_c_softclip = np.log(2) / _c_tanh


def tanhclip(x, a, b):
    """Canonical soft-clipping with tanh(x).  Must specify both endpoints"""
    scale = (b - a) / 2
    return (np.tanh((x - a) / scale - 1) + 1) * scale + a


def softclip(x, a=None, b=None, c=_c_softclip):
    """
    Clipping with softplus and softminus, with paramterized corner sharpness.
    Set either (or both) endpoint to None to indicate no clipping at that end.
    """
    # when clipping at both ends, make c dimensionless w.r.t. b - a / 2
    if a is not None and b is not None:
        c /= (b - a) / 2

    v = x
    if a is not None:
        v = v - softminus(c * (x - a)) / c
    if b is not None:
        v = v - softplus(c * (x - b)) / c
    return v


# def expclip(x, a=None, b=None, c=_c_expclip):
#     """
#     Exponential soft clipping, with parameterized corner sharpness.
#     Simpler functional form but 3rd, 5th, ... and subequent odd derivatives are discontinuous at 0
#     """
#     if a is not None and b is not None:
#         c /= (b - a) / 2

#     v = np.clip(x, a, b)
#     if a is not None:
#         v = v + np.exp(-c * np.abs(xs - a)) / (2 * c)
#     if b is not None:
#         v = v - np.exp(-c * np.abs(xs - b)) / (2 * c)
#     return v


def softclip_scale_ecg(
    ecg: np.ndarray, a: float | np.ndarray, b: float | np.ndarray, c: float | np.ndarray
) -> np.ndarray:
    """
    ecg: shape (n_leads, T)
    a, b, c: either scalar or array of length n_leads
    Returns: clipped & scaled ecg of same shape, each lead in [-1,1].
    """
    ecg = np.asarray(ecg)
    n_leads = ecg.shape[0]

    # Broadcast a, b, c to shape (n_leads,)
    a_arr = np.full(n_leads, a) if np.ndim(a) == 0 else np.asarray(a)
    b_arr = np.full(n_leads, b) if np.ndim(b) == 0 else np.asarray(b)
    c_arr = np.full(n_leads, c) if np.ndim(c) == 0 else np.asarray(c)

    # Soft-clip each lead independently
    clipped = np.vstack(
        [softclip(ecg[i], a=a_arr[i], b=b_arr[i], c=c_arr[i]) for i in range(n_leads)]
    )

    # Scale each lead into [-1,1]
    scale = np.maximum(np.abs(a_arr), np.abs(b_arr))
    scale[scale == 0] = 1.0  # guard against zero
    return clipped / scale[:, None]


def softclip_vectorized(
    ecg: np.ndarray, a: float = -10.0, b: float = 10.0, c: float = _c_softclip
) -> np.ndarray:
    """
    Vectorized softclipping of ECG data for all leads at once.
    Assumes ecg is shaped (12, T)
    """
    if a is not None and b is not None:
        c /= (b - a) / 2

    v = ecg.copy()

    if a is not None:
        v = (
            v
            - (
                np.log(1 + np.exp(-np.abs(c * (ecg - a))))
                + np.maximum(c * (ecg - a), 0)
            )
            / c
        )

    if b is not None:
        v = (
            v
            - (
                np.log(1 + np.exp(-np.abs(c * (ecg - b))))
                + np.maximum(c * (ecg - b), 0)
            )
            / c
        )

    return v


def preprocess_ecg_safe(path, target_sample_rate=400):
    try:
        if isinstance(path, str) and "gbm-radiomics" in path:
            path = path.replace("gbm-radiomics", "dh-face")
        record = wfdb.rdrecord(path)
        sample_rate = wfdb.rdheader(path).fs

        ecg = record.p_signal.T
        # if not np.all(np.isfinite(ecg)):
        #     raise ValueError("Non-finite values found in raw signal")

        ecg = butter_filter(ecg, sample_rate=sample_rate)
        # if not np.all(np.isfinite(ecg)):
        #     raise ValueError("Non-finite values after filtering")

        ecg, method = resample_ecg(
            ecg, sample_rate=sample_rate, target_sample_rate=target_sample_rate
        )
        # if not np.all(np.isfinite(ecg)):
        #     raise ValueError("Non-finite values after resampling")

        ecg = subtract_channel_means(ecg)
        # if not np.all(np.isfinite(ecg)):
        #     raise ValueError("Non-finite values after normalization")

        # max_abs = np.abs(ecg).max()
        # if max_abs > 1.0:
        #     print(f"[WARNING] Invalid ECG range for {path}")
        #     print(f"  Max: {np.nanmax(ecg):.3f}, Min: {np.nanmin(ecg):.3f}, Max abs: {max_abs:.3f}")
        #     print(f"  IQR stats per lead: {np.subtract(*np.percentile(ecg, [75, 25], axis=1))}")
        #     print(f"  Median stats per lead: {np.median(ecg, axis=1)}")
        #     print("Setting to zero array to prevent NaNs.")
        #     ecg = np.zeros_like(ecg)

        return torch.from_numpy(ecg).float(), method
    except Exception as e:
        print(f"[ERROR] ECG preprocessing failed for {path}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to data directory. Parent directory to search for .dat files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to output directory. Here we will save preprocessed .pt files and metadata.",
    )
    parser.add_argument(
        "--ptb_meta_csv",
        type=Path,
        default=None,
        help="Optional path to PTB-XL metadata CSV (ptbxl_database.csv).",
    )
    parser.add_argument(
        "--code_meta_csv",
        type=Path,
        default=None,
        help="Optional path to CODE-15 metadata CSV (exams.csv).",
    )
    parser.add_argument(
        "--sami_meta_csv",
        type=Path,
        default=None,
        help="Optional path to SaMi-Trop metadata CSV (exams.csv).",
    )
    parser.add_argument(
        "--save_meta_only",
        action="store_true",
        help="If set, only save metadata without processing ECG files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metadata.csv",
        help="Output file name. Saved into output_dir. CSV file containing metadata.",
    )
    cpu_count = os.cpu_count() or 1
    parser.add_argument(
        "--processes",
        type=int,
        default=(cpu_count // 2),
        help="Number of processes to use. If 0, don't use multiprocessing. Default: cpu_count() // 2.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=400,
        help="Target sample rate for ECG resampling.",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of splits.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Using {args.processes} processes.")
    print(f"Only saving metadata: {args.save_meta_only}")

    # Find all records *.dat files
    # records = find_all_records(args.data_dir)

    allowed_keywords = [
        "code15/processed/exams_part",  # matches exams_part0, exams_part20, etc.
        "sami-trop/processedOfficial",  # matches sami-trop/processedOfficial
        "ptb-xl/processedOfficial500",  # matches ptb-xl/processedOfficial500
    ]

    records = find_official_records(args.data_dir, allowed_keywords)

    if len(records) != TOTAL_EXPECTED_FILES:
        print(
            f"WARNING! Found {len(records)} records. Expected {TOTAL_EXPECTED_FILES}."
        )

    # Create metadata with path, age, sex, chagas label and resample method.
    # Save resampled ECG to .pt file.

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(
        records=records,
        output_dir=args.output_dir,
        processes=args.processes,
        target_sample_rate=args.sample_rate,
        save_meta_only=args.save_meta_only,
    )

    print(df.head())

    # Enrich metadata per dataset BEFORE creating folds
    ptb_meta_csv = getattr(args, "ptb_meta_csv", None)
    if ptb_meta_csv is not None and Path(ptb_meta_csv).is_file():
        ptb_df = pd.read_csv(ptb_meta_csv)
        df = update_ptb_meta(ptb_df=ptb_df, meta_df=df)

    code_meta_csv = getattr(args, "code_meta_csv", None)
    if code_meta_csv is not None and Path(code_meta_csv).is_file():
        code_df = pd.read_csv(code_meta_csv)
        df = update_code_meta(code_df=code_df, meta_df=df)

    sami_meta_csv = getattr(args, "sami_meta_csv", None)
    if sami_meta_csv is not None and Path(sami_meta_csv).is_file():
        sami_df = pd.read_csv(sami_meta_csv)
        df = update_sami_meta(sami_df=sami_df, meta_df=df)

    # Ensure patient_id exists
    if "patient_id" not in df.columns:
        df["patient_id"] = df["exam_id"]
    else:
        df["patient_id"] = df["patient_id"].fillna(df["exam_id"])

    # Generate cross validation splits using enriched metadata
    df = add_fold_column(
        df,
        nsplits=args.splits,
        label_col="chagas",
        dataset_col="source",
        group_col="patient_id",
    )

    # try:
    #     print("Fold distribution by chagas and source:")
    #     print(df.groupby(["fold", "source"])["chagas"].value_counts(normalize=True))
    # except Exception as e:
    #     print(f"Could not compute fold distribution: {e}")

    # Save metadata to CSV
    df.to_csv(args.output_dir / args.output_file, index=False)
    print(f"Metadata saved to {args.output_dir / args.output_file}.")
