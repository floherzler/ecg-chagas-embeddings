from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("analysis/embeddings_probe")

DEFAULT_MASTER_META_PATH = Path(
    "/home/flo178/projects/master-thesis/datasets/physionet2025/processedMaster/metadata.csv"
)
DEFAULT_PROCESSED_ROOT = Path(
    "/home/flo178/projects/master-thesis/datasets/physionet2025/processedMaster"
)

DEFAULT_CODE15_EXAMS_PATH = Path(
    "/home/flo178/projects/master-thesis/datasets/physionet2025/code15/exams.csv"
)
DEFAULT_SAMITROP_EXAMS_PATH = Path(
    "/home/flo178/projects/master-thesis/datasets/physionet2025/sami-trop/exams.csv"
)
DEFAULT_PTBXL_DB_PATH = Path(
    "/home/flo178/projects/master-thesis/datasets/physionet2025/ptb-xl/ptbxl_database.csv"
)

RUNS_DIRNAME = "runs"


def run_dir(out_dir: Path, run_id: str) -> Path:
    return Path(out_dir) / RUNS_DIRNAME / str(run_id)


def run_memmap_dir(out_dir: Path, run_id: str) -> Path:
    return run_dir(out_dir, run_id) / "memmap"


def run_coords_dir(out_dir: Path, run_id: str) -> Path:
    return run_dir(out_dir, run_id) / "coords"


def run_plots_dir(out_dir: Path, run_id: str) -> Path:
    return run_dir(out_dir, run_id) / "plots"


def legacy_memmap_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "memmap"


def legacy_coords_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "coords"


def legacy_plots_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "plots"


def parse_ptb_scp_codes(value: Any) -> dict[str, Any]:
    """
    Robustly parse PTB-XL `scp_codes` field.

    Expected input: string like "{'CRBBB': 100.0, 'LAFB': 100.0, 'SR': 0.0}".
    Returns {} for missing/invalid values.
    """
    if value is None:
        return {}
    if isinstance(value, float) and np.isnan(value):
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def add_ptb_rbbb_flags(df: pd.DataFrame, *, scp_col: str = "scp_codes") -> pd.DataFrame:
    codes = df.get(scp_col, pd.Series([None] * len(df), index=df.index))
    parsed = codes.map(parse_ptb_scp_codes)
    df = df.copy()
    df["ptb_crbbb"] = parsed.map(lambda d: "CRBBB" in d).astype(bool)
    df["ptb_irbbb"] = parsed.map(lambda d: "IRBBB" in d).astype(bool)
    df["ptb_any_rbbb"] = (df["ptb_crbbb"] | df["ptb_irbbb"]).astype(bool)
    df["ptb_lafb"] = parsed.map(lambda d: "LAFB" in d).astype(bool)
    df["ptb_normal_ecg"] = parsed.map(lambda d: "NORM" in d).astype(bool)
    return df


def l2_normalize_np(x: np.ndarray, *, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x)
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (denom + float(eps))


def compute_tpr_at_top_fraction(
    y_true: np.ndarray, y_score: np.ndarray, *, fraction: float = 0.05
) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    pos = int((y_true == 1).sum())
    if pos == 0:
        return float("nan")
    n = int(y_true.size)
    k = int(np.floor(float(fraction) * n))
    k = max(1, min(n, k))
    order = np.argsort(-y_score, kind="mergesort")
    top = y_true[order[:k]]
    tp = int((top == 1).sum())
    return float(tp / pos)


def compute_binary_pauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    max_fpr: float = 0.05,
    normalize: bool = True,
) -> float:
    """
    Partial AUC for ROC in FPR ∈ [0, max_fpr], normalized by max_fpr by default.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]

    pos = y_true == 1
    neg = y_true == 0
    P = int(pos.sum())
    N = int(neg.sum())
    if P == 0 or N == 0:
        return float("nan")

    max_fpr = float(max_fpr)
    if not (0.0 < max_fpr <= 1.0):
        raise ValueError(f"max_fpr must be in (0,1], got {max_fpr}.")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = tps / float(P)
    fpr = fps / float(N)

    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])

    if max_fpr >= 1.0:
        area = float(np.trapezoid(tpr, fpr))
        return area if not normalize else area

    idx = int(np.searchsorted(fpr, max_fpr, side="right") - 1)
    idx = int(np.clip(idx, 0, len(fpr) - 2))
    fpr_lo, fpr_hi = float(fpr[idx]), float(fpr[idx + 1])
    tpr_lo, tpr_hi = float(tpr[idx]), float(tpr[idx + 1])
    if fpr_hi > fpr_lo:
        alpha = (max_fpr - fpr_lo) / (fpr_hi - fpr_lo)
        tpr_at = tpr_lo + alpha * (tpr_hi - tpr_lo)
    else:
        tpr_at = tpr_lo

    fpr_seg = np.concatenate([fpr[: idx + 1], np.array([max_fpr])])
    tpr_seg = np.concatenate([tpr[: idx + 1], np.array([tpr_at])])
    area = float(np.trapezoid(tpr_seg, fpr_seg))
    return float(area / max_fpr) if normalize else float(area)


def _map_master_source_to_dataset_source(source: str) -> str:
    s = str(source)
    if s.startswith("CODE-15"):
        return "CODE15"
    if s.startswith("PTB-XL"):
        return "PTBXL"
    if s.startswith("SaMi-Trop"):
        return "SAMITROP"
    return s.replace("-", "").replace("%", "")


def load_test_master_table(
    *, meta_path: Path = DEFAULT_MASTER_META_PATH, test_fold: int = 4
) -> pd.DataFrame:
    df = pd.read_csv(
        meta_path,
        usecols=["exam_id", "source", "chagas", "fold"],
        dtype={"exam_id": str, "source": str},
        low_memory=False,
    )
    df = df[df["fold"] == int(test_fold)].copy()
    df = df[~df["chagas"].isna()].copy()
    df["exam_id"] = df["exam_id"].astype(str)
    df["chagas"] = (df["chagas"].astype(float) > 0.5).astype(int)
    df["dataset_source"] = df["source"].map(_map_master_source_to_dataset_source)
    return df[["exam_id", "dataset_source", "chagas"]].reset_index(drop=True)


def load_master_quality(*, meta_path: Path = DEFAULT_MASTER_META_PATH) -> pd.DataFrame:
    """
    Load per-exam signal quality metrics from processedMaster/metadata.csv.

    - qc_zhao2018_bp: categorical (e.g. Excellent/Barely acceptable/Unacceptable)
    - qc_templatematch_bp: continuous score (higher is better)
    - resample_method: string
    """
    df = pd.read_csv(
        meta_path,
        usecols=["exam_id", "qc_zhao2018_bp", "qc_templatematch_bp", "resample_method"],
        dtype={"exam_id": str},
        low_memory=False,
    )
    df["exam_id"] = df["exam_id"].astype(str)
    return df.drop_duplicates(subset=["exam_id"], keep="first").reset_index(drop=True)


def load_code15_exams(path: Path = DEFAULT_CODE15_EXAMS_PATH) -> pd.DataFrame:
    usecols = [
        "exam_id",
        "age",
        "is_male",
        "nn_predicted_age",
        "1dAVb",
        "RBBB",
        "LBBB",
        "SB",
        "ST",
        "AF",
        "patient_id",
        "death",
        "timey",
        "normal_ecg",
        "trace_file",
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["exam_id"] = df["exam_id"].astype(str)
    return df


def load_samitrop_exams(path: Path = DEFAULT_SAMITROP_EXAMS_PATH) -> pd.DataFrame:
    usecols = [
        "exam_id",
        "age",
        "is_male",
        "normal_ecg",
        "death",
        "timey",
        "nn_predicted_age",
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["exam_id"] = df["exam_id"].astype(str)
    return df


def load_ptbxl_database(path: Path = DEFAULT_PTBXL_DB_PATH) -> pd.DataFrame:
    usecols = ["ecg_id", "patient_id", "age", "sex", "scp_codes", "filename_hr"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["exam_id"] = df["filename_hr"].map(lambda v: Path(str(v)).name)
    df = add_ptb_rbbb_flags(df, scp_col="scp_codes")
    df["is_male"] = df["sex"].map(
        lambda v: bool(int(v) == 1) if pd.notna(v) else np.nan
    )
    # PTB-XL uses HIPAA-compliant age obfuscation: ages >89 appear in a high range (e.g. ~300).
    # Map them to 100 for a readable "age 0..100" visualization range.
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    df.loc[df["age"] > 89, "age"] = 100.0
    return df


def build_stratification_frame(
    df_test: pd.DataFrame,
    *,
    code15: pd.DataFrame,
    samitrop: pd.DataFrame,
    ptbxl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich the fold4 master table with phenotype columns used for stratification.
    """
    parts: list[pd.DataFrame] = []

    # CODE15
    code = df_test[df_test["dataset_source"] == "CODE15"].merge(
        code15[["exam_id", "RBBB", "normal_ecg", "age"]],
        on="exam_id",
        how="left",
    )
    code["has_rbbb"] = code["RBBB"]
    parts.append(code)

    # SAMITROP
    sami = df_test[df_test["dataset_source"] == "SAMITROP"].merge(
        samitrop[["exam_id", "normal_ecg", "age"]],
        on="exam_id",
        how="left",
    )
    sami["has_rbbb"] = np.nan
    parts.append(sami)

    # PTBXL
    ptb = df_test[df_test["dataset_source"] == "PTBXL"].merge(
        ptbxl[["exam_id", "ptb_any_rbbb", "ptb_normal_ecg", "age"]],
        on="exam_id",
        how="left",
    )
    ptb["has_rbbb"] = ptb["ptb_any_rbbb"]
    ptb["normal_ecg"] = ptb["ptb_normal_ecg"]
    parts.append(ptb)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out["age_bin"] = pd.cut(
        pd.to_numeric(out["age"], errors="coerce"),
        bins=[-np.inf, 30, 50, 70, np.inf],
        labels=["<30", "30-50", "50-70", ">70"],
    ).astype(object)
    return out


def _make_stratum(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    # Use pandas string dtype to avoid fillna() downcasting warnings on mixed columns.
    tmp = tmp.astype("string[python]").fillna("NA")
    return tmp.agg("|".join, axis=1)


def _allocate_counts_by_fraction(
    total: int, fracs: dict[str, float]
) -> dict[str, int]:
    if total < 0:
        raise ValueError(f"total must be >=0, got {total}")
    if total == 0:
        return {k: 0 for k in fracs}
    items = [(k, float(v)) for k, v in fracs.items() if float(v) > 0]
    if not items:
        return {}
    s = sum(v for _k, v in items)
    if s <= 0:
        return {}
    items = [(k, v / s) for k, v in items]
    raw = {k: v * total for k, v in items}
    base = {k: int(np.floor(raw[k])) for k, _v in items}
    rem = int(total - sum(base.values()))
    if rem > 0:
        frac = sorted(((k, raw[k] - base[k]) for k in base), key=lambda kv: -kv[1])
        for k, _f in frac[:rem]:
            base[k] += 1
    return base


def _stratified_sample_indices(
    df: pd.DataFrame,
    n: int,
    *,
    strata_col: str,
    rng: np.random.Generator,
) -> list[int]:
    if n <= 0:
        return []
    if df.empty:
        return []

    counts = df[strata_col].value_counts()
    targets = _allocate_counts_by_fraction(n, counts.to_dict())

    chosen: list[int] = []
    for stratum, tgt in targets.items():
        if tgt <= 0:
            continue
        candidates = df[df[strata_col] == stratum].index.to_numpy()
        take = min(int(tgt), int(candidates.size))
        if take <= 0:
            continue
        picked = rng.choice(candidates, size=take, replace=False).tolist()
        chosen.extend(picked)

    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < n:
        remaining = df.drop(index=chosen, errors="ignore")
        need = n - len(chosen)
        if need > 0 and not remaining.empty:
            extra = rng.choice(
                remaining.index.to_numpy(), size=min(need, len(remaining)), replace=False
            ).tolist()
            chosen.extend(extra)

    if len(chosen) > n:
        chosen = rng.choice(np.array(chosen), size=n, replace=False).tolist()
    return chosen


def build_probe_index(
    df_test: pd.DataFrame,
    *,
    seed: int = 1337,
    neg_multiplier: int = 2,
    strat_cols: Sequence[str] = ("dataset_source", "has_rbbb", "normal_ecg", "age_bin"),
    neg_source_fracs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build a fixed probe subset:
      - all positives
      - `neg_multiplier` × negatives, stratified to match positive strata when possible.
    """
    df = df_test.copy()
    df["stratum"] = _make_stratum(df, strat_cols)

    if "chagas" not in df.columns:
        raise ValueError("df_test must include binary label column 'chagas'")
    pos = df[df["chagas"] == 1].copy()
    neg = df[df["chagas"] == 0].copy()
    n_pos = int(len(pos))
    n_neg_target = int(neg_multiplier) * n_pos

    rng = np.random.default_rng(int(seed))

    if neg_source_fracs is not None:
        # Allocate negatives across dataset sources (to enforce representation), then stratify
        # within each source across the remaining phenotype columns.
        available_sources = sorted(neg["dataset_source"].dropna().unique().tolist())
        fracs = {k: float(v) for k, v in neg_source_fracs.items() if k in available_sources}
        if not fracs:
            raise ValueError(
                f"neg_source_fracs provided but none match available negative sources: {available_sources}"
            )
        neg_targets = _allocate_counts_by_fraction(n_neg_target, fracs)
        # If a source runs out (n_target > available), reallocate the deficit.
        remaining_n = int(n_neg_target)
        remaining_sources = set(available_sources)
        chosen_neg_idx: list[int] = []

        within_cols = tuple(c for c in strat_cols if c != "dataset_source")
        for src in list(available_sources):
            n_src = int(neg_targets.get(src, 0))
            if n_src <= 0:
                remaining_sources.discard(src)
                continue
            df_src = neg[neg["dataset_source"] == src].copy()
            if df_src.empty:
                remaining_sources.discard(src)
                continue
            df_src["stratum_within_src"] = _make_stratum(df_src, within_cols)
            take = min(n_src, len(df_src))
            picked = _stratified_sample_indices(
                df_src, take, strata_col="stratum_within_src", rng=rng
            )
            chosen_neg_idx.extend(picked)
            remaining_n -= len(picked)
            remaining_sources.discard(src)

        chosen_neg_idx = list(dict.fromkeys(chosen_neg_idx))
        if remaining_n > 0:
            # Reallocate any remainder from all remaining negatives (across sources), still stratified.
            remaining = neg.drop(index=chosen_neg_idx, errors="ignore").copy()
            if not remaining.empty:
                remaining["stratum"] = _make_stratum(remaining, strat_cols)
                extra = _stratified_sample_indices(
                    remaining, remaining_n, strata_col="stratum", rng=rng
                )
                chosen_neg_idx.extend(extra)

        if len(chosen_neg_idx) < n_neg_target:
            raise RuntimeError(
                f"Could only sample {len(chosen_neg_idx)}/{n_neg_target} negatives; "
                "check fold availability and neg_source_fracs."
            )

        probe = pd.concat([pos, df.loc[chosen_neg_idx]], axis=0, ignore_index=True)
        probe = probe.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
        probe.insert(0, "row_idx", np.arange(len(probe), dtype=int))
        return probe[["row_idx", "exam_id", "dataset_source", "chagas"]]

    pos_counts = pos["stratum"].value_counts()
    chosen_neg_idx: list[int] = []

    # First pass: try to match positive strata.
    for stratum, c_pos in pos_counts.items():
        desired = int(neg_multiplier) * int(c_pos)
        neg_in = neg[neg["stratum"] == stratum]
        if neg_in.empty:
            continue
        candidates = neg_in.index.to_numpy()
        take = min(desired, candidates.size)
        if take <= 0:
            continue
        picked = rng.choice(candidates, size=take, replace=False)
        chosen_neg_idx.extend(picked.tolist())

    chosen_neg_idx = list(
        dict.fromkeys(chosen_neg_idx)
    )  # preserve order, ensure unique

    # Fill remainder from the remaining negatives.
    if len(chosen_neg_idx) < n_neg_target:
        remaining = neg.drop(index=chosen_neg_idx, errors="ignore")
        need = n_neg_target - len(chosen_neg_idx)
        if need > 0 and not remaining.empty:
            extra = rng.choice(
                remaining.index.to_numpy(),
                size=min(need, len(remaining)),
                replace=False,
            )
            chosen_neg_idx.extend(extra.tolist())

    # If we overshot for any reason, downsample.
    if len(chosen_neg_idx) > n_neg_target:
        chosen_neg_idx = rng.choice(
            np.array(chosen_neg_idx), size=n_neg_target, replace=False
        ).tolist()

    probe = pd.concat([pos, df.loc[chosen_neg_idx]], axis=0, ignore_index=True)
    probe = probe.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    probe.insert(0, "row_idx", np.arange(len(probe), dtype=int))
    return probe[["row_idx", "exam_id", "dataset_source", "chagas"]]


def build_probe_metadata(
    probe_index: pd.DataFrame,
    *,
    code15: pd.DataFrame,
    samitrop: pd.DataFrame,
    ptbxl: pd.DataFrame,
    master_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = probe_index.copy()

    code = out[out["dataset_source"] == "CODE15"].merge(
        code15, on="exam_id", how="left"
    )
    sami = out[out["dataset_source"] == "SAMITROP"].merge(
        samitrop, on="exam_id", how="left"
    )
    ptb = out[out["dataset_source"] == "PTBXL"].merge(
        ptbxl[
            [
                "exam_id",
                "age",
                "is_male",
                "patient_id",
                "ptb_crbbb",
                "ptb_irbbb",
                "ptb_any_rbbb",
                "ptb_lafb",
                "ptb_normal_ecg",
            ]
        ],
        on="exam_id",
        how="left",
    )
    ptb["normal_ecg"] = ptb["ptb_normal_ecg"]

    merged = pd.concat([code, sami, ptb], axis=0, ignore_index=True)
    if master_quality is not None and not master_quality.empty:
        mq = master_quality.copy()
        mq["exam_id"] = mq["exam_id"].astype(str)
        merged["exam_id"] = merged["exam_id"].astype(str)
        merged = merged.merge(mq, on="exam_id", how="left")

    merged["age"] = pd.to_numeric(merged.get("age"), errors="coerce")
    merged.loc[
        (merged["dataset_source"] == "PTBXL") & (merged["age"] > 89), "age"
    ] = 100.0
    merged["nn_predicted_age"] = pd.to_numeric(
        merged.get("nn_predicted_age"), errors="coerce"
    )
    merged["delta_age"] = merged["nn_predicted_age"] - merged["age"]

    # Keep a compact set of columns (some are dataset-specific and may be missing).
    preferred = [
        "row_idx",
        "exam_id",
        "dataset_source",
        "chagas",
        "age",
        "is_male",
        "nn_predicted_age",
        "delta_age",
        "normal_ecg",
        # CODE-specific
        "RBBB",
        "LBBB",
        "1dAVb",
        "AF",
        "SB",
        "ST",
        # PTB-specific
        "ptb_crbbb",
        "ptb_irbbb",
        "ptb_any_rbbb",
        "ptb_lafb",
        # Outcomes (CODE/SAMI)
        "death",
        "timey",
        # Extra identifiers that are small and sometimes useful
        "patient_id",
        # Quality (processedMaster)
        "qc_zhao2018_bp",
        "qc_templatematch_bp",
        "resample_method",
    ]
    keep = [c for c in preferred if c in merged.columns]
    return merged[keep].sort_values("row_idx").reset_index(drop=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
