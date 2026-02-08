# Offline Preprocessing Regimes (`bp`, `bp_sc`, `bp_sc_norm`)

This project uses three *offline* preprocessing regimes that are computed once from the raw WFDB records and saved as PyTorch tensors (`.pt`). Training and evaluation then load one regime directory at a time.

The regimes are *cumulative*:

1. `bp`: bandpass filtering + resampling to 400 Hz + per-lead median centering
2. `bp_sc`: `bp` + soft clipping (smooth saturation) + per-lead amplitude scaling
3. `bp_sc_norm`: `bp_sc` + robust per-lead normalization

Implementation reference: `src/ecg_chagas_embeddings/data/prepare_dataset.py`.

## Outputs and directory layout

Given `--output_dir <OUT>`, the preprocessing script writes:

- `<OUT>/bp/*.pt`: bandpassed tensors (one file per record)
- `<OUT>/bp_sc/*.pt`: softclipped+scaled tensors
- `<OUT>/bp_sc_norm/*.pt`: softclipped+scaled+normalized tensors
- `<OUT>/softclip_bounds.npz`: softclip parameters used for `bp_sc` / `bp_sc_norm`
- `<OUT>/<output_file>` (default `metadata.csv`): metadata table including fold assignments and a pointer to the processed tensors via:
  - `processed_root`: the value of `--output_dir`
  - `proc_stem`: the WFDB record stem (used to locate `<processed_root>/<regime>/<proc_stem>.pt`)

Note: `TOTAL_EXPECTED_FILES` in code is a historical constant; the script may warn if it finds fewer records. Treat that warning as informational.

## How to run

The offline preprocessing entry point is:

`python -m ecg_chagas_embeddings.data.prepare_dataset --data_dir ... --output_dir ...`

### Record discovery

The script searches `--data_dir` recursively for WFDB records (by scanning for `*.dat` files and pairing them with their headers), then keeps only paths containing one of these substrings:

- `code15/processed/exams_part`
- `sami-trop/processedOfficial`
- `ptb-xl/processedOfficial500`

If your local dataset uses a different folder layout, adjust the `allowed_keywords` list in `src/ecg_chagas_embeddings/data/prepare_dataset.py`.

### Key arguments

Key arguments:

- `--data_dir`: root folder containing WFDB records (`.dat`/`.hea`)
- `--output_dir`: where to write `bp/`, `bp_sc/`, `bp_sc_norm/`, and `metadata.csv`
- `--sample_rate` (default `400`): target sampling frequency for all stored tensors
- `--splits` (default `5`): number of folds to create
- `--dev_folds` (default `0,1,2,3`): folds used to compute *softclip* statistics
- `--skip_existing`: skip rewriting tensor files if they already exist
- `--no_qc`: disable optional ECG quality metrics computation (see below)
- `--qc_lead` (default `1`): 0-based lead index used for QC metrics (default corresponds to Lead II in typical WFDB ordering)

Optional dataset metadata enrichment (does not change signal preprocessing, but affects `patient_id` and auxiliary columns):

- `--ptb_meta_csv`
- `--code_meta_csv`
- `--sami_meta_csv`

## Regime 1: `bp` (bandpass + resample + median centering)

### Input and representation

Each record is loaded with `wfdb.rdrecord`, and the signal is transposed to shape `(12, T)` with values in the dataset’s physical units (typically millivolts).

### Step A — Zero-phase bandpass filtering (FIR, BioSPPy/NeuroKit2-inspired)

The function name in code is `butter_filter`, but the current implementation applies a *linear-phase FIR* bandpass filter designed by `scipy.signal.firwin` and applied with `scipy.signal.filtfilt` (forward-backward filtering), yielding **zero-phase** distortion.

Exact parameters:

- High-pass cutoff: **0.67 Hz**
- Low-pass cutoff: **45.0 Hz**
- Tap-length factor: **`order = 1.5`**, used as `desired_taps = int(order * fs)`
  - This follows the BioSPPy/NeuroKit2 convention of using a filter length of approximately **1.5 seconds**.
- Per-record adaptation for short signals:
  - `filtfilt` requires the signal length `n` to be longer than the padding length; the implementation caps the tap count as
    - `max_taps = floor((n - 1) / 3)` and then `taps = min(desired_taps, max_taps)`
  - `taps` is forced to be odd and at least 3; very short records raise an error.

Justification for the **0.67 Hz** high-pass cutoff:

- This cutoff was adopted because it matches the BioSPPy setting exposed through NeuroKit2’s ECG filtering defaults during experimentation, where it is described as a physiologically motivated choice to attenuate baseline wander / very-low-frequency drift while preserving relevant ECG morphology.

### Step B — Resampling to a common sampling rate (default 400 Hz)

To make signals comparable across sources, all records are resampled to `target_sample_rate = 400 Hz`.

Resampling strategy:

- If the original sampling rate already equals 400 Hz: no resampling.
- If the ratio between original and target sampling rates is an integer (either direction): use **polyphase resampling** (`scipy.signal.resample_poly`).
- Otherwise: use **FFT resampling** (`scipy.signal.resample`) with a consistency check that the achieved sampling rate differs by < 0.5 Hz.

The resampling method used is recorded in metadata as `resample_method`.

### Step C — Per-lead median subtraction (baseline centering)

After filtering and resampling, each lead is centered by subtracting its median:

- For each lead `i`: `x_i ← x_i - median(x_i)`

This removes residual DC offsets and baseline shifts in a robust way.

### Saved artifact

The result of regime `bp` is saved as a float tensor at:

`<OUT>/bp/<record_stem>.pt`

During creation, per-record per-lead percentiles are computed and stored temporarily in the metadata table:

- `p1`: 1st percentile per lead (shape `(12,)`)
- `p99`: 99th percentile per lead (shape `(12,)`)

These percentiles are used later to derive dataset-level softclip bounds, and are dropped from the final `metadata.csv`.

### Optional: QC metrics on `bp`

Unless `--no_qc` is passed, two NeuroKit2 quality metrics are computed on one lead (default lead index `1`):

- `qc_zhao2018_bp`: categorical label from `nk.ecg_quality(..., method="zhao2018")`
- `qc_templatematch_bp`: numeric score aggregated by `nanmedian` from `nk.ecg_quality(..., method="templatematch")`

These metrics do not change the saved signals; they are recorded for later analysis/filtering.

## Fold assignment (used for softclip statistics and downstream CV)

Folds are created after metadata enrichment. The script assigns `fold ∈ {0, …, splits-1}` using `StratifiedGroupKFold`:

- Stratification label: `str(chagas) + "_" + str(source_code)` (i.e., stratify jointly by class label and dataset source)
- Grouping: `patient_id` if available (otherwise falls back to `exam_id`)

This aims to (1) keep class/source proportions balanced across folds and (2) reduce leakage by keeping patient-linked exams together.

## Regime 2: `bp_sc` (soft clipping + amplitude scaling)

`bp_sc` starts from the already-written `bp` tensors.

### Step A — Compute dataset-level softclip bounds (train folds only)

Soft clipping uses per-lead bounds learned from the development folds specified by `--dev_folds` (default `0,1,2,3`).

Computation details:

1. Collect all `p1` and `p99` vectors from records whose `fold ∈ dev_folds`.
2. For each lead independently:
   - `lower_raw = percentile(p1, 5th)`
   - `upper_raw = percentile(p99, 95th)`
3. Enforce symmetric bounds around 0 by taking:
   - `T = max(|lower_raw|, upper_raw)`
   - `lower = -T`, `upper = +T`
4. Define a per-lead scale parameter:
   - `c = (upper - lower) / 2`

These arrays are saved to `<OUT>/softclip_bounds.npz` along with `train_folds` and `sample_rate`.

Leakage/validation note (as used in this project):

- With `splits=5` and `dev_folds=0,1,2,3`, the softclip bounds are estimated without using fold 4.
- This matches the intended experimental setup where folds 0–3 are used for cross-validation/model selection and fold 4 is reserved as a held-out evaluation fold.
- If you need *per-CV-split* “train-only” softclip bounds (i.e., recompute bounds for each CV run’s training folds), that requires a different preprocessing strategy than the current single global bounds file.

### Step B — Apply soft clipping (smooth saturation) per lead

Each lead is transformed by a smooth, differentiable saturation function based on `softplus` / `softminus`:

- Lower-bound effect: values below `a` are smoothly pushed upward
- Upper-bound effect: values above `b` are smoothly pushed downward

Bounds are per-lead: `a = lower[i]`, `b = upper[i]`, `c = c[i]`.

The implementation uses:

- `softplus(x) = log(1 + exp(-|x|)) + max(x, 0)`
- `softminus(x) = -softplus(-x)`
- `softclip(v; a,b,c)` defined by applying a smooth lower and upper correction to `v`

Important detail (exactly as implemented):

- When `a` and `b` are provided, the function internally rescales the softness parameter as
  - `c_eff = c / ((b - a) / 2)`
- In this pipeline, `c` is set to `(upper - lower) / 2`, so `c_eff = 1` for every lead.

This behaves like a differentiable approximation of clipping, avoiding hard discontinuities.

### Step C — Scale amplitudes after clipping

After soft clipping, each lead is divided by its bound magnitude:

- `scale[i] = max(|a[i]|, |b[i]|)` (with a guard `scale=1` if zero)
- `y_i ← softclip(x_i) / scale[i]`

This converts the per-lead bounds into an approximate unit scale, so extreme amplitudes are damped and overall gain variability is reduced.

### Saved artifact

`bp_sc` tensors are written to:

`<OUT>/bp_sc/<record_stem>.pt`

## Regime 3: `bp_sc_norm` (robust per-lead normalization)

`bp_sc_norm` starts from the already-written `bp_sc` tensors.

For each record and lead, it applies **per-sample robust normalization**:

- `median_i = median(x_i)`
- `IQR_i = percentile_75(x_i) - percentile_25(x_i)`
- `IQR_i ← max(IQR_i, 1e-6)` (guard against flat leads)
- `y_i = (x_i - median_i) / IQR_i`

This removes remaining baseline offsets and rescales each lead by a robust spread measure.

### Saved artifact

`bp_sc_norm` tensors are written to:

`<OUT>/bp_sc_norm/<record_stem>.pt`

## Practical interpretation of the regimes

- `bp` preserves relative lead amplitudes and is the most “linear” representation (after filtering/resampling/centering).
- `bp_sc` reduces the influence of rare extreme amplitudes and stabilizes lead scales using dataset-derived bounds.
- `bp_sc_norm` removes most amplitude information by enforcing a robust, per-record per-lead scale, which can improve cross-dataset alignment but may discard clinically meaningful amplitude cues.
