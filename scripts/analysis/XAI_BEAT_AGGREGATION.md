# Beat-Level Normalization & Aggregation for ST-DFT-LRP (ECG)

This document defines how to compute **physiology-aligned relevance aggregates**
(P / QRS / T / Between × frequency bands) from **ST-DFT-LRP** outputs using **NeuroKit2**
delineation and **window-based relevance**.

It complements:
- `src/ecg_chagas_embeddings/notebooks/xai_probe_sanity.ipynb` (single-sample motivation + sanity checks)
- `scripts/analysis/compute_stdftlrp_beat_aggregates.py` (batch computation across many samples)

---

## Inputs (per sample)

- ECG signal (single lead for delineation + STDFT-LRP):
  - `x[0, lead] ∈ ℝ^T`, sampled at `fs = 400 Hz`
- ST-DFT-LRP heatmap:
  - `R[m, k]` with shape `[n_windows, n_freq_bins]`
  - `m` indexes time windows, `k` indexes RFFT frequency bins

### Windowing constraint (important)

To keep the mapping between signal time and time-frequency windows defensible, we use:

- window width `H` samples
- hop `D = H` (non-overlapping windows)
- window `m` covers samples `[m*H, (m+1)*H)`

**Note about the upstream submodule:** the `external/dft-lrp` implementation uses a *shift factor*.
The effective hop is:

`D = H // window_shift_factor`

So to obtain `D = H`, set `window_shift_factor = 1`.

---

## NeuroKit2 delineation (time-domain)

We delineate ECG morphology on a cleaned lead signal:

```python
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
signals_peaks, info_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
rpeaks = info_peaks["ECG_R_Peaks"]

signal_dwt, waves_dwt = nk.ecg_delineate(
    ecg_cleaned,
    rpeaks,
    sampling_rate=fs,
    method="dwt",
)
waves_df = pd.DataFrame({k: pd.Series(v) for k, v in waves_dwt.items()})
```

The relevant indices (per beat `b`) are:

- `P_on, P_off`
- `R_on, R_off`
- `T_on, T_off`

Beats with missing indices can be skipped. We then construct **disjoint** segments using midpoint rules (handles overlaps and gaps):

- **P**: `P_on → mid(P_off, R_on)`
- **QRS**: `mid(P_off, R_on) → mid(R_off, T_on)`
- **T**: `mid(R_off, T_on) → T_off`
- **Between**: `T_off → next P_on` (if next beat exists)

This yields full coverage from the first `P_on` to the last `T_off` while assigning each time bin to exactly one structure.

---

## Mapping time segments → STDFT windows

Given a segment `S = [t_on, t_off]` (inclusive indices), compute which windows overlap:

- `m0 = floor(t_on / H)`
- `m1 = floor(t_off / H)`

### Overlap-weighted assignment

For each overlapping window `W_m = [mH, (m+1)H)`:

```
overlap = max(0, min(W_m.end, S.end) - max(W_m.start, S.start))
w_{m,S} = overlap / H
```

This produces weights `w ∈ (0, 1]` and avoids boundary artifacts.

---

## Frequency bands (fixed)

We use fixed, physiology-inspired bands:

```python
FREQ_BANDS = {
  "low":   (0.67, 4.0),
  "mid":   (4.0, 12.0),
  "high":  (12.0, 25.0),
  "vhigh": (25.0, 45.0),
}
```

Frequency bins are from RFFT:

```python
freqs = np.fft.rfftfreq(T, d=1/fs)
```

---

## Beat-level 4×3 matrix

For beat `b`, segment `S ∈ {P, QRS, T, Between}`, band `B`:

```
R_{b,S,B} = Σ_m w_{m,S} · Σ_{k ∈ B} R[m,k]  (signed)
```

This yields a 4×4 matrix per beat:

```
        low   mid   high   vhigh
P
QRS
T
Between
```

---

## Per-beat normalization (required)

To remove scale effects (e.g., logit magnitude differences), normalize within each beat:

```
R̂_{b,S,B} = R_{b,S,B} / Σ_{S,B} |R_{b,S,B}|
```

Now each beat matrix is in [-1,1] and sums to 1 in **absolute mass** (signed relevance preserved).

---

## Sample-level aggregation (across beats)

For a sample with beats `b = 1..N`:

```
R̂_{sample,S,B} = mean_b ( R̂_{b,S,B} )  # signed, beat-normalized
```

This yields one normalized 4×4 matrix per sample, suitable for cross-sample comparisons.

---

## Lead aggregation (optional but recommended)

If multiple leads are available, compute a per-lead 4×4 matrix first, then aggregate:

- Unweighted mean across leads:
  `M_mean = mean_l M_lead[l,:,:]`
- Mass-weighted mean (preferred):
  `M_weighted = sum_l w[l]·M_lead[l,:,:] / (sum_l w[l] + eps)`

You can define `w[l]` as either:
- time-domain relevance mass per lead: `sum_t |R_time[l,t]|`, or
- time–frequency relevance mass per lead: `sum_{w,k} |R_tf[l,w,k]|`.

---

## Batch computation script

Use:

`scripts/analysis/compute_stdftlrp_beat_aggregates.py`

Inputs:
- checkpoint (`--checkpoint`)
- fold-4 exam list (`--exam_ids_csv`, must have `exam_id` column)
- preprocessed tensor dir matching the model preprocessing (`--data_dir`)

Output:
- per-sample CSV with `rel_mean_{P,QRS,T,Between}_{low,mid,high,vhigh}` and
  `rel_weighted_{P,QRS,T,Between}_{low,mid,high,vhigh}` columns plus metadata fields.
- per-sample lead importance columns: `p_lead_0..p_lead_11` and `lead_mass_0..lead_mass_11`.
- per-sample frequency summaries: `freq_rel_mean_{low,mid,high,vhigh}` and
  `freq_rel_weighted_{low,mid,high,vhigh}` (signed balance across frequency bands).
