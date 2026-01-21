# XAI sample selection from screening consensus (fold4)

This guide describes how to select samples for **DFT‑LRP / time‑frequency attribution** using **model screening consensus** on the **held‑out test set** (fold4).

The goal is to construct small, defensible sample sets (e.g. 6–9 ECGs for thesis figures) that:

- reflect *stable* model behavior (agreement across models),
- surface *interesting disagreements* (model sensitivity / domain shift),
- avoid trivial redundancy (e.g. repeated exams from the same patient),
- keep dataset and quality context attached to every chosen sample.

This is intentionally conservative: it is not an “error analysis” pipeline; it is a *model–label / model–model disagreement* pipeline.

## What you already have

From the probe pipeline you already produce:

- `analysis/embeddings_probe/test_index.csv` — frozen fold4 test set index (row order is fixed)
- per‑run full‑test logits memmaps:
  - `analysis/embeddings_probe/runs/<run_id>/memmap/<run_id>__logits__N{N_test}.fp32.mmap`

> Rankings are computed from logits; probabilities are not required because sigmoid is monotonic.

## Consensus definition (Step 2)

For a model `m`, define its screening set `T_m` as the **top `K%`** of the test set by the model score.

For each sample `i`, define the **consensus count**:

`c_i = Σ_m 1(i ∈ T_m)`

and optionally the **consensus fraction**:

`c_i / M ∈ [0,1]` where `M` is the number of models used for consensus.

This `(c_i, c_i/M)` is the core “agreement axis” used for sample selection.

## Why we use an allowlist (model pool)

Consensus is only meaningful if the pool of models is meaningful.

This workflow uses an explicit **allowlist** of `run_id`s:

- you can include only “good” models (by your own criteria),
- you can intentionally include one “bad” model as a stress test,
- you can exclude degenerate/collapsed runs that would dilute consensus.

The selection script does *not* try to guess your model quality thresholds.

## Patient de-duplication

For XAI figures you generally do **not** want multiple ECGs from the same patient:

- it artificially narrows variety,
- it can overemphasize patient-specific artifacts.

The selection script enforces uniqueness using:

- `patient_id` when available (CODE15 / PTB‑XL),
- otherwise it falls back to `exam_id` (e.g. SaMi‑Trop if `patient_id` is not available).

## Quality metrics: kept, but not enforced harshly

You indicated the QC metrics can be preprocessing-dependent. The selection script therefore:

- **keeps** `qc_zhao2018_bp` (categorical) and `qc_templatematch_bp` (continuous) in outputs
- does **not** hard-filter them by default
- tries to include a **mix of `qc_zhao2018_bp` categories** in candidate sets (round‑robin), so you can inspect them manually

## Script: `select_xai_samples.py`

This script computes consensus on **fold4 test** and writes two tables:

1) a full per‑sample table for all test samples (easy to filter interactively)
2) a “medium‑sized” candidate table sampled from several selection groups

### Command

```bash
python scripts/analysis/select_xai_samples.py \
  --run_specs configs/analysis/embeddings_probe_runs.toml \
  --out_dir ./analysis/embeddings_probe \
  --allowlist ./analysis/embeddings_probe/xai_model_allowlist.txt \
  --top_frac 0.05
```

Allowlist format (`xai_model_allowlist.txt`):

```text
# one run_id per line
t1-exp01-bp-bce
t1-exp02-bp-bce-rot10
...
```

### What it writes

All outputs go to:

- `analysis/embeddings_probe/xai/`

Files:

- `allowlist_run_ids.txt`
  - the exact model pool used (after skipping missing models if enabled)
- `top5_membership__N{N_test}__M{M}.u8.mmap`
  - per‑sample per‑model membership matrix (1 iff sample is in model’s top‑K%)
- `test_consensus_full.csv`
  - per‑sample table for *all* fold4 test samples, including:
    - identifiers: `row_idx`, `exam_id`, `dataset_source`, `patient_id`, `patient_key`
    - label/metadata: `chagas`, `age`, `sex`, `delta_age`, `RBBB`, … (where available)
    - QC: `qc_zhao2018_bp`, `qc_templatematch_bp`, `resample_method`
    - consensus: `top5_count_models`, `top5_frac_models`
    - pooled score stats across the model pool:
      - `mean_prob_models`, `mean_logit_models`, `std_logit_models`
- `test_consensus_candidates.csv`
  - “medium-sized” candidate list (unique patients) with a `group` column (see below)
- `test_consensus_candidates_summary.csv`
  - quick counts by `group × dataset_source × chagas`

### Candidate groups

The candidate table is sampled from these groups:

- `high_consensus__chagas1`
  - label positive samples ranked by **highest** consensus
- `high_consensus__chagas0`
  - label negative samples ranked by **highest** consensus
- `low_consensus__chagas1`
  - label positive samples ranked by **lowest** consensus (model–label disagreement candidates)
- `low_consensus__chagas0`
  - label negative samples ranked by **lowest** consensus
- `disagreement__any_label`
  - samples with consensus near 0.5 (configurable band) and high score dispersion
  - prioritizes high `std_logit_models` (models disagree most on these)

These are “label-aware views on top of a label-agnostic consensus axis”.

### Parameters you may want to tune later

- `--candidates_per_group` (default 200): how many rows to sample per group
- `--disagreement_band` (default 0.15): defines the “near‑0.5 consensus” window
- later (not implemented here on purpose): your own `α, β` thresholds to define three sets cleanly

## How you use these outputs for XAI

1) Run `select_xai_samples.py` with your chosen allowlist.
2) Browse `analysis/embeddings_probe/xai/test_consensus_candidates.csv`:
   - pick 2–3 samples from:
     - high consensus + label positive,
     - low consensus + label positive,
     - disagreement group,
   - try to cover multiple datasets where feasible (given label constraints),
   - optionally include a mix of QC categories.
3) Create a small “final selection” file (you can do this manually in a spreadsheet):
   - keep `row_idx`, `exam_id`, and (optional) a note column like `why_selected`
4) Run your DFT‑LRP analysis on that final selection set.

## Notes / limitations

- The selection script computes **pooled** score stats across the allowlisted models.
  - This is for browsing; it is not a calibrated ensemble.
- `patient_id` is not available for all datasets in the same way.
  - The script falls back to `dataset_source:exam_id` when needed.
- If you want to select “consensus negatives / non-consensus” using `α, β`:
  - you already have `top5_frac_models`; apply your thresholds later without recomputing.

