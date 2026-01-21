# Ranking agreement & screening consensus (fold4 test set)

This document describes the **model-vs-model ranking agreement** analysis and the **per-sample screening consensus score** that are computed from saved per-run logits.

It is designed to support:

- **Global model agreement**: “Do models rank patients similarly overall?”
- **Screening overlap**: “Do models pick the same top-risk subset?”
- **Screening consistency**: “Within the overlapped screening subset, do models agree on the ranking?”
- **Consensus scoring**: “How many models put each sample into their screening set?”

All computations are done on the **held-out test fold** (`fold=4`).

## Core idea

For each model `m`, define its *screening set* `T_m` as the **top `p%`** of the test set ranked by the model score (logit or probability; rank is identical because sigmoid is monotonic).

Then for each sample `i` define the **consensus count**:

`c_i = Σ_m 1(i ∈ T_m)`

This `c_i` (or normalized to `[0,1]`) is the central object used to:

- select “high-consensus” samples for visualization / XAI,
- find “non-consensus” samples where models disagree,
- study how preprocessing affects screening selection.

## Inputs

### 1) Frozen test index (`test_index.csv`)

Location:

- `analysis/embeddings_probe/test_index.csv`

Produced by:

- `scripts/analysis/build_probe_set.py`
- (or via) `scripts/analysis/run_probe_pipeline.py`

Required columns:

- `row_idx`: integer `0..N_test-1` (fixed order)
- `exam_id`: string identifier
- `dataset_source`: `CODE15` / `PTBXL` / `SAMITROP`
- `chagas`: integer label `0/1` (ground truth)

The `row_idx` defines the shared ordering across all saved test logits.

### 2) Per-run full-test logits memmap

Location (new layout):

- `analysis/embeddings_probe/runs/<run_id>/memmap/<run_id>__logits__N{N_test}.fp32.mmap`

Produced by:

- `scripts/analysis/evaluate_test_models.py --save_logits`
- (or via) `scripts/analysis/run_probe_pipeline.py` (it triggers `--save_logits` automatically when needed)

Format:

- dtype: `float32`
- shape: `(N_test,)`
- values: raw logits (pre-sigmoid)

Notes:

- Storing logits is enough because all ranking-based metrics are invariant to a monotonic transform like sigmoid.
- If you want to inspect calibrated probabilities, compute them later as `sigmoid(logits)` when needed.

## Outputs

All ranking-agreement outputs are written under:

- `analysis/embeddings_probe/ranking_agreement/test/`

### 1) `spearman_rho.csv`

Model×model matrix with Spearman’s ρ computed over the **entire test set**:

- `ρ(m1, m2) = corr(rank(scores_m1), rank(scores_m2))`

Interpretation:

- `ρ ≈ 1`: models produce near-identical ranking globally.
- `ρ ≈ 0`: rankings are unrelated.
- `ρ < 0`: inverted ranking (rare in practice).

Implementation note:

- Computed via Pearson correlation on ordinal ranks (ties broken deterministically by stable sort).

### 2) `top5_iou.csv`

Model×model matrix with Jaccard / IoU overlap of screening sets:

- `IoU(m1, m2) = |T_m1 ∩ T_m2| / |T_m1 ∪ T_m2|`

Interpretation:

- High IoU means both models choose largely the same screening subset.
- Low IoU means screening sets differ (domain shift / preprocessing sensitivity / loss effects).

### 3) `top5_kendall_tau.csv`

Model×model matrix with Kendall’s τ computed on the **intersection** `T_m1 ∩ T_m2`.

Interpretation:

- “Given both models selected these samples, do they agree on relative ordering inside the screening subset?”

Implementation note:

- Uses τ-a (ties ignored). With continuous logits ties are rare; if ties are common for some model, τ may be less stable.

### 4) `top5_intersection_n.csv`

Model×model matrix of `|T_m1 ∩ T_m2|` (intersection sizes).

This is useful to sanity-check Kendall τ interpretation:

- if intersection is tiny, τ is noisy / may be `NaN`.

### 5) `sample_top5_consensus.csv`  ← Step 2 (your screenshot)

Per-sample consensus score (on the **full test set**):

Columns:

- `row_idx`, `exam_id`, `dataset_source`, `chagas` (if present in index)
- `top5_count_models`: integer `c_i ∈ [0, M]`
- `top5_frac_models`: `c_i / M ∈ [0,1]`

This is the file you’ll typically use to:

- pick “consensus positives” (high `top5_frac_models`) and
- pick “non-consensus” or “disagreement” samples (mid-range `top5_frac_models`).

### 6) `top5_membership__N{N_test}__M{M}.u8.mmap` (+ `top5_membership_run_ids.txt`)

Compact per-sample per-model membership matrix:

- `membership[i, m] = 1` iff sample `i` is in the top-5% screening set of model `m`.

Files:

- `top5_membership__N{N_test}__M{M}.u8.mmap` (dtype `uint8`, shape `(N_test, M)`)
- `top5_membership_run_ids.txt` (run_id order corresponding to columns)

Why a memmap?

- The full test set can be large; a dense CSV can be huge and slow.
- A memmap is fast to load and easy to slice for later selection logic.

Example loading snippet:

```python
import numpy as np
from pathlib import Path

base = Path("analysis/embeddings_probe/ranking_agreement/test")
run_ids = base.joinpath("top5_membership_run_ids.txt").read_text().splitlines()
N, M = ...  # infer from filename or keep alongside your test_index.csv

mm = np.memmap(
    base / f"top5_membership__N{N}__M{M}.u8.mmap",
    mode="r",
    dtype="uint8",
    shape=(N, M),
)

consensus_count = mm.sum(axis=1)        # [N]
consensus_frac = consensus_count / M   # [N]
```

## How “top 5%” is defined

The top-k set is computed **within the chosen set**:

- For `--set test`: top-5% of **the full test fold**
- For `--set probe`: top-5% of **the probe subset**

This is important:

- “Top-5% on the probe” is not the same as “top-5% on the full test set”.
- For screening-style analysis and XAI selection, you usually want `--set test`.

## Commands

### End-to-end (recommended)

Runs the full pipeline and **skips steps that already exist** (unless `--overwrite`):

```bash
python scripts/analysis/run_probe_pipeline.py \
  --run_specs configs/analysis/embeddings_probe_runs.toml \
  --out_dir ./analysis/embeddings_probe \
  --plots
```

This will ensure:

- `test_index.csv` exists
- full-test logits memmaps exist (will run `evaluate_test_models.py --save_logits` if missing)
- ranking agreement outputs exist under `ranking_agreement/test/`

### Ranking agreement only

If full-test logits are already saved:

```bash
python scripts/analysis/compute_ranking_agreement.py \
  --run_specs configs/analysis/embeddings_probe_runs.toml \
  --out_dir ./analysis/embeddings_probe \
  --set test
```

Optional flags:

- `--top_frac 0.05` (default is 0.05)
- `--skip_missing` (skip runs missing logits instead of failing)
- `--write_membership_csv` (usually not recommended for `--set test` because it can be large)

## Computational cost (practical notes)

Let:

- `N_test` = number of test samples in fold4 (e.g. ~73k)
- `M` = number of models (e.g. 24)

Costs:

- Loading logits: reads `M × N_test` floats from disk (fast; sequential).
- Spearman ρ:
  - per model: one `argsort` over `N_test` (O(M·N log N))
  - then matrix correlation (O(M²·N))
- Top-k IoU:
  - boolean mask creation uses `argpartition` (fast), then `M×M` dot-products (O(M²·N))
- Kendall τ (on intersections):
  - worst-case heavier because it runs a merge-sort inversion count on each pair’s intersection.
  - in practice the intersection size is usually much smaller than `N_test`.

If Kendall τ becomes the bottleneck, you can:

- reduce `--top_frac` (smaller intersections),
- or temporarily skip Kendall τ computation (can be added later).

