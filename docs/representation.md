# Track 2 — Representation Learning (Supervised Contrastive / Prototype)

This document describes the *training procedure* for Track 2 (learning ECG embeddings) as implemented in this repository. It assumes you already created the offline preprocessed tensors and `metadata.csv` described in `docs/prepro.md`.

Track 2 uses the same backbone network as Track 1, but replaces the training objective with a contrastive/prototype objective computed on two augmented views of the same ECG.

## Entry point and configuration composition

Training is launched via LightningCLI:

`python main.py fit --config configs/base.yaml --config <track2-config> --config <preproc-config> ...`

Where:

- `configs/base.yaml`: shared trainer/model/data defaults
- `configs/track2_sup_standard.yaml`: supervised contrastive (“standard”)
- `configs/track2_sup_min.yaml`: supervised contrastive with reduced majority‑class positives (“imbalance-aware”)
- `configs/track2_sup_proto.yaml`: prototype-based variant (still contrastive internally)
- `configs/preproc/*.yaml`: preprocessing‑regime specific amplitude scaling

The Slurm convenience scripts in `scripts/experiments/track2/` combine these configs and inject dataset paths + fold splits.

## Data inputs

### Processed signals

Track 2 consumes offline‑processed tensors from one of the regime directories:

- `bp/`, `bp_sc/`, or `bp_sc_norm/` (see `docs/prepro.md`)

At runtime you point `data.data_dir` to the regime folder and `data.meta_path` to `metadata.csv`.

### Cross-validation folds and split convention

The experiment scripts implement 4-fold cross‑validation over folds `{0,1,2,3}`:

- Split 0: train `[0,1,2]`, val `[3]`
- Split 1: train `[1,2,3]`, val `[0]`
- Split 2: train `[2,3,0]`, val `[1]`
- Split 3: train `[3,0,1]`, val `[2]`

Fold `4` is reserved for a final held‑out evaluation in the overall project design.

The provided Track‑2 scripts do not automatically run a “train on 0–3, evaluate on 4” job, but you can do so by setting:

- `--data.train_folds [0,1,2,3]`
- `--data.valid_folds [4]`

## Online (training-time) augmentation and “two-view” protocol

Track 2 relies on two augmented “views” of each ECG. The dataloader produces `ecg_views` with shape `[B,2,C,T]` for both training and validation.

### Cropping and window length

Signals are cropped (or zero‑padded) to:

- `data.crop_size: 2500` samples (default), i.e. 6.25 seconds at 400 Hz.

### Default mild augmentations (from `configs/base.yaml`)

With the defaults in `configs/base.yaml` (unless overridden):

- Per-view amplitude scaling: `data.scaling: [0.98, 1.02]` (often overridden by `configs/preproc/*.yaml`)
- Per-view Gaussian noise: `data.gaussian_noise_std: 0.005`
- No axis rotation (`data.axis_rotation_max_deg: 0`, `data.axis_rotation_prob: 0`)
- No time warp (`data.max_time_warp: 0`)
- No masking (`data.mask_prob: 0`, `data.max_mask_duration: 0`)
- No wandering baseline (`data.wandering_max_amplitude: 0`)

### Regime-specific scaling (from `configs/preproc/*.yaml`)

The preprocessing regime configs override `data.scaling`:

- `configs/preproc/bp.yaml`: `scaling: [0.95, 1.05]`
- `configs/preproc/bp_sc.yaml`: `scaling: [0.975, 1.025]`
- `configs/preproc/bp_sc_norm.yaml`: `scaling: [1.0, 1.0]` (disabled)

### Optional: physiological frontal axis rotation (VCG-based)

Some Track‑2 scripts enable axis rotation (e.g. `scripts/experiments/track2/exp02_bp_sup_standard_rot.sh`), setting:

- `data.axis_rotation_max_deg: <ROT_DEG>` (default in scripts: 10°)
- `data.axis_rotation_prob: 1.0`
- `data.per_view_axis_rotation: true`

This augmentation is implemented as an approximate 12‑lead → Frank XYZ VCG transform, a rotation about the Z axis (frontal plane), and a reconstruction back to 12 leads, with limb‑lead identities enforced.

Because this augmentation assumes *linear mixing*, it is typically applied only with the `bp` regime (not after nonlinear soft clipping / normalization).

### Validation determinism

Validation uses deterministic augmentations per sample:

- view 0 is an “anchor” view; by default it is only cropped (`val_anchor_clean: true`)
- view 1 is deterministically augmented using a seed derived from `(augmentation_base_seed, exam_id, view_index)`

This stabilizes representation metrics across epochs.

## Model and objectives

### Backbone and projection head

The model is a 1D ResNet‑18 variant (`src/ecg_chagas_embeddings/models/resnet18_ecg_flex.py#LitResNet18`).

For Track 2:

- `model.track: 2` ensures the classifier loss is disabled (`use_classifier = false`).
- A 2-layer MLP projection head is enabled when `use_sup_con=true` or `use_prototypes=true`.
- The training loss is computed on L2-normalized projected embeddings shaped `[B,2,D]`.

### Objective A — Supervised contrastive loss (SupCon)

Track‑2 SupCon variants are configured via:

- `configs/track2_sup_standard.yaml`:
  - `use_sup_con: true`
  - `ratio_supervised_majority: -1.0` meaning **standard SupCon**: all same-class pairs are positives.
- `configs/track2_sup_min.yaml`:
  - `use_sup_con: true`
  - `ratio_supervised_majority: 0.0` meaning **no majority–majority positive pairs** are included in the supervised numerator (intended to reduce majority dominance in heavy imbalance).

In both cases, the loss uses:

- temperature: `model.sup_con_temp: 0.07` (default in `configs/base.yaml`)
- contrast mode: all views are anchors (implemented in `SupConLoss(..., contrast_mode="ALL_VIEWS")`)

### Objective B — Prototype variant

`configs/track2_sup_proto.yaml` sets:

- `use_prototypes: true`
- `use_sup_con: false`

This uses `ConSupPrototypeLoss` with two class prototypes in embedding space. Prototypes are initialized at fit start (and can be set programmatically by the module). Labels are passed as one‑hot vectors.

## Optimization (defaults)

From `configs/base.yaml` (unless overridden):

- optimizer: AdamW (`model.optimizer: adamw`)
- learning rate: `model.lr: 5e-4`
- scheduler: One‑Cycle (`model.lr_scheduler: one_cycle`) with:
  - `one_cycle_pct_start: 0.3`
  - `one_cycle_div_factor: 25`
  - `final_div_factor: 10000`

Trainer defaults:

- max epochs: 100 (`trainer.max_epochs`)
- precision: bf16 mixed (`trainer.precision: bf16-mixed`)

## Validation metrics, checkpointing, and early stopping

### Representation (TTC) metrics

At the end of each validation epoch, Track‑2 runs compute “TTC-style” representation metrics from the validation projections with two views per sample. In code (`compute_representation_metrics`), it:

- L2-normalizes embeddings
- builds pairwise cosine distances
- computes per-class metrics, including:
  - `emb_SAD_0`, `emb_SAD_1` (sample alignment distance)
  - `emb_SAA_0`, `emb_SAA_1` (sample alignment accuracy)
  - `emb_CAD_0`, `emb_CAD_1` (class alignment distance)
  - `emb_CAC_0`, `emb_CAC_1` (class alignment consistency)
  - `emb_GPU_0`, `emb_GPU_1` (Gaussian potential uniformity)

It also logs:

- `emb_CAC_mean = 0.5 * (CAC_0 + CAC_1)`

### Smoothed monitor for early stopping

Track‑2 configs use a rolling mean of `emb_CAC_mean` over 3 epochs:

- source: `emb_CAC_mean`
- target: `emb_CAC_mean3`

This is implemented by `ecg_chagas_embeddings.callbacks.rolling_mean_metric.RollingMeanMetric`.

### Checkpoints and early stopping (Track 2 configs)

All Track‑2 configs:

- checkpoint: monitors `emb_CAC_mean3` (max), saves top‑1 to `/tmp/ckpts`
- early stopping: monitors `emb_CAC_mean3` (max), patience 10

### Classification metrics during Track 2

Even though Track 2 does not optimize a classifier loss, the model still produces a classification logit head, and the validation loop computes:

- challenge score (`val/score`)
- AUROC/AP/pAUC@0.05

These are best interpreted as *diagnostics* (how separable the learned embeddings appear under the current head), not as the optimized objective for Track 2.

## UMAP logging (optional)

If `model.log_umap: true` (default), the code periodically logs a 2D UMAP projection of a balanced subset of validation embeddings to W&B. This is intended for qualitative assessment of dataset alignment and class separation.

UMAP frequency is controlled by:

- `model.umap_log_every_n_epochs` (default 5)
- `model.umap_log_first_n_epochs` (default 9)

## Reproducible commands (examples)

Run a single split locally (SupCon standard, no axis rotation):

`python main.py fit --config configs/base.yaml --config configs/preproc/bp.yaml --config configs/track2_sup_standard.yaml --data.meta_path <.../metadata.csv> --data.data_dir <.../bp> --data.train_folds [0,1,2] --data.valid_folds [3]`

For Slurm, use the provided scripts in `scripts/experiments/track2/` and submit an array job as annotated at the top of each script.
