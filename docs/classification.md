# Track 1 — Classification Training

This document describes the *training procedure* for Track 1 (binary classification: Chagas vs non‑Chagas) as implemented in this repository. It assumes you already created the offline preprocessed tensors and `metadata.csv` described in `docs/prepro.md`.

## Entry point and configuration composition

Training is launched via LightningCLI:

`python main.py fit --config configs/base.yaml --config configs/track1.yaml --config <loss-config> --config <preproc-config> ...`

Where the configuration is typically composed from:

- `configs/base.yaml`: shared trainer/model/data defaults
- `configs/track1.yaml`: track‑1 specific callbacks and toggles
- `configs/losses/*.yaml`: classification loss choice and its hyperparameters
- `configs/preproc/*.yaml`: preprocessing‑regime specific amplitude scaling

The Slurm convenience scripts in `scripts/experiments/track1/` combine these configs and inject dataset paths + fold splits.

## Data inputs

### Processed signals

Track 1 consumes offline‑processed tensors from one of the regime directories:

- `bp/`, `bp_sc/`, or `bp_sc_norm/` (see `docs/prepro.md`)

At runtime you point `data.data_dir` to the regime folder and `data.meta_path` to `metadata.csv`.

### Cross-validation folds and split convention

The experiment scripts implement 4-fold cross‑validation over folds `{0,1,2,3}`:

- Split 0: train `[0,1,2]`, val `[3]`
- Split 1: train `[1,2,3]`, val `[0]`
- Split 2: train `[2,3,0]`, val `[1]`
- Split 3: train `[3,0,1]`, val `[2]`

Fold `4` is reserved for a final held‑out evaluation in the overall project design; the provided Track‑1 scripts do not automatically run a “train on 0–3, evaluate on 4” job, but you can do so by setting:

- `--data.train_folds [0,1,2,3]`
- `--data.valid_folds [4]`

## Online (training-time) preprocessing / augmentation

All training and validation samples are cropped (or zero‑padded) to a fixed window:

- `data.crop_size: 2500` samples (default), i.e. 6.25 seconds at 400 Hz.

Augmentations are implemented in `src/ecg_chagas_embeddings/data/augmentation.py` and wired up in `src/ecg_chagas_embeddings/data/dataset.py#get_train_val_loaders`.

Important implementation detail:

- The dataloader always produces **two views** (`ecg_views` with shape `[B,2,C,T]`) for both training and validation.
- For **Track 1**, the classification loss is computed only on **view 0** (`ecg_views[:,0]`) in `LitResNet18.training_step`.
  - View 1 is still generated because the same dataloader/augmentation codepath is shared with Track 2 and because embeddings/UMAP diagnostics can use multiple views.

### Default mild augmentations (from `configs/base.yaml`)

With the defaults in `configs/base.yaml` (unless overridden):

- Per-view amplitude scaling: `data.scaling: [0.98, 1.02]`
- Per-view Gaussian noise: `data.gaussian_noise_std: 0.005`
- No axis rotation (`data.axis_rotation_max_deg: 0`, `data.axis_rotation_prob: 0`)
- No time warp (`data.max_time_warp: 0`)
- No masking (`data.mask_prob: 0`, `data.max_mask_duration: 0`)
- No wandering baseline (`data.wandering_max_amplitude: 0`)

### Regime-specific scaling (from `configs/preproc/*.yaml`)

The preprocessing regime configs override `data.scaling` to reflect the intended amplitude semantics of each offline regime:

- `configs/preproc/bp.yaml`: `scaling: [0.95, 1.05]`
- `configs/preproc/bp_sc.yaml`: `scaling: [0.975, 1.025]`
- `configs/preproc/bp_sc_norm.yaml`: `scaling: [1.0, 1.0]` (disabled)

### Optional: physiological frontal axis rotation (VCG-based)

Some Track‑1 scripts enable axis rotation (e.g. `scripts/experiments/track1/exp02_bp_bce_rot.sh`), setting:

- `data.axis_rotation_max_deg: <ROT_DEG>` (default in scripts: 10°)
- `data.axis_rotation_prob: 1.0`
- `data.per_view_axis_rotation: true`

This augmentation is implemented as an approximate 12‑lead → Frank XYZ VCG transform, a rotation about the Z axis (frontal plane), and a reconstruction back to 12 leads, with limb‑lead identities enforced.

Because this augmentation assumes *linear mixing*, it is typically applied only with the `bp` regime (not after nonlinear soft clipping / normalization).

### Validation determinism

Validation uses deterministic augmentations per sample:

- view 0 is an “anchor” view; by default it is only cropped (`val_anchor_clean: true`)
- view 1 is deterministically augmented using a seed derived from `(augmentation_base_seed, exam_id, view_index)`

This makes validation metrics stable across epochs even though views are used.

## Model and optimization

### Architecture

The model is a 1D ResNet‑18 variant (`src/ecg_chagas_embeddings/models/resnet18_ecg_flex.py#LitResNet18`) with:

- input channels: 12 leads (`model.channels: 12`)
- final head: single logit (`model.num_classes: 1`), interpreted with a sigmoid for probabilities

For Track 1, `use_sup_con=false` and `use_prototypes=false`, so:

- the “projection head” is an identity mapping (embeddings used for logging are encoder features)
- the training objective is purely the classification criterion

### Optimizer and LR schedule (defaults)

From `configs/base.yaml`:

- optimizer: AdamW (`model.optimizer: adamw`)
- learning rate: `model.lr: 5e-4`
- scheduler: One‑Cycle (`model.lr_scheduler: one_cycle`) with:
  - `one_cycle_pct_start: 0.3`
  - `one_cycle_div_factor: 25`
  - `final_div_factor: 10000`

Trainer defaults:

- max epochs: 100 (`trainer.max_epochs`)
- precision: bf16 mixed (`trainer.precision: bf16-mixed`)

## Loss functions (Track 1)

Loss is configured via `configs/losses/*.yaml` and passed through LightningCLI.
Common Track‑1 choices in this repo:

- Weighted BCE: `configs/losses/bce_weighted.yaml` (`pos_weight: 49.0`)
- Focal loss: `configs/losses/focal_gamma15.yaml` (`alpha: 0.75`, `gamma: 1.5`)
- RAT / Focal‑Tversky: `configs/losses/rat.yaml` (soft top‑k with `k=0.05`, `tau=0.1`, plus auxiliary terms)

## Metrics, checkpointing, and early stopping

During validation, the model computes:

- Challenge score (`val/score` and `val_score`): `compute_challenge_score` = TPR when selecting the top `fraction_capacity=0.05` of samples by predicted probability (with tie-handling via permutation averaging).
- Standard classification metrics:
  - `val/auroc`
  - `val/ap` (average precision)
  - `val/pauc5` (pAUC for FPR ∈ [0, 0.05], normalized to [0,1])

Track‑1 callbacks (`configs/track1.yaml`):

- Rolling mean of AP over the last 3 validation epochs:
  - source: `val/ap`
  - target: `val/ap_mean3`
- ModelCheckpoint:
  - monitors `val_score` (max), saves top‑1 to `/tmp/ckpts`
- EarlyStopping:
  - monitors `val/ap_mean3` (max), patience 10

## Reproducible commands (examples)

Run a single split locally by specifying the fold lists explicitly:

`python main.py fit --config configs/base.yaml --config configs/preproc/bp.yaml --config configs/track1.yaml --config configs/losses/bce_weighted.yaml --data.meta_path <.../metadata.csv> --data.data_dir <.../bp> --data.train_folds [0,1,2] --data.valid_folds [3]`

For Slurm, use the provided scripts in `scripts/experiments/track1/` and submit an array job as annotated at the top of each script.

