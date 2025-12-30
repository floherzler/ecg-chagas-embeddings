# Master Plan

## Problem Statement
- Goal: classify Chagas from 12‑lead ECG snippets, while understanding how preprocessing, losses, and augmentations shape both performance and embedding geometry.
- Data: highly imbalanced (~2% positives) with mixed label strength across sources (CODE-15 weak/self-reported vs PTB-XL / SaMi-Trop stronger but single-labeled).
- Research framing: follow Mildenberger et al. (“A Tale of Two Classes”) - treat representation geometry as a first-class object, not only downstream score.

## Global Principles (To Keep Experiments Interpretable)
- Change one axis at a time early on (loss vs preprocessing vs augmentation) to avoid confounded conclusions.
- Use the same batch size (256) and comparable training schedule across experiments; rely on early stopping to save compute.
- Prefer *linear/physiological* augmentations on linear signals (bandpassed) and avoid applying geometric transformations after nonlinear preprocessing.

## Data & Preprocessing Regimes
We maintain three cumulative preprocessing stages (stored in separate folders):
- `bp`: bandpass only (linear).
- `bp_sc`: bandpass + soft clipping (nonlinear; reduces extreme amplitudes).
- `bp_sc_norm`: bandpass + soft clipping + per-lead normalization (nonlinear + rescaling; changes inter-lead relationships).

Reasoning:
- `bp` is closest to raw physiology (still filtered) and preserves inter-lead geometry needed for VCG-based rotations.
- `bp_sc` improves robustness to outliers but breaks strict linear mixing assumptions.
- `bp_sc_norm` harmonizes scale but can remove clinically useful amplitude information and can change cross-dataset alignment.

## Augmentation Policy
### Base (Mild) Augmentations (Default for “basic experiments”)
Configured in `configs/base.yaml`:
- Amplitude scaling: `[0.98, 1.02]` (small).
- Gaussian noise: `0.005` (small).
- No time warp, no wandering, masking disabled by default.

Rationale:
- Keep augmentations “low risk” for early baselines and geometry studies.
- Avoid large distribution shifts that would dominate differences between losses / preprocessing.

### Physiological Augmentation: Frontal Axis Rotation (VCG-based)
- Only valid on `bp` signals.
- Implementation: approximate 12-lead → Frank VCG → rotate about Z (frontal plane) → back to 12-lead, then enforce limb-lead identities.
- Enabled via CLI overrides:
  - `--data.axis_rotation_max_deg <deg>`
  - `--data.axis_rotation_prob <p>`
  - `--data.per_view_axis_rotation true`

Rationale:
- This is a physiologically motivated augmentation (axis variability) and a better “rotated heart” proxy than limb-only rotations.
- Not used on `bp_sc` / `bp_sc_norm` because nonlinear transforms break the linear mixing model.

### Masking (Optional, Later Ablation)
Masking can be strong in this codebase, so I keep it off initially. If used, add stochastic gating:
- Time masking: `max_mask_duration > 0` plus `time_mask_apply_prob < 1.0`.
- Channel masking: `mask_prob > 0` plus `channel_mask_apply_prob < 1.0`.

Rationale:
- Masking changes the task (inpainting / missing-lead robustness). Use only if explicitly studying robustness.

## Oversampling Policy (Imbalance Handling)
- Default: `oversample: false` for early experiments (clean story; avoids mixing distribution-shift effects with loss effects).
- One ablation (classification): test oversampling to `pos_fraction=0.05` (2.5×) if needed.

Rationale:
- Oversampling can help recall and training stability in track1 but may hurt calibration and confound geometry.
- Track2 (SimCLR-style positives are same-sample views) is less dependent on class balance for forming positive pairs.

## Metrics & Logging (W&B)
Validation logs include:
- Challenge metric: `val/score` (`TPR@5%` style score) and `val_score`.
- Standard classification metrics: `val/auroc`, `val/ap`, `val/acc`, `val/loss`.
- Per-source metrics when sources exist: `val/code15_*` and `val/strong_*`.
- Probability distributions: histograms and class/source quartiles.
- Embedding metrics (TTC): `emb_*` (includes `CAC_0`, `CAC_1`, etc.).
- UMAP: colored by dataset and styled by label for readability.

Interpretation note:
- AP baseline ≈ prevalence (~0.02). Values like 0.07–0.10 can still be meaningful improvements over random ranking.

## Early Stopping & Checkpointing
Track 1 (`configs/track1.yaml`):
- Monitor `val_score` (mode=max, patience=10).

Track 2 (`configs/track2_sup_min.yaml`, `configs/track2_sup_proto.yaml`):
- Monitor `emb_CAC_1` (mode=max, patience=10).

Rationale:
- Track1 optimizes classification performance directly (challenge-aligned).
- Track2 optimizes representation geometry; `CAC_1` is a reasonable single-metric proxy emphasizing minority-class neighborhood purity.

## Loss Functions (Track 1 Classification)
Baselines and thesis focus:
- Weighted BCE: `configs/losses/bce_weighted.yaml` (pos_weight tuned for ~2% positives).
- Focal: `configs/losses/focal_gamma15.yaml` (start here; expand gamma only if needed).
- RAT (“Ranking-Aware Tversky”): `configs/losses/rat.yaml` using `FocalTverskyLoss`:
  - soft top‑k (`k=0.05`, `tau`)
  - Tversky + focal exponent
  - entropy regularizer to reduce overconfidence
  - BCE-with-logits anchor for stability

Entropy sign note:
- Current implementation encourages higher entropy (less overconfident probabilities), which matches “mitigate overconfidence” better than “punish p≈0.5”.

## Track Structure & Experiment Order
### Track 1 — Classification (CV)
Objective:
- Establish strong, interpretable baselines and quantify impact of physiological augmentation (axis rotation) on `bp`.

Plan (4-fold CV each):
1) `bp` + weighted BCE, mild augs (no rotation)
2) `bp` + weighted BCE, mild augs + axis rotation
3) Repeat (1–2) for Focal
4) Repeat (1–2) for RAT
5) Optional: oversampling ablation at `pos_fraction=0.05` for the best loss (no rotation), to test whether sampling materially helps.
6) Only after choosing best 1–2 losses: run preprocessing comparison (`bp` vs `bp_sc` vs `bp_sc_norm`) without axis rotation.

Scripts:
- `scripts/experiments/exp1_track1_bp_bce.sh`
- `scripts/experiments/exp2_track1_bp_bce_rotation.sh`
- `scripts/experiments/exp3_track1_bp_focal.sh`
- `scripts/experiments/exp4_track1_bp_focal_rotation.sh`
- `scripts/experiments/exp5_track1_bp_rat.sh`
- `scripts/experiments/exp6_track1_bp_rat_rotation.sh`

### Track 2 — Projection / Representation Learning
Objective:
- Learn embeddings with SimCLR-style two-view positives and analyze geometry (TTC metrics + UMAP).

Configs:
- `configs/track2_sup_min.yaml` (SupCon)
- `configs/track2_sup_proto.yaml` (prototype variant)

Plan:
1) Start with mild augs (same base) and no oversampling.
2) Select best embedding checkpoint via `emb_CAC_1` early stopping.
3) Compare preprocessing regimes only after establishing a stable baseline (to keep geometry comparisons interpretable).

### Track 3 — Classification with Pretrained Encoder
Objective:
- Quantify how much of the classification performance can be recovered from track2 embeddings.

Plan:
1) Linear probe first: `configs/track3_probe.yaml` (frozen encoder).
2) Frozen encoder head training: `configs/track3.yaml`.
3) Optional “future work”: unfreeze last block if time permits, but treat as secondary.

## Analysis Deliverables (Thesis)
- For each experiment family: report mean±std across 4 folds for `val/score`, `val/auroc`, `val/ap`.
- For geometry: report `emb_CAC_0/1`, `emb_CAD_0/1`, `emb_SAA/SAD`, plus curated UMAPs colored by dataset and marker by label.
- Discuss harmonization tradeoffs: improved alignment vs potential loss of informative amplitude cues.
