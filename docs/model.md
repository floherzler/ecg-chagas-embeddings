# Model Architecture — `LitResNet18` (`resnet18_ecg_flex`)

This repository’s core model is implemented in `src/ecg_chagas_embeddings/models/resnet18_ecg_flex.py` as `LitResNet18`.

The file name includes “flex” because it originated from iterative experimentation during the PhysioNet Challenge development: the same codebase supported swapping blocks, normalization layers, squeeze‑and‑excitation (SE) modules, stochastic depth, and other variants.

For this master’s thesis, the architecture is intentionally **fixed across all tracks** to keep comparisons attributable to *losses, preprocessing, and augmentations* rather than to capacity changes. Concretely, thesis runs:

- use a **ResNet‑18** layout (BasicBlock, `[2,2,2,2]`)
- use **Batch Normalization** (not GroupNorm)
- use **no SE blocks**

The “flex” knobs remain in code for reproducibility of earlier experiments, but are not used for the main study.

## Input and tensor shapes

The network processes 12‑lead ECG windows as tensors of shape:

- input: `[B, C, T]` with `C=12` and `T=data.crop_size` (default `2500` samples at 400 Hz)

The time dimension `T` is reduced through strided convolutions/pooling and then aggregated by global average pooling.

## Fixed backbone used in all tracks

The backbone follows a standard ResNet‑18 pattern adapted to 1D signals:

1. **Stem**
   - `Conv1d(12 → 64, kernel_size=7, stride=2, padding=3)`
   - normalization: BatchNorm1d
   - ReLU
   - MaxPool1d(kernel_size=3, stride=2, padding=1)

2. **Residual stages** (BasicBlock, expansion = 1)
   - Stage 1: 2 blocks at 64 channels (no downsampling inside the first block)
   - Stage 2: 2 blocks at 128 channels (downsampling via stride=2 in the first block)
   - Stage 3: 2 blocks at 256 channels (downsampling via stride=2 in the first block)
   - Stage 4: 2 blocks at 512 channels (downsampling via stride=2 in the first block)

3. **Pooling and features**
   - AdaptiveAvgPool1d(output_size=1)
   - flatten to `[B, 512]`
   - Dropout (`model.dropout_rate`, default `0.1`)

With these defaults, the backbone feature dimension is:

- `D = 512` (because `inplanes=64`, 4 stages with doubling channels, BasicBlock expansion=1)

## Residual block definition (BasicBlock)

Each BasicBlock performs:

- `3×1 Conv → Norm → ReLU → 3×1 Conv → Norm`
- optional **SE** module (disabled in thesis runs)
- optional **downsample** path when changing resolution/channels
- optional **stochastic depth** (disabled in thesis runs)
- residual add + ReLU

## Heads and track-specific usage

`LitResNet18.forward` returns three tensors:

- `feats`: backbone features after pooling+dropout, shape `[B, D]`
- `proj`: projection output, shape `[B, D]` (identity unless Track 2)
- `logits`: classifier output

### Track 1 (classification)

- Uses the classifier head `fc: Linear(D → 1)` to produce a single logit.
- Projection head is **identity** (`proj = feats`).

### Track 2 (representation learning)

- Enables a **2-layer MLP projection head** (`Linear(D→D) → ReLU → Linear(D→D)`).
- The training objective is computed on **L2-normalized** projected embeddings.
- The classifier loss is disabled by design (`track=2` implies `use_classifier=false`).

### Track 3 (classification with pretrained encoder)

- Loads encoder weights from a Track‑2 checkpoint (`model.pretrained_encoder_path`).
- Freezes the encoder by default (`model.freeze_encoder: true`), optionally unfreezing `layer4`.
- Can optionally replace `fc` with a small MLP head (`model.use_linear_probe_head: true` in `configs/track3_probe.yaml`).

#### Linear Probe vs. End-to-End Classification

This repo uses Track 3 primarily as a **linear evaluation / linear probe** protocol:

- **Track 1 (end-to-end classifier)**: the encoder and classification head are trained together on the label. This answers: *how well can a fully supervised model solve the task?*
- **Track 3 (linear probe)**: the encoder is initialized from Track 2 and typically frozen (`model.freeze_encoder: true`), and only a lightweight head is trained on top. This answers: *how linearly accessible is the label information in the learned representation?*

Important comparison caveats:

- Track 1 vs Track 3 is **not an apples-to-apples “best classifier” comparison**, because Track 1 is allowed to adapt the encoder to the label while Track 3 is not.
- Heads can differ: Track 1 uses `fc`, while Track 3 can use `linear_probe_head` (`model.use_linear_probe_head: true`).
- For a “standard” linear probe baseline, prefer plain BCE on the probe head (rather than focal/RAT-style losses), to avoid mixing representation evaluation with loss shaping.

## “Flex” options kept in code (not used for thesis runs)

The implementation supports, but the thesis configuration does not use:

- alternative residual block type (`block: bottleneck`)
- alternative normalization (`norm_type: group|instance|layer|none`)
- SE blocks (`se_reduction: <int>`)
- stochastic depth (`stochastic_depth_prob > 0`)
- dilation settings (`replace_stride_with_dilation`)

The fixed thesis architecture is fully specified by `configs/base.yaml` plus the track toggles in `configs/track1.yaml`, `configs/track2_sup_*.yaml`, and `configs/track3*.yaml`.
