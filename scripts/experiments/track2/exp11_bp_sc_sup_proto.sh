#!/usr/bin/env bash
# Track 2 (projection) — 4-fold CV — Prototype variant — bp_sc — no axis rotation
# Usage: sbatch --array=0-3 scripts/experiments/track2/exp11_bp_sc_sup_proto.sh
#SBATCH --job-name=t2-bpsc-proto
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --export=ALL

set -euo pipefail

source .venv/bin/activate
wandb login "$WANDB_API_KEY"

BASE=configs/base.yaml
TRACK=configs/track2_sup_proto.yaml

VARIANT="proto"
PREPROC="bp_sc"
AUG="norot"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

SCRATCH_BASE="${SCRATCH_BASE:-/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster}"
META_PATH="${META_PATH:-$SCRATCH_BASE/metadata.csv}"
DATA_DIR="${DATA_DIR:-$SCRATCH_BASE/$PREPROC}"

TRAIN_SPLITS=(
  "[0,1,2]"
  "[1,2,3]"
  "[2,3,0]"
  "[3,0,1]"
)
VAL_SPLITS=(
  "[3]"
  "[0]"
  "[1]"
  "[2]"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
TRAIN=${TRAIN_SPLITS[$IDX]}
VAL=${VAL_SPLITS[$IDX]}

mkdir -p logs
GROUP_NAME="t2-${VARIANT}-${PREPROC}-${AUG}"
RUN_NAME="t2-${VARIANT}-${PREPROC}-${AUG}-fold${IDX}-train${TRAIN}-val${VAL}"

python main.py fit \
  --config "$BASE" \
  --config "$TRACK" \
  --trainer.max_epochs "$MAX_EPOCHS" \
  --data.meta_path "$META_PATH" \
  --data.data_dir "$DATA_DIR" \
  --data.train_folds "$TRAIN" \
  --data.valid_folds "$VAL" \
  --trainer.logger.init_args.group "$GROUP_NAME" \
  --trainer.logger.init_args.name "$RUN_NAME" \
  --trainer.default_root_dir "/tmp/$RUN_NAME"
