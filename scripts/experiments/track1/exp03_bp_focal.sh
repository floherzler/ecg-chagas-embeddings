#!/usr/bin/env bash
# Track 1 (classification) — 4-fold CV — focal loss — bp data — no axis rotation
# Usage: sbatch --array=0-3 scripts/experiments/track1/exp03_bp_focal.sh
# Override epochs at submission if needed (e.g., --export=ALL,MAX_EPOCHS=100).
#SBATCH --job-name=t1-bp-focal
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
TRACK=configs/track1.yaml
EXP_NAME="$(basename "$0" .sh)"

LOSS_CFG="configs/losses/focal_gamma15.yaml"
LOSS_TAG="focal-g15"
PREPROC="bp"
PREPROC_CFG="configs/preproc/${PREPROC}.yaml"

MAX_EPOCHS="${MAX_EPOCHS:-2}"

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

if [[ ${#TRAIN_SPLITS[@]} -ne ${#VAL_SPLITS[@]} ]]; then
  echo "TRAIN_SPLITS and VAL_SPLITS length mismatch" >&2
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ $IDX -ge ${#TRAIN_SPLITS[@]} ]]; then
  echo "SLURM_ARRAY_TASK_ID=$IDX out of range (max $((${#TRAIN_SPLITS[@]} - 1)))" >&2
  exit 1
fi

TRAIN=${TRAIN_SPLITS[$IDX]}
VAL=${VAL_SPLITS[$IDX]}

mkdir -p logs
echo "Running split $IDX train=$TRAIN val=$VAL with $LOSS_CFG"

GROUP_NAME="t1-exp03-bp-focal"
RUN_NAME="t1-exp03-bp-focal-train${TRAIN}-val${VAL}"

python main.py fit \
  --config "$BASE" \
  --config "$PREPROC_CFG" \
  --config "$TRACK" \
  --config "$LOSS_CFG" \
  --trainer.max_epochs "$MAX_EPOCHS" \
  --data.meta_path "$META_PATH" \
  --data.data_dir "$DATA_DIR" \
  --data.train_folds "$TRAIN" \
  --data.valid_folds "$VAL" \
  --trainer.logger.init_args.group "$GROUP_NAME" \
  --trainer.logger.init_args.name "$RUN_NAME" \
  --trainer.default_root_dir "/tmp/$RUN_NAME"
