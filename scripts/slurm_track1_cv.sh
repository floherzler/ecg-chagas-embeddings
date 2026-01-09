#!/usr/bin/env bash
# Simple CV sweep: one loss config, SLURM array iterates train/val splits.
# Usage: sbatch --array=0-1 scripts/slurm_track1_cv.sh
# Adjust TRAIN_SPLITS/VAL_SPLITS below as needed.
#SBATCH --job-name=track1-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1
# Keep environment (PATH/venv) intact.
#SBATCH --export=ALL

set -euo pipefail

source .venv/bin/activate

wandb login "$WANDB_API_KEY"

BASE=configs/base.yaml
TRACK=configs/track1.yaml

# Single experiment config
LOSS_CFG="configs/losses/focal_gamma15.yaml"
BASE_NAME="loss-focal-g15"

# Cross-validation splits (must be same length)
TRAIN_SPLITS=(
  "[0,1,2]"
  "[1,2,3]"
)
VAL_SPLITS=(
  "[3]"
  "[0]"
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

GROUP_NAME="track1-${BASE_NAME}"
RUN_NAME="track1-${BASE_NAME}-train${TRAIN}-val${VAL}"

python main.py fit \
  --config "$BASE" \
  --config "$TRACK" \
  --config "$LOSS_CFG" \
  --trainer.max_epochs 2 \
  --data.meta_path "/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster/metadata.csv" \
  --data.data_dir "/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster/bp" \
  --data.train_folds "$TRAIN" \
  --data.valid_folds "$VAL" \
  --trainer.logger.init_args.group "$GROUP_NAME" \
  --trainer.logger.init_args.name "$RUN_NAME" \
  --trainer.default_root_dir "/tmp/$RUN_NAME"
