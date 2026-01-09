#!/usr/bin/env bash
#SBATCH --job-name=track1-loss-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3  # override at submission: sbatch --array=0-$((${NUM_EXPERIMENTS}-1)) ...

set -euo pipefail

source .venv/bin/activate

wandb login "$WANDB_API_KEY"

BASE=configs/base.yaml
TRACK=configs/track1.yaml

# Define experiments here; arrays must have the same length.
# Loss configs (array over these)
LOSSES=(
  "configs/losses/bce_weighted.yaml"   # exp 0
  "configs/losses/focal_gamma15.yaml"  # exp 1
)
NAMES=(
  "loss-bce-weighted"
  "loss-focal-g15"
)

# CV splits must be provided via env vars TRAIN_FOLDS_SINGLE and VAL_FOLDS_SINGLE
if [[ -z "${TRAIN_FOLDS_SINGLE:-}" || -z "${VAL_FOLDS_SINGLE:-}" ]]; then
  echo "Please provide TRAIN_FOLDS_SINGLE and VAL_FOLDS_SINGLE via --export or env." >&2
  exit 1
fi

NUM_EXPERIMENTS=${#LOSSES[@]}
if [[ ${#NAMES[@]} -ne $NUM_EXPERIMENTS ]]; then
  echo "Length mismatch: LOSSES=${#LOSSES[@]}, NAMES=${#NAMES[@]}" >&2
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ $IDX -ge $NUM_EXPERIMENTS ]]; then
  echo "SLURM_ARRAY_TASK_ID=$IDX out of range (NUM_EXPERIMENTS=$NUM_EXPERIMENTS); exiting." >&2
  exit 1
fi
LOSS_CFG=${LOSSES[$IDX]}
BASE_NAME=${NAMES[$IDX]}

mkdir -p logs

echo "Running $BASE_NAME with loss config $LOSS_CFG"

GROUP_NAME="track1-${BASE_NAME}"
RUN_NAME="track1-${BASE_NAME}-train${TRAIN_FOLDS_SINGLE}-val${VAL_FOLDS_SINGLE}"

srun python main.py fit \
  --config "$BASE" \
  --config "$TRACK" \
  --config "$LOSS_CFG" \
  --trainer.max_epochs 2 \
  --data.meta_path "/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster/metadata.csv" \
  --data.data_dir "/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster/bp" \
  --data.train_folds "$TRAIN_FOLDS_SINGLE" \
  --data.valid_folds "$VAL_FOLDS_SINGLE" \
  --trainer.logger.init_args.group "$GROUP_NAME" \
  --trainer.logger.init_args.name "$RUN_NAME" \
  --trainer.default_root_dir "/tmp/$RUN_NAME"
