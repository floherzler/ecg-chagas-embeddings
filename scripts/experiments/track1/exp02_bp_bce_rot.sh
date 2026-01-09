#!/usr/bin/env bash
# Track 1 (classification) — 4-fold CV — weighted BCE — bp — WITH axis rotation
# Usage: sbatch --array=0-3 scripts/experiments/track1/exp02_bp_bce_rot.sh
#SBATCH --job-name=t1-bp-bcew-rot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3
# Keep environment (PATH/venv) intact.
#SBATCH --export=ALL

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# activate venv and login to wandb
source .venv/bin/activate

wandb login "$WANDB_API_KEY"

# use base + track1 config
BASE=configs/base.yaml
TRACK=configs/track1.yaml

# Single experiment config
LOSS_CFG="configs/losses/bce_weighted.yaml"
LOSS_TAG="bcew"
PREPROC="bp"
PREPROC_CFG="configs/preproc/${PREPROC}.yaml"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
ROT_DEG="${ROT_DEG:-10}"

SCRATCH_BASE="${SCRATCH_BASE:-/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster}"
META_PATH="${META_PATH:-$SCRATCH_BASE/metadata.csv}"
DATA_DIR="${DATA_DIR:-$SCRATCH_BASE/$PREPROC}"

# Cross-validation splits (must be same length)
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

GROUP_NAME="t1-${PREPROC}-${LOSS_TAG}-rot"
RUN_NAME="t1-${PREPROC}-${LOSS_TAG}-rot${ROT_DEG}-fold${IDX}-train${TRAIN}-val${VAL}"

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
  --trainer.default_root_dir "/tmp/$RUN_NAME" \
  --data.axis_rotation_max_deg "$ROT_DEG" \
  --data.axis_rotation_prob 1.0 \
  --data.per_view_axis_rotation true
