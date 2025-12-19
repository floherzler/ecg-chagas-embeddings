#!/usr/bin/env bash
set -euo pipefail

uv sync

# Define CV splits here; edit as needed.
TRAIN_SPLITS=(
  "[0,1,2]"
  "[1,2,3]"
)
VAL_SPLITS=(
  "[3]"
  "[0]"
)

SCRIPT="scripts/slurm_track1_losses.sh"

if [[ ${#TRAIN_SPLITS[@]} -ne ${#VAL_SPLITS[@]} ]]; then
  echo "TRAIN_SPLITS and VAL_SPLITS length mismatch" >&2
  exit 1
fi

for i in "${!TRAIN_SPLITS[@]}"; do
  TRAIN=${TRAIN_SPLITS[$i]}
  VAL=${VAL_SPLITS[$i]}
  echo "Submitting CV split $i: train=$TRAIN val=$VAL"
  sbatch --array=0-1 \
    --export=TRAIN_FOLDS_SINGLE="$TRAIN",VAL_FOLDS_SINGLE="$VAL" \
    "$SCRIPT"
done
