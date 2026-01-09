#!/usr/bin/env bash
# Run thesis track experiments using the *bandpassed* (bp) data folder on scratch.
#
# Examples:
#   bash scripts/experiment_tracks_bp.sh track1
#   bash scripts/experiment_tracks_bp.sh track2 sup_min
#   bash scripts/experiment_tracks_bp.sh track3 /path/to/track2.ckpt
#   bash scripts/experiment_tracks_bp.sh all sup_min /path/to/track2.ckpt
#
# Optional env overrides:
#   SCRATCH_BASE=/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster
#   TRAIN_FOLDS='[0,1,2,3]'  VAL_FOLDS='[4]'
#   MAX_EPOCHS=50  BATCH_SIZE=256
#   AXIS_ROT_MAX_DEG=15  AXIS_ROT_PROB=0.5  PER_VIEW_AXIS_ROT=true

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

BASE_CFG="configs/base.yaml"

SCRATCH_BASE="${SCRATCH_BASE:-/sc-scratch/sc-scratch-dh-face/physionet2025/processedMaster}"
META_PATH="${META_PATH:-$SCRATCH_BASE/metadata.csv}"
DATA_DIR="${DATA_DIR:-$SCRATCH_BASE/bp}"

TRAIN_FOLDS="${TRAIN_FOLDS:-[0,1,2,3]}"
VAL_FOLDS="${VAL_FOLDS:-[4]}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"


# VCG frontal-axis rotation augmentation (best used on bp signals; keep 0 to disable)
AXIS_ROT_MAX_DEG="${AXIS_ROT_MAX_DEG:-0}"
AXIS_ROT_PROB="${AXIS_ROT_PROB:-1.0}"
PER_VIEW_AXIS_ROT="${PER_VIEW_AXIS_ROT:-true}"

MODE="${1:-}"
TRACK2_VARIANT="${2:-sup_min}"  # sup_min | sup_proto
TRACK2_CKPT="${3:-}"           # used for track3

usage() {
  cat <<EOF
Usage:
  $0 track1
  $0 track2 [sup_min|sup_proto]
  $0 track3 <track2_ckpt_path>
  $0 all [sup_min|sup_proto] <track2_ckpt_path>
EOF
}

if [[ -z "$MODE" ]]; then
  usage
  exit 2
fi

common_args=(
  --config "$BASE_CFG"
  --data.meta_path "$META_PATH"
  --data.data_dir "$DATA_DIR"
  --data.train_folds "$TRAIN_FOLDS"
  --data.valid_folds "$VAL_FOLDS"
  --data.batch_size "$BATCH_SIZE"
  --trainer.max_epochs "$MAX_EPOCHS"
)

if [[ "${AXIS_ROT_MAX_DEG}" != "0" && "${AXIS_ROT_MAX_DEG}" != "0.0" ]]; then
  common_args+=(
    --data.axis_rotation_max_deg "$AXIS_ROT_MAX_DEG"
    --data.axis_rotation_prob "$AXIS_ROT_PROB"
    --data.per_view_axis_rotation "$PER_VIEW_AXIS_ROT"
  )
fi

run_fit() {
  local run_name="$1"
  shift
  mkdir -p logs
  echo "==> $run_name"
  python main.py fit \
    "${common_args[@]}" \
    --trainer.logger.init_args.group "thesis-${MODE}" \
    --trainer.logger.init_args.name "$run_name" \
    --trainer.default_root_dir "/tmp/$run_name" \
    "$@"
}

run_track1() {
  run_fit "track1-bp" --config "configs/track1.yaml"
}

run_track2() {
  local variant="$1"
  local cfg
  case "$variant" in
    sup_min) cfg="configs/track2_sup_min.yaml" ;;
    sup_proto) cfg="configs/track2_sup_proto.yaml" ;;
    *) echo "Unknown track2 variant: $variant (use sup_min or sup_proto)" >&2; exit 2 ;;
  esac
  run_fit "track2-${variant}-bp" --config "$cfg"
}

run_track3() {
  local ckpt="$1"
  if [[ -z "$ckpt" ]]; then
    echo "track3 requires a track2 checkpoint path" >&2
    exit 2
  fi
  run_fit "track3-bp" \
    --config "configs/track3.yaml" \
    --model.pretrained_encoder_path "$ckpt"
}

case "$MODE" in
  track1) run_track1 ;;
  track2) run_track2 "$TRACK2_VARIANT" ;;
  track3) run_track3 "$TRACK2_CKPT" ;;
  all)
    run_track1
    run_track2 "$TRACK2_VARIANT"
    run_track3 "$TRACK2_CKPT"
    ;;
  *)
    usage
    exit 2
    ;;
esac
