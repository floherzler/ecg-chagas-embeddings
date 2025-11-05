#!/bin/bash
CONFIG=$1
EXP_NAME=$(basename $CONFIG .yaml)

srun python -m ecg_chagas_embeddings.train \
  --config $CONFIG \
  --default_root_dir outputs/$EXP_NAME
