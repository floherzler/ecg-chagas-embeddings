#!/bin/bash
for f in track*/*.sh; do
  sed -i 's|wandb login|source /home/lukasses/wandb.txt\
wandb login|g' "$f"
done