#!/bin/bash
set -e
HOSTFILE="${HOSTFILE:-~/.mlx/hostfile.json}"
MODEL="${MODEL:?Set MODEL}" DATA="${DATA:?Set DATA}" ADAPTERS="${ADAPTERS:-./adapters}"
ITERS="${ITERS:-1000}" BATCH="${BATCH:-2}" LAYERS="${LAYERS:-12}" LR="${LR:-1e-5}" SEQ="${SEQ:-2048}"
echo "Training: $MODEL | Data: $DATA | Iters: $ITERS"
mlx.launch --hostfile "$HOSTFILE" -- /opt/homebrew/bin/python3 -m mlx_lm lora \
  --model "$MODEL" --data "$DATA" --train --adapter-path "$ADAPTERS" \
  --iters "$ITERS" --batch-size "$BATCH" --num-layers "$LAYERS" --learning-rate "$LR" \
  --steps-per-eval 200 --steps-per-report 25 --max-seq-length "$SEQ" \
  --seed 42 --save-every 200 --grad-checkpoint
