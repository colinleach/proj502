#!/bin/bash

TOP_DIR="/home/colin/data/munch1tb/zoobot_data/shards"
TRAIN_DIR=${TOP_DIR}/gz2_partial/train_shards
EVAL_DIR=${TOP_DIR}/gz2_partial/val_shards

EXPT_DIR="results/gz2_debug"
mkdir -p $EXPT_DIR

python train_model.py \
  --pair-type gz2_partial \
  --experiment-dir $EXPT_DIR \
  --shard-img-size 32 \
  --train-dir $TRAIN_DIR \
  --eval-dir $EVAL_DIR \
  --epochs 1 \
  --batch-size 8 \
  --resize-size 128
