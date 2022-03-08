#!/bin/bash

TOP_DIR="/home/colin/data/munch1tb/zoobot_data/shards"
SUB_DIR=${TOP_DIR}/gz2_partial
mkdir -p $SUB_DIR

python catalog_to_shards.py \
  --shard-type 'gz2_partial' \
  --labelled-catalog gz2_partial_pairs.csv \
  --shard-dir $SUB_DIR \
  --img-size 300 \
  --eval-size 100 \
  --max-labelled 500 \
  --max-unlabelled 300 \
  --img-size 32