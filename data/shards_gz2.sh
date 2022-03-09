#!/bin/bash

TOP_DIR="/home/colin/data/munch1tb/zoobot_data/shards"
SUB_DIR=${TOP_DIR}/gz2
mkdir -p $SUB_DIR

python catalog_to_shards.py \
  --shard-type 'gz2' \
  --labelled-catalog gz2_pairs.csv \
  --shard-dir $SUB_DIR \
  --img-size 256 \
  --eval-size 1000 \
