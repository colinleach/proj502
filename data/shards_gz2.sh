#!/bin/bash

TOP_DIR="/home/colin/data/munch1tb/zoobot_data"
CATALOG_DIR=${TOP_DIR}/pairs_catalogs
SHARD_DIR=${TOP_DIR}/shards/gz2
mkdir -p $SHARD_DIR

python catalog_to_shards.py \
  --shard-type 'gz2' \
  --labelled-catalog ${CATALOG_DIR}/gz2_pairs.csv \
  --shard-dir $SHARD_DIR \
  --img-size 256 \
  --eval-size 1000 \
