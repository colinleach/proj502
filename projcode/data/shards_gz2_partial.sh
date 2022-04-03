#!/bin/bash

TOP_DIR="/home/colin/data/munch1tb/zoobot_data"
CATALOG_DIR=${TOP_DIR}/pairs_catalogs
SHARD_DIR=${TOP_DIR}/shards/gz2_partial
mkdir -p $SHARD_DIR

python catalog_to_shards.py \
  --shard-type 'gz2_partial' \
  --labelled-catalog ${CATALOG_DIR}/gz2_partial_pairs.csv \
  --shard-dir $SHARD_DIR \
  --eval-size 100 \
  --max-labelled 500 \
  --max-unlabelled 300 \
  --img-size 32
