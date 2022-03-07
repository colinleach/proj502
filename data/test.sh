#!/bin/bash

python /home/colin/code/galaxycnn/zoobot/zoobot/data_utils/create_shards.py \
  --labelled-catalog gz2_partial_pairs.csv \
  --shard-dir /home/colin/data/munch1tb/zoobot_data/shards \
  --img-size 300 \
  --eval-size 100 \
  --max-labelled 500 \
  --max-unlabelled 300 \
  --img-size 32