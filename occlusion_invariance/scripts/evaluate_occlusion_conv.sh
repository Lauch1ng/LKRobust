#!/bin/bash

DATA_PATH="/dataset/imagenet/val"

python evaluate.py \
  --model_name replknet-B-22k \
  --test_dir "$DATA_PATH" \
  --pretrained_weights /checkpoints/RepLKNet-31B_ImageNet-22K-to-1K_224.pth \
  --dino
#--random
