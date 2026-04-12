#!/bin/bash
# Highlight generation for t20-test.mp4
# Features extracted on-the-fly and cached as t20-test_imagebind_feats.npz
# Output: highlights_t20/

set -e

PYTHON=/home/azhan1303/anaconda3/envs/tdl-project/bin/python
SCRIPT=/home/azhan1303/Documents/vishnu/MATR/generate_highlights_imagebind.py

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/azhan1303/Documents/vishnu/MATR \
$PYTHON $SCRIPT \
  --match_video   t20-test.mp4 \
  --checkpoint    "data/sportsmr/tdl project /imagebind-best.ckpt" \
  --query_feat_dir data/sportsmr/vid_imagebind_query \
  --classes 1 2 5 9 11 18 28 \
  --num_queries   35 \
  --top_k         40 \
  --confidence_threshold 0.9 \
  --nms_iou       0.4 \
  --min_clip_dur  2.0 \
  --max_clip_dur  30.0 \
  --output_dir    highlights_t20 \
  "$@"
