#!/bin/bash
# Full highlight generation — all 29 classes, 35 queries each, top_k=40
# Target: ≥25% of match duration (~37 min for a 2h26m match)
# Output: highlights_tdl_v2/

set -e

PYTHON=/home/azhan1303/anaconda3/envs/tdl-project/bin/python
SCRIPT=/home/azhan1303/Documents/vishnu/MATR/generate_highlights_imagebind.py

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/azhan1303/Documents/vishnu/MATR \
$PYTHON $SCRIPT \
  --match_video   tdl-videos/test-video.mp4 \
  --match_feat_cache tdl-videos/test-video_imagebind_feats.npz \
  --checkpoint    "data/sportsmr/tdl project /imagebind-best.ckpt" \
  --query_feat_dir data/sportsmr/vid_imagebind_query \
  --classes 1 2 5 9 11 18 28 \
  --num_queries   35 \
  --top_k         40 \
  --confidence_threshold 0.9 \
  --nms_iou       0.4 \
  --min_clip_dur  2.0 \
  --max_clip_dur  30.0 \
  --output_dir    highlights_tdl_v2 \
  "$@"
