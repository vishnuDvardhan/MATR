#!/bin/bash
# Continues the pipeline after downloads:
# 1. Wait for target zip extraction to finish
# 2. Move target videos to correct location
# 3. Extract CLIP features for target videos
# 4. Wait for query features to finish too
# 5. Run training

MATR_DIR="/home/user/Desktop/MATR"
LOG="$MATR_DIR/pipeline_log.txt"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

> "$LOG"
log "Pipeline started"

# ── Step 1: Wait for unzip to finish ──────────────────────────────────────────
log "Waiting for target video extraction to complete..."
while pgrep -f "unzip.*raw_target_video" > /dev/null; do
    EXTRACTED=$(ls "$MATR_DIR/data/sportsmr_target_tmp/all_references/" 2>/dev/null | wc -l)
    log "  Extracting... $EXTRACTED files so far"
    sleep 30
done
log "Extraction complete"

# ── Step 2: Move to correct location ──────────────────────────────────────────
SRC="$MATR_DIR/data/sportsmr_target_tmp/all_references"
DST="$MATR_DIR/data/sportsmr/raw_target_videos"
log "Moving target videos from $SRC to $DST ..."
mv "$SRC"/*.mp4 "$DST"/
TOTAL=$(ls "$DST"/*.mp4 2>/dev/null | wc -l)
log "Moved $TOTAL target video files to $DST"
rm -rf "$MATR_DIR/data/sportsmr_target_tmp"

# ── Step 3: Extract CLIP features for target videos ───────────────────────────
log "Starting CLIP feature extraction for target videos..."
conda run -n MATR python "$MATR_DIR/extract_clip_features.py" \
    --video_dir "$DST" \
    --out_dir "$MATR_DIR/data/sportsmr/vid_clip" \
    --device cuda 2>&1 | tee -a "$LOG"
log "Target CLIP extraction done: $(ls "$MATR_DIR/data/sportsmr/vid_clip"/*.npz 2>/dev/null | wc -l) features"

# ── Step 4: Wait for query feature extraction to finish ───────────────────────
log "Waiting for query CLIP feature extraction to finish..."
while pgrep -f "extract_clip_features.*vid_clip_query" > /dev/null; do
    Q=$(ls "$MATR_DIR/data/sportsmr/vid_clip_query"/*.npz 2>/dev/null | wc -l)
    log "  Query features: $Q / 4998"
    sleep 60
done
Q=$(ls "$MATR_DIR/data/sportsmr/vid_clip_query"/*.npz 2>/dev/null | wc -l)
log "Query extraction done: $Q features"

# ── Step 5: Run training ───────────────────────────────────────────────────────
log "Starting training..."
cd "$MATR_DIR" && conda run -n MATR bash train.sh 2>&1 | tee -a "$LOG"
log "Training complete"
