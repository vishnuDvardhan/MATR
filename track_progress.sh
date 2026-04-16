#!/bin/bash
TOTAL_GB=45.9
LOGFILE="/home/user/Desktop/MATR/download_progress.txt"
HF_DIR="/home/user/.cache/huggingface/hub"
DEST="/home/user/Desktop/MATR/data/raw_target_video.zip"

> "$LOGFILE"
while true; do
    TMPFILE=$(ls -t "$HF_DIR"/tmp* 2>/dev/null | head -1)

    if [ -f "$DEST" ]; then
        SIZE=$(du -sh "$DEST" | cut -f1)
        echo "[$(date '+%H:%M:%S')] COMPLETE — saved to $DEST ($SIZE)" | tee -a "$LOGFILE"
        break
    elif [ -n "$TMPFILE" ] && [ -f "$TMPFILE" ]; then
        SIZE_GB=$(du -s "$TMPFILE" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
        PCT=$(awk "BEGIN {printf \"%.1f\", ($SIZE_GB/$TOTAL_GB)*100}")
        echo "[$(date '+%H:%M:%S')] Downloading: ${SIZE_GB}G / ${TOTAL_GB}G  (${PCT}%)" | tee -a "$LOGFILE"
    else
        echo "[$(date '+%H:%M:%S')] Waiting..." | tee -a "$LOGFILE"
    fi
    sleep 30
done
