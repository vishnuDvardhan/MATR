# MATR — Project Context for Claude

## What This Project Is
MATR (Aligning Moments in Time using Video Queries) is an ICCV 2025 model for video-to-video moment retrieval. Given a short query video (e.g. a cricket six) and a long target video (e.g. a full match), it finds timestamps in the target where the same action occurs.

## Dataset: SportsMoments
- **Location**: `data/sportsmr/`
- **HuggingFace**: 29-class cricket + football video dataset
- **750,393 training samples**, 10,000 val samples
- Raw videos: `data/sportsmr/raw_target_videos/` and `data/sportsmr/raw_query_videos/`
- CLIP features (512-dim): `data/sportsmr/vid_clip/` (target) and `data/sportsmr/vid_clip_query/` (query)
- ImageBind features (1024-dim): `data/sportsmr/vid_imagebind/` (target) and `data/sportsmr/vid_imagebind_query/` (query) — extraction in progress

### Class ID → Label Mapping (29 classes)
```
1=lbw, 2=four, 3=cover drive, 4=ducking a bouncer, 5=run out,
6=corner kick, 7=goal kick, 8=full toss, 9=bowled, 10=yorker,
11=stumping, 12=red card, 13=sliding tackle, 14=wide ball, 15=pull shot,
16=throw, 17=offside, 18=catching, 19=penalty, 20=substitute,
21=bouncer, 22=goal, 23=save, 24=yellow card, 25=free kick,
26=header, 27=sweep shot, 28=six, 29=upper cut
```
Video filenames follow the format: `{YouTubeID}__{classID}_{videoID}.mp4`

## Environment
- **MATR conda env** (`/home/user/miniconda3/envs/MATR/`) uses **GraalPy** (JVM-based Python), NOT CPython. This means numpy C-extensions are unreliable outside of training.
- **Use `ml_fresh_start`** (`/home/user/miniconda3/envs/ml_fresh_start/bin/python`) for inference scripts — it is CPython 3.10 with numpy 1.26.4, torch 2.0.1+cu117, CLIP, torchvision.
- **Training** uses `/home/user/miniconda3/envs/MATR/bin/python` (GraalPy works for training via CUDA).

## GPUs
- GPU 0: NVIDIA RTX 4000 Ada (20GB) — used for inference / feature extraction
- GPU 1: NVIDIA RTX 6000 Ada (48GB) — used for training (`CUDA_VISIBLE_DEVICES=1` implicitly)

## Bug Fixes Applied to Original Author Code
The original code had several bugs that were fixed to make it runnable:

1. **`main/dataset.py`**: Removed broken imports (`nncore`, `config_hl`, `load_pickle`, `pad_sequences_1d`). Fixed `IndentationError` in `_get_query_feat_by_qid`. Added missing `start_end_collate_mr` and `prepare_batch_inputs_mr` functions (imported by `train.py` but never defined).

2. **`model/matr.py`**: Fixed forward signature to accept dataloader kwargs (`query_feat`→`src_txt`, `video_feat`→`src_vid`). Fixed `NameError`: `txt.shape[1]` → `src_txt.shape[1]`. Fixed mask generation: use `torch.ones` (1=valid convention), not zeros.

3. **`model/align_transformer.py`**: Removed broken DTW extraction loop (VarTable is not a tensor; boolean indexing produces variable-length results that can't be stacked). Replaced with `aligned_tensor = vid_mem` direct pass-through.

4. **`main/inference_mr.py`**: Fixed wrong dict key: `model_inputs["src_vid_mask"]` → `targets["timestamp_mask"]`.

5. **`utils/basic_utils.py`**: Added `max` and `min` attributes to `AverageMeter` (required by `train.py` line 149).

## Training
```bash
# Current training command (edit train.sh for config):
bash train.sh > training_progress.txt 2>&1 &
```

**Current config in `train.sh`:**
- `dset_name="sportsmr"` (must be `actnet` or `sportsmr` — config rejects other values)
- `exp_id="sportsmoment-clip-2"`
- `bsz=400` (optimal for RTX 6000 Ada — bsz=600 is slower due to GPU saturation)
- `n_epoch=30`
- `v_feat_dirs`: `${feat_root}/vid_clip` (NOT `target_vid_clip`)
- `t_feat_dir`: `${feat_root}/vid_clip_query` (NOT `query_vid_clip`)
- Python: `/home/user/miniconda3/envs/MATR/bin/python`
- PYTHONPATH: `/home/user/Desktop/MATR`
- Speed: ~2.3 hrs/epoch → ~3 days total for 30 epochs

**Latest checkpoint**: `results/mr-sportsmr/sportsmoment-clip-2-clip-clip-2026_04_03_18/model_best.ckpt`

Monitor training: `tail -f training_progress.txt`

## Feature Extraction

### CLIP features (already done)
```bash
PYTHONPATH=/home/user/Desktop/MATR /home/user/miniconda3/envs/MATR/bin/python extract_clip_features.py \
  --video_dir data/sportsmr/raw_target_videos --out_dir data/sportsmr/vid_clip
PYTHONPATH=/home/user/Desktop/MATR /home/user/miniconda3/envs/MATR/bin/python extract_clip_features.py \
  --video_dir data/sportsmr/raw_query_videos --out_dir data/sportsmr/vid_clip_query
```

### ImageBind features (in progress, ~14hrs ETA)
Vision+audio fused, 1024-dim. To train with ImageBind: set `v_feat_dim=1024`, `t_feat_dim=1024` in `train.sh` and point feature dirs to `vid_imagebind` / `vid_imagebind_query`.
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/MATR/bin/python extract_imagebind_features.py \
  --video_dir data/sportsmr/raw_target_videos --out_dir data/sportsmr/vid_imagebind
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/MATR/bin/python extract_imagebind_features.py \
  --video_dir data/sportsmr/raw_query_videos --out_dir data/sportsmr/vid_imagebind_query
```

## Highlight Generation

Use `generate_highlights.py` to find and cut highlight moments from a match video.

### Usage
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/ml_fresh_start/bin/python generate_highlights.py \
  --match_video test_videos/cricket_sample.mp4 \
  --query_videos "six:q1.mp4,q2.mp4,q3.mp4" "four:q4.mp4,q5.mp4" \
  --checkpoint results/mr-sportsmr/sportsmoment-clip-2-clip-clip-2026_04_03_18/model_best.ckpt \
  --output_dir test_output/highlights_out \
  --top_k 5 \
  --confidence_threshold 0.3 \
  --nms_iou 0.5
```

### Key flags
- `--query_videos`: `label:path1,path2,...` — multiple query videos per label for multi-query aggregation
- `--confidence_threshold`: Filter out predictions below this score (0.3 is a good starting point after a few epochs; lower to 0.05–0.1 for early checkpoints)
- `--top_k`: Max clips per label after NMS
- `--nms_iou`: IoU threshold for non-maximum suppression

### Multi-query boosting
When multiple query videos are given for a label, windows confirmed by multiple queries get a confidence boost (up to 2×). This significantly improves precision — e.g. max confidence went from 0.18 (single query) to 0.44 (5 queries).

### Using SportsMoments dataset videos as queries
The dataset query videos are in `data/sportsmr/raw_query_videos/`. Use class ID from the filename to find the right action:
```bash
# Six (class 28) examples:
ls data/sportsmr/raw_query_videos/ | grep "__28_" | head -5

# Four (class 2) examples:
ls data/sportsmr/raw_query_videos/ | grep "__2_" | head -5
```

### Output
- `output_dir/highlights.mp4` — stitched highlight reel
- `output_dir/clips/` — individual trimmed clips
- `output_dir/summary.json` — per-query predictions with timestamps and confidence scores

## Notes
- Training is ongoing. Better checkpoints will improve confidence scores and recall.
- Once ImageBind features are extracted, retrain with `v_feat_dim=1024` for potentially better accuracy (richer multimodal embeddings vs CLIP's vision-only 512-dim).
- `ffmpeg` path used by scripts: `/home/user/miniconda3/envs/ml_fresh_start/bin/ffmpeg`
