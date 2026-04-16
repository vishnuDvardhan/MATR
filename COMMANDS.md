# MATR — Commands & Environments Reference

## Conda Environments

| Environment | Python | Purpose |
|-------------|--------|---------|
| `matr_imagebind` | CPython 3.10 | **Training** (both CLIP and ImageBind models) |
| `ml_fresh_start` | CPython 3.10 | **Highlight generation & inference** (has older setuptools with `pkg_resources` — required by ImageBind's data.py) |
| `MATR` | GraalPy (JVM) | Legacy — do NOT use for inference, numpy C-extensions unreliable |

### Key binary paths
```
/home/user/miniconda3/envs/matr_imagebind/bin/python
/home/user/miniconda3/envs/ml_fresh_start/bin/python
/home/user/miniconda3/envs/ml_fresh_start/bin/ffmpeg
```

---

## GPU Assignment

| Physical GPU | nvidia-smi index | CUDA index | Used for |
|-------------|-----------------|------------|---------|
| NVIDIA RTX 6000 Ada (48GB) | 1 | 0 | ImageBind training |
| NVIDIA RTX 4000 Ada (20GB) | 0 | 1 | CLIP training |

> **Note:** CUDA device order is REVERSED from nvidia-smi. Always use `CUDA_VISIBLE_DEVICES` to pin correctly.

---

## Feature Extraction

### CLIP features (512-dim, vision-only) — already done
```bash
# Target videos
PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/matr_imagebind/bin/python extract_clip_features.py \
  --video_dir data/sportsmr/raw_target_videos \
  --out_dir data/sportsmr/vid_clip

# Query videos
PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/matr_imagebind/bin/python extract_clip_features.py \
  --video_dir data/sportsmr/raw_query_videos \
  --out_dir data/sportsmr/vid_clip_query
```

### ImageBind features (1024-dim, vision+audio fused) — already done
```bash
# Target videos (CUDA 0 = RTX 6000 physically)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/matr_imagebind/bin/python extract_imagebind_features.py \
  --video_dir data/sportsmr/raw_target_videos \
  --out_dir data/sportsmr/vid_imagebind

# Query videos
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/matr_imagebind/bin/python extract_imagebind_features.py \
  --video_dir data/sportsmr/raw_query_videos \
  --out_dir data/sportsmr/vid_imagebind_query
```
- Fuses vision + audio per 2-second segment: `fused = (vis_emb + aud_emb) / 2.0`, L2 normalized
- Falls back to vision-only if no audio track

---

## Training

### ImageBind model (RTX 6000, CUDA_VISIBLE_DEVICES=0)
```bash
# Edit train.sh for config, then:
bash train.sh > training_progress.txt 2>&1 &
tail -f training_progress.txt
```

**Current stable config (`train.sh` — sportsmoment-imagebind-4):**
```
exp_id="sportsmoment-imagebind-4"
v_feat_types=imagebind   # vid_imagebind/ (1024-dim)
t_feat_type=imagebind    # vid_imagebind_query/ (1024-dim)
CUDA_VISIBLE_DEVICES=0   # RTX 6000
bsz=400
n_epoch=100
lr=5e-5                  # reduced from 1e-4 (was causing divergence)
lr_drop=70
lr_warmup=20             # doubled from 10 (slower ramp-up)
s_loss_intra_coef=0      # disabled until MR losses stabilize
s_loss_inter_coef=0      # disabled until MR losses stabilize
grad_clip=0.1
hidden_dim=1024
enc_layers=4
```

> **Why lr=5e-5:** With 1024-dim ImageBind features, lr=1e-4 caused loss_f to spike 3x after epoch 1
> (0.074 → 0.222) and diverge. Saliency losses stuck at 2.7–3.3 with no decrease.
> Plan: re-enable saliency at s_inter=0.2, s_intra=0.1 once training is stable (~epoch 10).

**Best checkpoint so far:** `results/mr-sportsmr/sportsmoment-imagebind-3-imagebind-imagebind-2026_04_06_08/model_best.ckpt`
(epoch 1: R1@0.5=56.51%, mAP=30.56%, mIoU=49.28%)

---

### CLIP model (RTX 4000, CUDA_VISIBLE_DEVICES=1)
```bash
bash train_clip.sh > training_progress_clip.txt 2>&1 &
tail -f training_progress_clip.txt
```

**Current config (`train_clip.sh` — sportsmoment-clip-4):**
```
exp_id="sportsmoment-clip-4"
v_feat_types=clip        # vid_clip/ (512-dim)
t_feat_type=clip         # vid_clip_query/ (512-dim)
CUDA_VISIBLE_DEVICES=1   # RTX 4000
bsz=400
n_epoch=100
lr=1e-4
lr_drop=70
lr_warmup=10
s_loss_intra_coef=0.5
s_loss_inter_coef=1
hidden_dim=1024
enc_layers=4
```

**Training trajectory (epoch → R1@0.5 / mAP):**
- Epoch 1: 44.02% / 25.32%
- Epoch 3: 47.20% / 25.13% ← stable and improving

**Latest checkpoint:** `results/mr-sportsmr/sportsmoment-clip-4-clip-clip-2026_04_06_08/`

---

## Highlight Generation

### Using ImageBind features — `generate_highlights_imagebind.py`
**Environment:** `ml_fresh_start`  
**Script:** `generate_highlights_imagebind.py`

#### With pre-extracted match features (fast, recommended)
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/ml_fresh_start/bin/python generate_highlights_imagebind.py \
  --match_video /home/user/Desktop/MATR/video.mp4 \
  --match_feat /home/user/Desktop/MATR/video.npz \
  --query_feat_dir data/sportsmr/vid_imagebind_query \
  --query_dir data/sportsmr/query_videos_by_class \
  --classes six four \
  --checkpoint /home/user/Desktop/MATR/results/mr-sportsmr/sportsmoment-imagebind-4-imagebind-imagebind-2026_04_06_17/model_latest.ckpt \
  --output_dir highlights_output \
  --top_k 10 \
  --confidence_threshold 0.1 \
  --nms_iou 0.5 \
  > highlights_run.log 2>&1 &
```
- `--match_feat`: path to pre-extracted `.npz` (skips on-the-fly extraction — much faster)
- `--classes`: uses dataset query videos by class name (six=28, four=2, etc.)
- Output: `highlights_output/highlights.mp4`, `clips/`, `summary.json`

#### On-the-fly feature extraction (slower, for new videos without pre-extracted features)
Same command but omit `--match_feat`. Extracts per 2-second segment on the fly.

#### Using specific query videos instead of dataset classes
```bash
  --query_videos "six:path/q1.mp4,path/q2.mp4" "four:path/q3.mp4"
```

**Results (video.mp4, 8 highlights found):**
- six: 02:19:08, 05:31:48, 03:26:23, 02:05:13 (conf up to 1.68)
- four: 02:05:13, 02:21:41, 01:42:24, 01:33:32, 01:50:08 (conf up to 1.13)
- Output: `highlights_output/highlights.mp4`

---

### Using CLIP features — `generate_highlights.py`
**Environment:** `ml_fresh_start`  
**Script:** `generate_highlights.py`

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/user/Desktop/MATR \
  /home/user/miniconda3/envs/ml_fresh_start/bin/python generate_highlights.py \
  --match_video test_videos/cricket_sample.mp4 \
  --query_videos "six:q1.mp4,q2.mp4,q3.mp4" "four:q4.mp4,q5.mp4" \
  --checkpoint results/mr-sportsmr/sportsmoment-clip-4-clip-clip-2026_04_06_08/model_best.ckpt \
  --output_dir highlights_cricket_output \
  --top_k 5 \
  --confidence_threshold 0.3 \
  --nms_iou 0.5 \
  > highlights_cricket_run.log 2>&1 &
```

**Results (cricket_sample.mp4, 5 highlights found):**
- Output: `highlights_output_cricket/highlights.mp4`

---

## Finding Query Videos by Class

Dataset query videos are in `data/sportsmr/raw_query_videos/`. Filename format: `{YouTubeID}__{classID}_{videoID}.mp4`

```bash
# Six (class 28)
ls data/sportsmr/raw_query_videos/ | grep "__28_" | head -5

# Four (class 2)
ls data/sportsmr/raw_query_videos/ | grep "__2_" | head -5
```

**Class ID reference (selected):**
```
2=four, 3=cover drive, 15=pull shot, 21=bouncer,
22=goal, 27=sweep shot, 28=six, 29=upper cut
```

---

## Comparing Training Runs

Use `compare_runs.py` to print a side-by-side loss + metrics table for any number of runs.

```bash
# Two runs with custom labels
python compare_runs.py \
  "results/mr-sportsmr/sportsmoment-clip-4-clip-clip-2026_04_06_08:CL-4 lr=1e-4 saliency=ON" \
  "results/mr-sportsmr/sportsmoment-imagebind-4-imagebind-imagebind-2026_04_06_17:IB-4 lr=5e-5 saliency=OFF"

# Three runs — auto-labels from directory names (no colon needed)
python compare_runs.py \
  results/mr-sportsmr/sportsmoment-imagebind-4-* \
  results/mr-sportsmr/sportsmoment-clip-5-*

# Pass eval.log.txt paths directly
python compare_runs.py path/to/run1/eval.log.txt:"Run 1" path/to/run2/eval.log.txt
```

- Accepts any number of arguments (minimum 1)
- Each arg: `path/to/run_dir` or `path/to/run_dir:"custom label"`
- Auto-finds `eval.log.txt` inside a result directory
- Output columns: `loss_f  loss_g  loss_s | R1@0.5  R1@0.7  mAP  mIoU`

---

## Monitoring

```bash
# ImageBind training
tail -f training_progress.txt

# CLIP training
tail -f training_progress_clip.txt

# Highlights generation
tail -f highlights_run.log

# GPU usage
nvidia-smi
```

---

## Novelty Added vs Original Paper

The original MATR paper used a broken DTW alignment loop. We replaced it with **cross-attention alignment** in `model/align_transformer.py`:

```python
# Each video frame attends to query steps before entering the decoder
attn_scores = torch.einsum('lbd,kbd->lkb', vid_mem, query_mem) / (d ** 0.5)
attn_weights = F.softmax(attn_scores, dim=1)
query_context = torch.einsum('lkb,kbd->lbd', attn_weights, query_mem)
aligned_tensor = vid_mem + query_context  # residual
```

Also enabled saliency losses (`loss_s_inter`, `loss_s_intra`) which were defined but unused in the original code.
