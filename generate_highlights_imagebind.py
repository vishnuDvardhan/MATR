"""
Highlight Generator for Full Match Videos using MATR + ImageBind

For a full cricket/football match (hours long), finds moments matching each
query class (e.g. "six", "four", "goal") and stitches them into a highlight reel.

Strategy
--------
  - Model was trained on ~2.5 min (75-clip) windows. Full matches are hours long.
  - We slide a 75-clip (150s) window over the match with 50% overlap.
  - For each window, we run model inference against multiple query videos per class.
  - Window-local predicted timestamps are shifted to global match timestamps.
  - Global NMS merges duplicate detections; multi-query voting boosts confidence.

Usage
-----
  # Use pre-extracted imagebind query features grouped by class ID.
  # Class IDs follow SportsMoments: 28=six, 2=four, 22=goal, etc.

  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/path/to/MATR \\
    /path/to/envs/tdl-project/bin/python generate_highlights_imagebind.py \\
      --match_video  my_match.mp4 \\
      --checkpoint   "data/sportsmr/tdl project /imagebind-best.ckpt" \\
      --query_feat_dir data/sportsmr/vid_imagebind_query \\
      --class_ids    28 2 22 \\
      --output_dir   highlights_out \\
      --top_k        5 \\
      --confidence_threshold 0.1

  # Class ID → label mapping (SportsMoments):
  # 1=lbw  2=four  3=cover_drive  4=ducking  5=run_out  6=corner_kick
  # 7=goal_kick  8=full_toss  9=bowled  10=yorker  11=stumping  12=red_card
  # 13=sliding_tackle  14=wide_ball  15=pull_shot  16=throw  17=offside
  # 18=catching  19=penalty  20=substitute  21=bouncer  22=goal  23=save
  # 24=yellow_card  25=free_kick  26=header  27=sweep_shot  28=six  29=upper_cut
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import random
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CLIP_LENGTH = 2.0          # seconds per 2-second clip segment
WINDOW_CLIPS = 75          # model max_v_l: 75 clips = 150s per window
STRIDE_CLIPS = 38          # ~50% overlap between consecutive windows
AUDIO_SR = 16000

FFMPEG = next(p for p in [
    '/usr/bin/ffmpeg',
    '/home/azhan1303/anaconda3/envs/tdl-project/bin/ffmpeg',
] if os.path.exists(p))
FFPROBE = next(p for p in [
    '/usr/bin/ffprobe',
    '/home/azhan1303/anaconda3/envs/tdl-project/bin/ffprobe',
] if os.path.exists(p))

CLASS_NAMES = {
    1: "lbw", 2: "four", 3: "cover_drive", 4: "ducking_bouncer", 5: "run_out",
    6: "corner_kick", 7: "goal_kick", 8: "full_toss", 9: "bowled", 10: "yorker",
    11: "stumping", 12: "red_card", 13: "sliding_tackle", 14: "wide_ball",
    15: "pull_shot", 16: "throw", 17: "offside", 18: "catching", 19: "penalty",
    20: "substitute", 21: "bouncer", 22: "goal", 23: "save", 24: "yellow_card",
    25: "free_kick", 26: "header", 27: "sweep_shot", 28: "six", 29: "upper_cut",
}
# Reverse: name → id  (also accept aliases with spaces, e.g. "cover drive" → "cover_drive")
NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
NAME_TO_ID.update({v.replace('_', ' '): k for k, v in CLASS_NAMES.items()})


# ── ImageBind feature extraction ──────────────────────────────────────────────


def waveform_to_melspec(waveform, sr=AUDIO_SR, num_mel_bins=128, target_length=204,
                        mean=-4.268, std=9.138):
    import torchaudio
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10)
    T = fbank.shape[0]
    if T < target_length:
        fbank = F.pad(fbank, (0, 0, 0, target_length - T))
    else:
        fbank = fbank[:target_length]
    fbank = (fbank - mean) / std
    return fbank.unsqueeze(0)  # (1, target_length, num_mel_bins)


def extract_audio_segments(video_path, n_clips):
    probe = subprocess.run(
        [FFPROBE, '-v', 'error', '-select_streams', 'a',
         '-show_entries', 'stream=codec_type',
         '-of', 'default=noprint_wrappers=1', video_path],
        capture_output=True, text=True)
    if 'audio' not in probe.stdout:
        return None

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger.info(f"  Converting audio to WAV (this may take a few minutes for long videos)...")
        subprocess.run([
            FFMPEG, '-y', '-loglevel', 'error', '-stats',
            '-i', video_path, '-ac', '1', '-ar', str(AUDIO_SR), tmp_path
        ], check=True)

        import torchaudio
        waveform, file_sr = torchaudio.load(tmp_path)
        if file_sr != AUDIO_SR:
            waveform = torchaudio.functional.resample(waveform, file_sr, AUDIO_SR)

        total_samples = waveform.shape[1]
        samples_per_clip = int(AUDIO_SR * CLIP_LENGTH)
        segments = []
        for ci in range(n_clips):
            start = ci * samples_per_clip
            end = start + samples_per_clip
            if start >= total_samples:
                seg = torch.zeros(1, samples_per_clip)
            else:
                seg = waveform[:, start:min(end, total_samples)]
                if seg.shape[1] < samples_per_clip:
                    seg = F.pad(seg, (0, samples_per_clip - seg.shape[1]))
            mel = waveform_to_melspec(seg).permute(0, 2, 1)  # (1, 128, 204)
            segments.append(mel)
        return segments
    except Exception as e:
        logger.warning(f"Audio extraction failed for {video_path}: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_imagebind_features(video_path, ib_model, _transform, device, batch_size=16):
    """Extract per-clip (2s) ImageBind fused features. Returns (n_clips, 1024).

    Uses decord for fast frame seeking + streaming batch processing so frames
    are never all held in RAM at once — decode a batch, run GPU inference,
    discard, repeat.
    """
    from imagebind.models.imagebind_model import ModalityType
    import decord
    decord.bridge.set_bridge('torch')

    # ── video metadata ───────────────────────────────────────────────────────
    # Always use CPU for video decoding to avoid GPU thermal pressure.
    # GPU is reserved for ImageBind inference only.
    ctx = decord.cpu()
    vr = decord.VideoReader(video_path, ctx=ctx, num_threads=4)

    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps if fps > 0 else 0
    n_clips = max(1, int(duration / CLIP_LENGTH))

    # Center frame index for each 2-second clip
    frame_indices = [
        min(int((ci + 0.5) * CLIP_LENGTH * fps), total_frames - 1)
        for ci in range(n_clips)
    ]

    logger.info(f"  Video: {duration:.1f}s, {n_clips} clips, fps={fps:.2f}")

    # ── normalisation constants ───────────────────────────────────────────────
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).view(3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=device).view(3, 1, 1)

    def preprocess_batch_gpu(frames_thwc):
        """frames_thwc: uint8 torch tensor (B, H, W, C) on GPU → (B, 3, 224, 224) float"""
        x = frames_thwc.permute(0, 3, 1, 2).float().to(device) / 255.0  # (B,3,H,W)
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        # Centre-crop to 224×224
        _, _, h, w = x.shape
        top  = (h - 224) // 2
        left = (w - 224) // 2
        x = x[:, :, top:top+224, left:left+224]
        x = (x - mean) / std
        return x

    # ── streaming batch: decode → GPU inference → discard ────────────────────
    vis_feats = []
    for batch_start in tqdm(range(0, n_clips, batch_size),
                            desc="  extracting vision features", leave=False):
        batch_idx = frame_indices[batch_start: batch_start + batch_size]
        try:
            frames = vr.get_batch(batch_idx)   # (B, H, W, C) uint8 on CPU
        except Exception as e:
            logger.warning(f"  Skipping corrupt batch at clip {batch_start}: {e}")
            n_bad = len(batch_idx)
            vis_feats.append(torch.zeros(n_bad, 1024))
            continue
        frames = frames.to(device)          # move to GPU for ImageBind inference
        imgs = preprocess_batch_gpu(frames)
        with torch.no_grad():
            emb = ib_model({ModalityType.VISION: imgs})[ModalityType.VISION]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vis_feats.append(emb.cpu().float())
        del frames, imgs
        torch.cuda.synchronize()
        time.sleep(0.05)  # brief pause to limit GPU thermal load

    vis_emb = torch.cat(vis_feats, dim=0)

    # ── audio ─────────────────────────────────────────────────────────────────
    audio_segs = extract_audio_segments(video_path, n_clips)

    if audio_segs is not None:
        aud_feats = []
        for i in tqdm(range(0, n_clips, batch_size),
                      desc="  extracting audio features", leave=False):
            batch = torch.stack(audio_segs[i:i+batch_size]).to(device)
            with torch.no_grad():
                emb = ib_model({ModalityType.AUDIO: batch})[ModalityType.AUDIO]
                emb = emb / emb.norm(dim=-1, keepdim=True)
            aud_feats.append(emb.cpu().float())
        aud_emb = torch.cat(aud_feats, dim=0)
        fused = (vis_emb + aud_emb) / 2.0
        fused = fused / fused.norm(dim=-1, keepdim=True)
    else:
        logger.info("  No audio track; using vision-only features.")
        fused = vis_emb

    return fused.numpy(), duration


# ── Model loading ─────────────────────────────────────────────────────────────

def load_matr_model(checkpoint_path, device):
    """Load MATR from imagebind-best.ckpt. Reads config from the checkpoint's 'opt'."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib, argparse

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    saved_opt = ckpt['opt']

    # Write a temporary opt.json next to the checkpoint so TestOptions can find it
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    opt_json_path = os.path.join(ckpt_dir, 'opt.json')
    opt_dict = vars(saved_opt).copy()
    if opt_dict.get('dset_name', '') not in ('actnet', 'sportsmr'):
        opt_dict['dset_name'] = 'sportsmr'
    with open(opt_json_path, 'w') as f:
        json.dump(opt_dict, f)

    from main.config import TestOptions, setup_model
    sys.argv = ['inference', '--resume', checkpoint_path]
    opt = TestOptions().parse()

    # v_feat_dim in the saved opt already includes the +2 TEF;
    # setup_model/config will add 2 again when use_tef=True — compensate.
    if getattr(opt, 'use_tef', False) and opt.v_feat_dim > 1024:
        opt.v_feat_dim -= 2

    # model_id in checkpoint may be uppercase ('MATR') but module file is lowercase
    opt.model_id = opt.model_id.lower()

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    model, _, _, _ = setup_model(opt)
    model.eval().to(device)
    logger.info(f"Loaded checkpoint (epoch {ckpt['epoch']}, exp={saved_opt.exp_id})")
    return model, opt


# ── Inference helpers ─────────────────────────────────────────────────────────

def l2_norm(arr):
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def infer_window(model, vid_chunk, src_txt, src_txt_mask, device, win_start_sec):
    """
    Run model on one video window (up to WINDOW_CLIPS clips).
    Returns (windows_global_sec, scores) both as plain Python lists.
    """
    n = vid_chunk.shape[0]  # actual clips in this window

    # TEF normalized 0→1 within this window
    tef_st = torch.arange(0, n, dtype=torch.float32) / n
    tef_ed = tef_st + 1.0 / n
    tef = torch.stack([tef_st, tef_ed], dim=1)  # (n, 2)

    vid_norm = torch.from_numpy(l2_norm(vid_chunk))  # (n, 1024)
    vid = torch.cat([vid_norm, tef], dim=1)           # (n, 1026)

    # timestamp positions (used to decode predicted spans)
    timestamp = ((torch.arange(0, n) + CLIP_LENGTH / 2) / n).unsqueeze(1).repeat(1, 2)

    src_vid = vid.unsqueeze(0).to(device)            # (1, n, 1026)
    src_vid_mask = torch.ones(1, n).to(device)

    with torch.no_grad():
        out = model(src_vid=src_vid, src_txt=src_txt,
                    src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    # pred_spans: (1, n, 2) residuals in normalized coords
    # pred_logits: (1, n, 1) sigmoid foreground scores
    pred_spans  = out['pred_spans'][0].cpu()    # (n, 2)
    pred_logits = out['pred_logits'][0].cpu()   # (n, 1) or (n,)
    saliency    = out['saliency_scores'][0].cpu()  # (n,) cosine sim scores

    # Decode: span = timestamp + residual, then scale to seconds
    win_dur = n * CLIP_LENGTH
    spans_local = (timestamp + pred_spans) * win_dur   # (n, 2) seconds in window
    spans_global = spans_local + win_start_sec         # shift to global time

    # Combined score: foreground logit + saliency (same as eval_mode='add')
    fg_scores = pred_logits.squeeze(-1)  # (n,)
    combined  = (fg_scores + saliency).tolist()

    windows = spans_global.tolist()   # list of [start, end]
    return windows, combined


def sliding_window_inference(model, vid_feats, q_feats, device,
                              window_clips=WINDOW_CLIPS, stride=STRIDE_CLIPS):
    """
    Slide a window over the full match and collect predictions.
    Returns all raw (window, score) pairs in global seconds.
    """
    q_norm = l2_norm(q_feats)
    src_txt = torch.from_numpy(q_norm).unsqueeze(0).to(device)   # (1, Lq, 1024)
    src_txt_mask = torch.ones(1, src_txt.shape[1]).to(device)

    total = vid_feats.shape[0]
    starts = list(range(0, max(1, total - window_clips + 1), stride))
    # Ensure we cover the tail of the video
    if not starts or starts[-1] + window_clips < total:
        starts.append(max(0, total - window_clips))

    all_wins, all_scores = [], []
    for w_start in tqdm(starts, desc="  sliding windows", leave=False):
        chunk = vid_feats[w_start: w_start + window_clips]
        win_start_sec = w_start * CLIP_LENGTH
        ws, cs = infer_window(model, chunk, src_txt, src_txt_mask, device, win_start_sec)
        all_wins.extend(ws)
        all_scores.extend(cs)

    return all_wins, all_scores


# ── NMS & aggregation ─────────────────────────────────────────────────────────

def nms(windows, scores, iou_threshold):
    if not windows:
        return [], []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    windows  = [windows[i] for i in order]
    scores   = [scores[i] for i in order]
    kept_w, kept_s = [], []
    suppressed = set()
    for i, (w, s) in enumerate(zip(windows, scores)):
        if i in suppressed:
            continue
        kept_w.append(w)
        kept_s.append(s)
        s1, e1 = sorted(w)
        for j in range(i + 1, len(windows)):
            if j in suppressed:
                continue
            s2, e2 = sorted(windows[j])
            inter = max(0.0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(j)
    return kept_w, kept_s


def multi_query_aggregate(windows_list, scores_list, iou_threshold, pre_nms_topk=2000):
    """Pool predictions from N query videos, NMS, then boost windows with
    multiple-query agreement (up to 2× boost).
    pre_nms_topk: keep only top-k windows per query before pooling to keep
    NMS tractable for long videos."""
    all_w, all_s = [], []
    for wl, sl in zip(windows_list, scores_list):
        if len(sl) > pre_nms_topk:
            # keep top-k per query before pooling
            top_idx = sorted(range(len(sl)), key=lambda i: sl[i], reverse=True)[:pre_nms_topk]
            all_w.extend(wl[i] for i in top_idx)
            all_s.extend(sl[i] for i in top_idx)
        else:
            all_w.extend(wl)
            all_s.extend(sl)
    merged_w, merged_s = nms(all_w, all_s, iou_threshold)

    n_queries = len(windows_list)
    if n_queries <= 1:
        return merged_w, merged_s

    boosted = []
    for w, s in zip(merged_w, merged_s):
        s1, e1 = sorted(w)
        votes = 0
        for wl in windows_list:
            for w2 in wl:
                s2, e2 = sorted(w2)
                inter = max(0.0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter
                if union > 0 and inter / union > iou_threshold:
                    votes += 1
                    break
        boosted.append(s * (1.0 + votes / n_queries))
    return merged_w, boosted


# ── Video cutting & stitching ─────────────────────────────────────────────────

def trim_clip(input_video, start_sec, end_sec, output_path, caption=None):
    dur = max(1.0, end_sec - start_sec)
    subprocess.run([
        FFMPEG, '-y', '-loglevel', 'error',
        '-ss', str(max(0.0, start_sec - 1.0)),   # seek 1s before for keyframe
        '-i', input_video,
        '-ss', '1.0',                             # trim the extra seek second
        '-t', str(dur),
        '-c:v', 'libx264', '-c:a', 'aac',
        '-movflags', '+faststart',
        output_path
    ], check=True)


def stitch_clips(clip_paths, output_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_file = f.name
    try:
        subprocess.run([
            FFMPEG, '-y', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ], check=True)
    finally:
        os.unlink(list_file)


def fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate highlights from a full match using MATR + ImageBind")
    parser.add_argument('--match_video', required=True,
                        help='Path to full match video (mp4/mkv/etc.)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to imagebind-best.ckpt')
    parser.add_argument('--query_feat_dir', required=True,
                        help='Dir of pre-extracted ImageBind query .npz files '
                             '(e.g. data/sportsmr/vid_imagebind_query)')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Classes to search for — use names or IDs. '
                             'Examples: six four bowled  OR  28 2 9  OR mixed: six 2 goal. '
                             'Valid names: ' + ', '.join(CLASS_NAMES.values()))
    parser.add_argument('--output_dir', default='highlights_imagebind')
    parser.add_argument('--match_feat_cache', default=None,
                        help='Path to pre-extracted match .npz features. '
                             'If not set, features are extracted on-the-fly and '
                             'cached next to the match video.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Max highlight clips per class')
    parser.add_argument('--num_queries', type=int, default=5,
                        help='Max query videos to sample per class for multi-query voting')
    parser.add_argument('--confidence_threshold', type=float, default=0.05,
                        help='Min combined score to keep a window')
    parser.add_argument('--nms_iou', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--min_clip_dur', type=float, default=2.0,
                        help='Min duration (s) of a highlight clip')
    parser.add_argument('--max_clip_dur', type=float, default=30.0,
                        help='Max duration (s) of a highlight clip')
    parser.add_argument('--window_clips', type=int, default=WINDOW_CLIPS,
                        help='Primary sliding window size in clips (default 75 = 150s)')
    parser.add_argument('--stride_clips', type=int, default=STRIDE_CLIPS,
                        help='Stride for primary window in clips (default 38 = 76s)')
    parser.add_argument('--window_passes', type=str,
                        default='75:38,50:20,30:10',
                        help='Comma-separated list of window:stride pairs for multi-scale '
                             'inference (default "75:38,50:20,30:10"). Each pass uses a '
                             'different window size so short/boundary moments are not missed.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Step 1: extract / load match video features ──────────────────────────
    cache_path = args.match_feat_cache or (
        os.path.splitext(args.match_video)[0] + "_imagebind_feats.npz")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached match features: {cache_path}")
        d = np.load(cache_path)
        vid_feats, match_duration = d['features'].astype(np.float32), float(d['duration'])
        logger.info(f"Match: {fmt(match_duration)}, {vid_feats.shape[0]} clips")
    else:
        logger.info("Loading ImageBind model for match feature extraction...")
        from imagebind.models import imagebind_model as ibm
        ib = ibm.imagebind_huge(pretrained=True)
        ib.eval().to(device)
        logger.info(f"Extracting ImageBind features for: {args.match_video}")
        vid_feats, match_duration = extract_imagebind_features(
            args.match_video, ib, None, device)
        np.savez_compressed(cache_path, features=vid_feats, duration=match_duration)
        logger.info(f"Saved match features → {cache_path}  shape={vid_feats.shape}")
        del ib  # free GPU memory before loading MATR

    # ── Step 2: load MATR ────────────────────────────────────────────────────
    logger.info("Loading MATR model...")
    model, opt = load_matr_model(args.checkpoint, device)

    # ── Resolve class names / IDs ─────────────────────────────────────────────
    class_ids = []
    for token in args.classes:
        token_lower = token.lower().replace('-', '_')
        if token_lower.isdigit():
            cls_id = int(token_lower)
            if cls_id not in CLASS_NAMES:
                raise ValueError(f"Unknown class ID: {cls_id}. Valid IDs: {sorted(CLASS_NAMES)}")
            class_ids.append(cls_id)
        else:
            # Try exact match, then with underscores
            resolved = NAME_TO_ID.get(token_lower) or NAME_TO_ID.get(token_lower.replace(' ', '_'))
            if resolved is None:
                valid = ', '.join(sorted(NAME_TO_ID.keys()))
                raise ValueError(f"Unknown class name '{token}'. Valid names: {valid}")
            class_ids.append(resolved)
    logger.info(f"Classes to detect: {[(CLASS_NAMES[i], i) for i in class_ids]}")

    # ── Step 3: per-class inference ──────────────────────────────────────────
    all_highlights = []   # [(start, end, class_name, score), ...]
    per_class_summary = {}

    for cls_id in class_ids:
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        logger.info(f"\n{'='*60}")
        logger.info(f"Class {cls_id} — {cls_name}")

        # Gather pre-extracted query features for this class
        q_files = sorted([
            f for f in os.listdir(args.query_feat_dir)
            if f.endswith('.npz') and f"__{cls_id}_" in f
        ])
        if not q_files:
            logger.warning(f"  No query features found for class {cls_id} in {args.query_feat_dir}")
            continue
        logger.info(f"  Found {len(q_files)} query videos for class {cls_id}")

        # Sample up to num_queries
        sampled = random.sample(q_files, min(args.num_queries, len(q_files)))
        logger.info(f"  Using {len(sampled)} sampled queries")

        # Parse multi-scale window passes: "75:38,50:20,30:10"
        passes = []
        for token in args.window_passes.split(','):
            w, s = token.strip().split(':')
            passes.append((int(w), int(s)))
        logger.info(f"  Window passes: {passes}")

        windows_list, scores_list = [], []
        for qi, qf in enumerate(sampled):
            q_feats = np.load(os.path.join(args.query_feat_dir, qf))['features'].astype(np.float32)
            logger.info(f"  Query {qi+1}/{len(sampled)}: {qf}  shape={q_feats.shape}")
            # Run all window-size passes and pool their raw predictions
            all_ws, all_cs = [], []
            for win, stride in passes:
                ws, cs = sliding_window_inference(
                    model, vid_feats, q_feats, device,
                    window_clips=win, stride=stride)
                all_ws.extend(ws)
                all_cs.extend(cs)
            windows_list.append(all_ws)
            scores_list.append(all_cs)
            logger.info(f"    → {len(all_ws)} raw candidate windows across {len(passes)} passes")

        # Aggregate + global NMS
        windows, scores = multi_query_aggregate(windows_list, scores_list, args.nms_iou)
        # Sort by score descending
        pairs = sorted(zip(windows, scores), key=lambda x: x[1], reverse=True)

        kept = []
        for w, s in pairs:
            if len(kept) >= args.top_k:
                break
            if s < args.confidence_threshold:
                continue
            start, end = float(min(w)), float(max(w))
            start = max(0.0, start)
            end   = min(match_duration, end)
            dur   = end - start
            if dur < args.min_clip_dur or dur > args.max_clip_dur:
                continue
            kept.append((start, end, s))
            all_highlights.append((start, end, cls_name, s))
            logger.info(f"  ✓ {fmt(start)} – {fmt(end)}  ({dur:.1f}s, score={s:.4f})")

        per_class_summary[cls_name] = [
            {'start': fmt(s), 'end': fmt(e),
             'start_sec': round(s, 2), 'end_sec': round(e, 2),
             'duration_sec': round(e - s, 2), 'score': round(float(sc), 4)}
            for s, e, sc in kept
        ]
        logger.info(f"  → Kept {len(kept)} moments for '{cls_name}'")

    # ── Step 4: cut clips ────────────────────────────────────────────────────
    if not all_highlights:
        logger.warning("No highlights found. Try lowering --confidence_threshold.")
        return

    # Cross-class deduplication: if two predictions from different classes
    # overlap significantly, keep only the highest-scoring one.
    def cross_class_nms(highlights, iou_threshold=0.5):
        ranked = sorted(highlights, key=lambda x: x[3], reverse=True)
        kept = []
        suppressed = set()
        for i, (s1, e1, c1, sc1) in enumerate(ranked):
            if i in suppressed:
                continue
            kept.append(ranked[i])
            for j in range(i + 1, len(ranked)):
                if j in suppressed:
                    continue
                s2, e2, c2, sc2 = ranked[j]
                if c1 == c2:
                    continue  # same-class already handled by per-class NMS
                inter = max(0.0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter
                if union > 0 and inter / union > iou_threshold:
                    suppressed.add(j)
        return kept

    before = len(all_highlights)
    all_highlights = cross_class_nms(all_highlights, iou_threshold=args.nms_iou)
    logger.info(f"\nCross-class dedup: {before} → {len(all_highlights)} predictions")

    # Sort chronologically, merge nearby/overlapping windows
    all_highlights.sort(key=lambda x: x[0])

    def merge_close(intervals, gap=1.0):
        merged = [list(intervals[0])]
        for item in intervals[1:]:
            if item[0] - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], item[1])
                merged[-1][2] = item[2]  # keep last label/score (arbitrary)
                merged[-1][3] = max(merged[-1][3], item[3])
            else:
                merged.append(list(item))
        return [tuple(m) for m in merged]

    merged = merge_close(all_highlights)
    logger.info(f"\nMerged {len(all_highlights)} windows → {len(merged)} clips")

    clips_dir = os.path.join(args.output_dir, 'clips')
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []
    clip_meta  = []

    for i, (start, end, cls_name, score) in enumerate(merged):
        fname = f"{i+1:03d}_{cls_name}_{fmt(start).replace(':','-')}.mp4"
        out_path = os.path.join(clips_dir, fname)
        logger.info(f"  [{i+1}/{len(merged)}] {fmt(start)} – {fmt(end)}  "
                    f"({end-start:.1f}s)  [{cls_name}]")
        try:
            trim_clip(args.match_video, start, end, out_path, caption=cls_name)
            clip_paths.append(out_path)
            clip_meta.append({'start': fmt(start), 'end': fmt(end),
                               'start_sec': round(start, 2), 'end_sec': round(end, 2),
                               'duration_sec': round(end - start, 2),
                               'class': cls_name, 'score': round(float(score), 4)})
        except Exception as e:
            logger.warning(f"  Trim failed: {e}")

    # ── Step 5: stitch highlight reel ────────────────────────────────────────
    if clip_paths:
        reel_path = os.path.join(args.output_dir, 'highlights.mp4')
        logger.info(f"\nStitching {len(clip_paths)} clips → {reel_path}")
        stitch_clips(clip_paths, reel_path)

    # ── Step 6: save summary ─────────────────────────────────────────────────
    summary = {
        'match_video': args.match_video,
        'match_duration': fmt(match_duration),
        'classes': [CLASS_NAMES[i] for i in class_ids],
        'per_class': per_class_summary,
        'highlight_clips': clip_meta,
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HIGHLIGHTS ({len(clip_paths)} clips from {fmt(match_duration)} match)")
    print(f"{'='*60}")
    for c in clip_meta:
        print(f"  [{c['class']:15s}] {c['start']} – {c['end']}  ({c['duration_sec']:.1f}s,  score={c['score']:.4f})")
    print(f"\nHighlight reel → {os.path.join(args.output_dir, 'highlights.mp4')}")
    print(f"Clips dir      → {clips_dir}")
    print(f"Summary JSON   → {summary_path}")


if __name__ == '__main__':
    main()
