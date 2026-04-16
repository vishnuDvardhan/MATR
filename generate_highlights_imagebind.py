"""
Cricket Match Highlights Generator using MATR + ImageBind features

Uses ImageBind (vision+audio fused, 1024-dim) for both match and query features.
Supports pre-extracted .npz files or on-the-fly extraction for the match video.

Usage:
  python generate_highlights_imagebind.py \
    --match_video match.mp4 \
    --query_feat_dir data/sportsmr/vid_imagebind_query \
    --query_dir data/sportsmr/query_videos_by_class \
    --classes six four \
    --checkpoint results/mr-sportsmr/sportsmoment-imagebind-1-.../model_best.ckpt \
    --output_dir highlights_imagebind/
"""

import os
import sys
import argparse
import json
import random
import subprocess
import tempfile
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import logging
import torch.backends.cudnn as cudnn

logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

CLIP_LENGTH = 2
AUDIO_SAMPLE_RATE = 16000
FFMPEG = next((p for p in [
    '/home/user/miniconda3/envs/ml_fresh_start/bin/ffmpeg',
    'ffmpeg',
] if os.path.exists(p) or p == 'ffmpeg'), 'ffmpeg')
FFPROBE = next((p for p in [
    '/home/user/miniconda3/envs/ml_fresh_start/bin/ffprobe',
    'ffprobe',
] if os.path.exists(p) or p == 'ffprobe'), 'ffprobe')


# ── ImageBind feature extraction ──────────────────────────────────────────────

def get_vision_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def waveform_to_melspec(waveform, sr=AUDIO_SAMPLE_RATE, num_mel_bins=128, target_length=204,
                        mean=-4.268, std=9.138):
    import torchaudio
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr,
        use_energy=False, window_type='hanning',
        num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10)
    T = fbank.shape[0]
    if T < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - T))
    else:
        fbank = fbank[:target_length]
    fbank = (fbank - mean) / std
    return fbank.unsqueeze(0)  # (1, target_length, num_mel_bins)


def extract_audio_segments(video_path, n_clips, clip_length=CLIP_LENGTH, sr=AUDIO_SAMPLE_RATE):
    probe = subprocess.run(
        [FFPROBE, '-v', 'error', '-select_streams', 'a', '-show_entries',
         'stream=codec_type', '-of', 'default=noprint_wrappers=1', video_path],
        capture_output=True, text=True)
    if 'audio' not in probe.stdout:
        return None

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([
            FFMPEG, '-y', '-loglevel', 'error',
            '-i', video_path, '-ac', '1', '-ar', str(sr), tmp_path
        ], check=True, capture_output=True)

        import torchaudio
        waveform, file_sr = torchaudio.load(tmp_path)
        if file_sr != sr:
            waveform = torchaudio.functional.resample(waveform, file_sr, sr)

        total_samples = waveform.shape[1]
        samples_per_clip = sr * clip_length
        segments = []
        for clip_idx in range(n_clips):
            start = int(clip_idx * clip_length * sr)
            end = int(start + samples_per_clip)
            if start >= total_samples:
                seg = torch.zeros(1, int(samples_per_clip))
            else:
                seg = waveform[:, start:min(end, total_samples)]
                if seg.shape[1] < samples_per_clip:
                    seg = torch.nn.functional.pad(seg, (0, int(samples_per_clip) - seg.shape[1]))
            mel = waveform_to_melspec(seg, sr=sr)
            mel = mel.permute(0, 2, 1)  # (1, 128, 204)
            segments.append(mel)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return segments


def extract_imagebind_features(video_path, ib_model, transform, device, clip_length=CLIP_LENGTH):
    from imagebind.models.imagebind_model import ModalityType

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    if duration <= 0 or fps <= 0:
        cap.release()
        return None, 0.0

    n_clips = max(1, int(duration / clip_length))
    frames = []
    for clip_idx in range(n_clips):
        center_time = (clip_idx + 0.5) * clip_length
        frame_idx = min(int(center_time * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(Image.fromarray(frame_rgb)))
    cap.release()

    # Process in batches to avoid OOM on long videos
    batch_size = 64
    vis_feats = []
    for i in range(0, len(frames), batch_size):
        batch = torch.stack(frames[i:i+batch_size]).to(device)
        with torch.no_grad():
            emb = ib_model({ModalityType.VISION: batch})[ModalityType.VISION]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vis_feats.append(emb.cpu().float())
    vis_emb = torch.cat(vis_feats, dim=0)  # (n_clips, 1024)

    audio_segments = extract_audio_segments(video_path, n_clips, clip_length)
    if audio_segments is not None:
        aud_feats = []
        for i in range(0, len(audio_segments), batch_size):
            batch = torch.stack(audio_segments[i:i+batch_size]).to(device)
            with torch.no_grad():
                emb = ib_model({ModalityType.AUDIO: batch})[ModalityType.AUDIO]
                emb = emb / emb.norm(dim=-1, keepdim=True)
            aud_feats.append(emb.cpu().float())
        aud_emb = torch.cat(aud_feats, dim=0)
        fused = (vis_emb + aud_emb) / 2.0
        fused = fused / fused.norm(dim=-1, keepdim=True)
    else:
        fused = vis_emb

    return fused.numpy().astype(np.float32), duration


# ── Multi-query aggregation ───────────────────────────────────────────────────

def nms_windows(windows, confidences, iou_threshold=0.5):
    if not windows:
        return [], []
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    windows = [windows[i] for i in order]
    confidences = [confidences[i] for i in order]
    kept_w, kept_c = [], []
    suppressed = set()
    for i, (w, c) in enumerate(zip(windows, confidences)):
        if i in suppressed:
            continue
        kept_w.append(w)
        kept_c.append(c)
        s1, e1 = sorted(w)
        for j in range(i + 1, len(windows)):
            s2, e2 = sorted(windows[j])
            inter = max(0.0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(j)
    return kept_w, kept_c


def aggregate_multi_query(windows_list, confidences_list, iou_threshold=0.5):
    all_w, all_c = [], []
    for ws, cs in zip(windows_list, confidences_list):
        all_w.extend(ws)
        all_c.extend(cs)
    merged_w, merged_c = nms_windows(all_w, all_c, iou_threshold)
    n_queries = len(windows_list)
    if n_queries == 1:
        return merged_w, merged_c
    boosted_c = []
    for (w, c) in zip(merged_w, merged_c):
        s1, e1 = sorted(w)
        votes = 0
        for ws in windows_list:
            for w2 in ws:
                s2, e2 = sorted(w2)
                inter = max(0.0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter
                if union > 0 and inter / union > iou_threshold:
                    votes += 1
                    break
        boosted_c.append(c * (1.0 + votes / n_queries))
    return merged_w, boosted_c


# ── MATR model ────────────────────────────────────────────────────────────────

def load_matr_model(checkpoint_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main.config import TestOptions, setup_model
    import json as _json

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    saved_opt = ckpt['opt']

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    opt_json_path = os.path.join(ckpt_dir, 'opt.json')
    opt_dict = vars(saved_opt).copy()
    if opt_dict.get('dset_name', '') not in ('actnet', 'sportsmr'):
        opt_dict['dset_name'] = 'actnet'
    with open(opt_json_path, 'w') as f:
        _json.dump(opt_dict, f)

    sys.argv = ['inference', '--resume', checkpoint_path]
    opt = TestOptions().parse()
    if getattr(opt, 'use_tef', False):
        opt.v_feat_dim -= 2
    cudnn.benchmark = True
    model, _, _, _ = setup_model(opt)
    model.eval()
    logger.info(f"Loaded checkpoint (epoch {ckpt['epoch']}, trained on {saved_opt.dset_name})")
    return model, opt


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference_window(model, vid_chunk, src_txt, src_txt_mask, device,
                         win_start_sec, clip_length=CLIP_LENGTH):
    from utils.basic_utils import l2_normalize_np_array

    vid = torch.from_numpy(l2_normalize_np_array(vid_chunk))
    ctx_l = vid.shape[0]
    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef_ed = tef_st + 1.0 / ctx_l
    tef = torch.stack([tef_st, tef_ed], dim=1)
    vid = torch.cat([vid, tef], dim=1)

    timestamp = ((torch.arange(0, ctx_l) + clip_length / 2) / ctx_l).unsqueeze(1).repeat(1, 2)
    src_vid = vid.unsqueeze(0).to(device)
    src_vid_mask = torch.ones(1, ctx_l).to(device)

    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt,
                       src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    pred_spans = output['pred_spans'][0].cpu()
    pred_logits = output['pred_logits'][0].cpu()
    win_duration = ctx_l * clip_length
    pred_windows_local = (pred_spans + timestamp) * win_duration
    pred_windows_global = pred_windows_local + win_start_sec

    k = min(20, pred_logits.shape[0])
    top_vals, top_idx = torch.topk(pred_logits.flatten(), k=k)
    return pred_windows_global[top_idx].tolist(), top_vals.tolist()


def run_inference(model, vid_feats, txt_feats, device, clip_length=CLIP_LENGTH,
                  window_size=75, stride=38):
    from utils.basic_utils import l2_normalize_np_array

    txt = torch.from_numpy(l2_normalize_np_array(txt_feats))
    src_txt = txt.unsqueeze(0).to(device)
    src_txt_mask = torch.ones(1, src_txt.shape[1]).to(device)

    total_segs = vid_feats.shape[0]
    all_windows, all_confidences = [], []
    starts = list(range(0, total_segs, stride))
    if starts[-1] + window_size < total_segs:
        starts.append(total_segs - window_size)

    for w_start in starts:
        w_end = min(w_start + window_size, total_segs)
        chunk = vid_feats[w_start:w_end]
        win_start_sec = w_start * clip_length
        ws, cs = run_inference_window(model, chunk, src_txt, src_txt_mask,
                                      device, win_start_sec, clip_length)
        all_windows.extend(ws)
        all_confidences.extend(cs)
    return all_windows, all_confidences


# ── Video trimming ────────────────────────────────────────────────────────────

def trim_clip(input_video, start_sec, end_sec, output_path):
    duration = max(1.0, end_sec - start_sec)
    subprocess.run([
        FFMPEG, '-y', '-loglevel', 'error',
        '-ss', str(max(0.0, start_sec)),
        '-i', input_video,
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ], check=True)


def stitch_clips(clip_paths, output_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_path = f.name
    try:
        subprocess.run([
            FFMPEG, '-y', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', list_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ], check=True)
    finally:
        os.unlink(list_path)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_video', required=True)
    parser.add_argument('--match_feat', default=None,
                        help='Pre-extracted ImageBind .npz for match video (optional, extracted on-the-fly if not given)')
    parser.add_argument('--query_feat_dir', required=True,
                        help='Dir of pre-extracted ImageBind query features (vid_imagebind_query)')
    parser.add_argument('--query_dir', default=None,
                        help='Root folder of class-organised query videos')
    parser.add_argument('--classes', nargs='+', default=[])
    parser.add_argument('--query_videos', nargs='+', default=[],
                        help='label:path1,path2,... or label:/folder/')
    parser.add_argument('--checkpoint', required=True,
                        help='ImageBind-trained MATR checkpoint')
    parser.add_argument('--output_dir', default='highlights_imagebind')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--min_duration', type=float, default=2.0)
    parser.add_argument('--max_duration', type=float, default=60.0)
    parser.add_argument('--nms_iou', type=float, default=0.5)
    parser.add_argument('--confidence_threshold', type=float, default=0.0)
    parser.add_argument('--num_queries', type=int, default=5,
                        help='Random query videos to sample per class for multi-query inference')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Build query list
    queries = []
    if args.query_dir:
        query_dir = args.query_dir.rstrip('/')
        class_names = args.classes if args.classes else sorted(os.listdir(query_dir))
        for label in class_names:
            class_folder = os.path.join(query_dir, label)
            if not os.path.isdir(class_folder):
                logger.warning(f"Class folder not found: {class_folder}")
                continue
            paths = sorted([os.path.join(class_folder, f)
                            for f in os.listdir(class_folder)
                            if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])
            if paths:
                queries.append((label, paths))

    for item in args.query_videos:
        if ':' not in item:
            raise ValueError(f"Query videos must be label:path, got: {item}")
        label, paths_str = item.split(':', 1)
        if os.path.isdir(paths_str):
            paths = sorted([os.path.join(paths_str, f)
                            for f in os.listdir(paths_str)
                            if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])
        else:
            paths = [p.strip() for p in paths_str.split(',') if p.strip()]
        queries.append((label, paths))

    if not queries:
        raise ValueError("No queries specified. Use --query_dir or --query_videos.")
    logger.info(f"Queries: {[(l, len(ps)) for l, ps in queries]}")

    # Load match video features
    if args.match_feat and os.path.exists(args.match_feat):
        logger.info(f"Loading match features from: {args.match_feat}")
        vid_feats = np.load(args.match_feat)["features"].astype(np.float32)
        cap = cv2.VideoCapture(args.match_video)
        match_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        logger.info(f"Match: {format_time(match_duration)}, {vid_feats.shape[0]} segments")
    else:
        cache_path = os.path.splitext(args.match_video)[0] + "_imagebind_feats.npz"
        if os.path.exists(cache_path):
            logger.info(f"Loading cached match features: {cache_path}")
            cached = np.load(cache_path)
            vid_feats, match_duration = cached["features"].astype(np.float32), float(cached["duration"])
            logger.info(f"Match: {format_time(match_duration)}, {vid_feats.shape[0]} segments (cached)")
        else:
            logger.info("Loading ImageBind model for match feature extraction...")
            from imagebind.models import imagebind_model as ib_module
            ib_model = ib_module.imagebind_huge(pretrained=True)
            ib_model.eval().to(device)
            transform = get_vision_transform()
            logger.info(f"Extracting ImageBind features for: {args.match_video}")
            vid_feats, match_duration = extract_imagebind_features(
                args.match_video, ib_model, transform, device)
            np.savez_compressed(cache_path, features=vid_feats, duration=match_duration)
            logger.info(f"Saved to {cache_path}")
            del ib_model
            torch.cuda.empty_cache()

    # Load MATR model
    logger.info("Loading MATR model...")
    model, opt = load_matr_model(args.checkpoint)
    model = model.to(device)

    # Inference per query class
    all_windows = []
    results_summary = {}

    for label, query_paths in queries:
        logger.info(f"\n=== Query: '{label}' ({len(query_paths)} video(s)) ===")

        # Load pre-extracted ImageBind query features
        all_q_feats = []
        for query_path in query_paths:
            vid_id = os.path.splitext(os.path.basename(query_path))[0]
            npz_path = os.path.join(args.query_feat_dir, f"{vid_id}.npz")
            if os.path.exists(npz_path):
                all_q_feats.append(np.load(npz_path)["features"].astype(np.float32))

        if not all_q_feats:
            logger.warning(f"  No ImageBind features found for '{label}', skipping.")
            continue

        # Randomly sample num_queries for multi-query inference
        sampled = random.sample(all_q_feats, min(args.num_queries, len(all_q_feats)))
        logger.info(f"  Running {len(sampled)} query passes (from {len(all_q_feats)} available)")

        windows_list, confidences_list = [], []
        for qi, q_feats in enumerate(sampled):
            ws, cs = run_inference(model, vid_feats, q_feats, device)
            windows_list.append(ws)
            confidences_list.append(cs)
            logger.info(f"    Query {qi+1}/{len(sampled)}: {len(ws)} raw windows")

        windows, confidences = aggregate_multi_query(windows_list, confidences_list,
                                                     iou_threshold=args.nms_iou)

        saved = []
        for window, conf in zip(windows, confidences):
            if len(saved) >= args.top_k:
                break
            if conf < args.confidence_threshold:
                continue
            start, end = sorted(window)
            start = max(0.0, start)
            end = min(match_duration, end)
            dur = end - start
            if dur < args.min_duration or dur > args.max_duration:
                continue
            saved.append((start, end, conf))
            all_windows.append((start, end, label, conf))
            logger.info(f"  {format_time(start)} – {format_time(end)} ({dur:.1f}s, conf={conf:.3f})")

        results_summary[label] = [
            {'start': format_time(s), 'end': format_time(e),
             'start_sec': round(s, 2), 'end_sec': round(e, 2),
             'duration_sec': round(e - s, 2), 'confidence': round(float(c), 4)}
            for s, e, c in saved
        ]
        logger.info(f"  Kept {len(saved)} windows for '{label}'")

    # Merge overlapping windows across classes
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged]

    raw_intervals = [(s, e) for s, e, _, _ in all_windows]
    merged = merge_intervals(raw_intervals)
    logger.info(f"\nMerged {len(all_windows)} windows → {len(merged)} unique intervals")

    # Trim and stitch
    clips_dir = os.path.join(args.output_dir, 'clips')
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []
    merged_summary = []

    for i, (start, end) in enumerate(merged):
        dur = end - start
        clip_path = os.path.join(clips_dir, f"{i+1:02d}_{format_time(start).replace(':','-')}.mp4")
        logger.info(f"  [{i+1}] {format_time(start)} – {format_time(end)} ({dur:.1f}s)")
        try:
            trim_clip(args.match_video, start, end, clip_path)
            clip_paths.append(clip_path)
            merged_summary.append({
                'start': format_time(start), 'end': format_time(end),
                'start_sec': round(start, 2), 'end_sec': round(end, 2),
                'duration_sec': round(dur, 2)
            })
        except Exception as e:
            logger.warning(f"  Trim failed: {e}")

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'per_query': results_summary, 'merged_highlights': merged_summary}, f, indent=2)

    if clip_paths:
        reel_path = os.path.join(args.output_dir, 'highlights.mp4')
        logger.info(f"\nStitching {len(clip_paths)} clips → {reel_path}")
        stitch_clips(clip_paths, reel_path)

        print(f"\n{'='*60}")
        print(f"HIGHLIGHTS ({len(clip_paths)} clips)")
        print(f"{'='*60}")
        for c in merged_summary:
            print(f"  {c['start']} – {c['end']} ({c['duration_sec']:.1f}s)")
        print(f"\nHighlight reel → {reel_path}")
        print(f"Summary        → {summary_path}")
    else:
        logger.warning("No clips found. Try lowering --confidence_threshold or --min_duration.")


if __name__ == '__main__':
    main()
