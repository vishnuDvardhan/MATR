"""
Cricket Match Highlights Generator using MATR

Given a full match video and query videos (e.g. a clip of a six, a wicket),
finds matching moments in the match and stitches them into a highlight reel.

Usage:
  python generate_highlights.py \
    --match_video match.mp4 \
    --query_videos six:examples/six.mp4 wicket:examples/wicket.mp4 boundary:examples/boundary.mp4 \
    --checkpoint checkpoints/model_best.ckpt \
    --output_dir highlights/
"""

import os
import sys
import argparse
import json
import subprocess
import numpy as np
import torch
import clip
import cv2
from PIL import Image
from tqdm import tqdm
import logging
import torch.backends.cudnn as cudnn

logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

CLIP_MODEL_VERSION = "ViT-B/32"
CLIP_LENGTH = 2  # seconds per segment


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_video_features(video_path, clip_model, preprocess, device, clip_length=CLIP_LENGTH):
    """Extract per-segment CLIP visual features from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    n_clips = max(1, int(duration / clip_length))

    logger.info(f"Video: {duration:.1f}s, extracting {n_clips} segments...")
    frames = []
    for clip_idx in tqdm(range(n_clips), desc="Extracting frames"):
        center_time = (clip_idx + 0.5) * clip_length
        frame_idx = min(int(center_time * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess(Image.fromarray(frame_rgb)))
    cap.release()

    # Batch inference
    batch_size = 64
    all_features = []
    for i in range(0, len(frames), batch_size):
        batch = torch.stack(frames[i:i+batch_size]).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu().float())

    features = torch.cat(all_features, dim=0).numpy()
    return features, duration


def summarize_query_embeddings(all_feats_list):
    """
    Summarize embeddings from multiple query videos into one representative vector.

    Strategy:
      1. Per-video mean — average each video's segments to 1 vector (prevents long
         videos from dominating the global pool).
      2. Self-similarity weighting — weight each per-video vector by its average
         cosine similarity to all others. Prototypical (central) examples get higher
         weight; outliers get downweighted.
      3. Weighted mean → L2 normalize.

    Args:
        all_feats_list: list of np.ndarray, each (n_segs, d)
    Returns:
        summary: np.ndarray (1, d)
    """
    # Step 1: per-video mean
    video_vecs = []
    for feats in all_feats_list:
        v = feats.mean(axis=0)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        video_vecs.append(v)
    video_vecs = np.stack(video_vecs)  # (N, d)

    if len(video_vecs) == 1:
        return video_vecs[:1]

    # Step 2: self-similarity weights
    # sim[i,j] = cosine similarity between video i and j (vecs already L2-normed)
    sim_matrix = video_vecs @ video_vecs.T  # (N, N)
    # Average similarity to all other videos (exclude self-similarity on diagonal)
    np.fill_diagonal(sim_matrix, 0.0)
    weights = sim_matrix.sum(axis=1) / (len(video_vecs) - 1)  # (N,)
    # Shift to positive range and softmax for stable weighting
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()

    # Step 3: weighted mean + normalize
    summary = (weights[:, None] * video_vecs).sum(axis=0)
    norm = np.linalg.norm(summary)
    if norm > 0:
        summary = summary / norm
    return summary[None]  # (1, d)


def nms_windows(windows, confidences, iou_threshold=0.5):
    """Non-maximum suppression on time windows."""
    if not windows:
        return [], []
    # Sort by confidence descending
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
    """
    Pool predictions from multiple query videos for the same label.
    After NMS, windows confirmed by more query videos get a confidence boost.
    """
    # Pool all predictions
    all_w, all_c = [], []
    for ws, cs in zip(windows_list, confidences_list):
        all_w.extend(ws)
        all_c.extend(cs)

    merged_w, merged_c = nms_windows(all_w, all_c, iou_threshold)

    n_queries = len(windows_list)
    if n_queries == 1:
        return merged_w, merged_c

    # Boost confidence proportional to how many query videos agree on each window
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
        # Scale up by fraction of queries that agreed (1.0x → 2.0x boost)
        boosted_c.append(c * (1.0 + votes / n_queries))

    return merged_w, boosted_c


# ── Model ─────────────────────────────────────────────────────────────────────

def load_matr_model(checkpoint_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main.config import TestOptions, setup_model

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    saved_opt = ckpt['opt']

    # Write opt.json next to checkpoint so TestOptions can load it
    import json as _json
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    opt_json_path = os.path.join(ckpt_dir, 'opt.json')
    opt_dict = vars(saved_opt).copy()
    # dset_name must be one of actnet/sportsmr
    if opt_dict.get('dset_name', '') not in ('actnet', 'sportsmr'):
        opt_dict['dset_name'] = 'actnet'
    with open(opt_json_path, 'w') as f:
        _json.dump(opt_dict, f)

    sys.argv = ['inference', '--resume', checkpoint_path]
    opt = TestOptions().parse()
    # v_feat_dim in saved opt already includes TEF (+2); setup_model adds 2 again — undo it
    if getattr(opt, 'use_tef', False):
        opt.v_feat_dim -= 2
    cudnn.benchmark = True
    model, _, _, _ = setup_model(opt)
    model.eval()
    logger.info(f"Loaded checkpoint (epoch {ckpt['epoch']}, trained on {saved_opt.dset_name})")
    return model, opt


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference_window(model, vid_chunk, src_txt, src_txt_mask, device, win_start_sec, clip_length=CLIP_LENGTH):
    """Run inference on a single video window, returning predictions in global seconds."""
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

    pred_spans = output['pred_spans'][0].cpu()   # (ctx_l, 2) local normalized
    pred_logits = output['pred_logits'][0].cpu() # (ctx_l,) or (ctx_l, 1)

    # Convert local normalized coords → global seconds
    win_duration = ctx_l * clip_length
    pred_windows_local = (pred_spans + timestamp) * win_duration   # (ctx_l, 2) in seconds within window
    pred_windows_global = pred_windows_local + win_start_sec       # shift to global timeline

    k = min(20, pred_logits.shape[0])
    top_vals, top_idx = torch.topk(pred_logits.flatten(), k=k)
    top_windows = pred_windows_global[top_idx].tolist()
    top_confidences = top_vals.tolist()
    return top_windows, top_confidences


def run_inference(model, vid_feats, txt_feats, device, clip_length=CLIP_LENGTH, window_size=75, stride=38):
    """
    Sliding-window inference over a long video.
    window_size: number of segments per window (model was trained on max_v_l=75)
    stride: step in segments between windows (50% overlap)
    """
    from utils.basic_utils import l2_normalize_np_array

    txt = torch.from_numpy(l2_normalize_np_array(txt_feats))
    src_txt = txt.unsqueeze(0).to(device)
    src_txt_mask = torch.ones(1, src_txt.shape[1]).to(device)

    total_segs = vid_feats.shape[0]
    all_windows, all_confidences = [], []

    starts = list(range(0, total_segs, stride))
    # Always include a final window ending at the last segment
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

FFMPEG = next((p for p in [
    '/home/user/miniconda3/envs/ml_fresh_start/bin/ffmpeg',
    '/home/user/miniconda3/pkgs/ffmpeg-5.0.1-h964e5f1_2/bin/ffmpeg',
    'ffmpeg',
] if os.path.exists(p) or p == 'ffmpeg'), 'ffmpeg')


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
    """Concatenate clips in chronological order into one video."""
    import tempfile
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


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_video', required=True,
                        help='Full cricket match video')
    parser.add_argument('--query_videos', nargs='+', default=[],
                        help='Query videos as label:path1,path2,... or label:/folder/  e.g. six:a.mp4,b.mp4 or six:data/sportsmr/query_videos_by_class/six/')
    parser.add_argument('--query_dir', default=None,
                        help='Root folder of class-organised query videos (e.g. data/sportsmr/query_videos_by_class). '
                             'Use with --classes to pick specific classes, or leave --classes empty to use all.')
    parser.add_argument('--classes', nargs='+', default=[],
                        help='Class names to use from --query_dir (e.g. six four). If omitted, all sub-folders are used.')
    parser.add_argument('--vid_feat_dir', default=None,
                        help='Dir of pre-extracted target video .npz features (e.g. data/sportsmr/vid_clip)')
    parser.add_argument('--query_feat_dir', default=None,
                        help='Dir of pre-extracted query video .npz features (e.g. data/sportsmr/vid_clip_query)')
    parser.add_argument('--checkpoint', default='checkpoints/model_best.ckpt')
    parser.add_argument('--output_dir', default='highlights')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Max highlight clips per query label')
    parser.add_argument('--min_duration', type=float, default=2.0)
    parser.add_argument('--max_duration', type=float, default=60.0)
    parser.add_argument('--nms_iou', type=float, default=0.5)
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence score to keep a predicted window (e.g. 0.1)')
    parser.add_argument('--num_queries', type=int, default=5,
                        help='Number of query videos to randomly sample per class for multi-query inference')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Build queries list from --query_dir / --classes
    queries = []  # [(label, [path1, path2, ...]), ...]

    if args.query_dir:
        query_dir = args.query_dir.rstrip('/')
        class_names = args.classes if args.classes else sorted(os.listdir(query_dir))
        for label in class_names:
            class_folder = os.path.join(query_dir, label)
            if not os.path.isdir(class_folder):
                logger.warning(f"Class folder not found, skipping: {class_folder}")
                continue
            paths = sorted([
                os.path.join(class_folder, f)
                for f in os.listdir(class_folder)
                if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))
            ])
            if not paths:
                continue
            queries.append((label, paths))

    # Also parse any --query_videos entries (label:path1,path2,... or label:/folder/)
    for item in args.query_videos:
        if ':' not in item:
            raise ValueError(f"Query videos must be label:path (or label:/folder/), got: {item}")
        label, paths_str = item.split(':', 1)
        paths_str = paths_str.strip()
        # Support label:/folder/ syntax
        if os.path.isdir(paths_str):
            paths = sorted([
                os.path.join(paths_str, f)
                for f in os.listdir(paths_str)
                if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))
            ])
        else:
            paths = [p.strip() for p in paths_str.split(',') if p.strip()]
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Query video not found: {path}")
        queries.append((label, paths))

    if not queries:
        raise ValueError("No queries specified. Use --query_dir or --query_videos.")
    logger.info(f"Queries: {[(l, len(ps)) for l, ps in queries]}")

    def load_npz_or_extract(video_path, feat_dir, clip_model=None, preprocess=None):
        """Load pre-extracted .npz if available, otherwise extract with CLIP."""
        vid_id = os.path.splitext(os.path.basename(video_path))[0]
        npz_path = os.path.join(feat_dir, f"{vid_id}.npz") if feat_dir else None
        if npz_path and os.path.exists(npz_path):
            return np.load(npz_path)["features"].astype(np.float32)
        # fallback: extract with CLIP
        if clip_model is None:
            raise FileNotFoundError(f"No pre-extracted features found for {video_path} in {feat_dir}")
        feats, _ = extract_video_features(video_path, clip_model, preprocess, device)
        return feats

    # Load match video features from pre-extracted dir or cache
    match_vid_id = os.path.splitext(os.path.basename(args.match_video))[0]
    match_npz = os.path.join(args.vid_feat_dir, f"{match_vid_id}.npz") if args.vid_feat_dir else None
    cache_path = os.path.splitext(args.match_video)[0] + "_clip_feats.npz"

    if match_npz and os.path.exists(match_npz):
        logger.info(f"Loading match features from: {match_npz}")
        vid_feats = np.load(match_npz)["features"].astype(np.float32)
        cap = cv2.VideoCapture(args.match_video)
        match_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        logger.info(f"Match: {format_time(match_duration)}, {vid_feats.shape[0]} segments")
    elif os.path.exists(cache_path):
        logger.info(f"Loading cached match features: {cache_path}")
        cached = np.load(cache_path)
        vid_feats, match_duration = cached["features"], float(cached["duration"])
        logger.info(f"Match: {format_time(match_duration)}, {vid_feats.shape[0]} segments (from cache)")
    else:
        logger.info("Loading CLIP model for feature extraction...")
        clip_model, preprocess = clip.load(CLIP_MODEL_VERSION, device=device, jit=False)
        clip_model.eval()
        logger.info(f"Extracting match video features: {args.match_video}")
        vid_feats, match_duration = extract_video_features(args.match_video, clip_model, preprocess, device)
        np.savez_compressed(cache_path, features=vid_feats, duration=match_duration)
        logger.info(f"Match: {format_time(match_duration)}, {vid_feats.shape[0]} segments (saved to {cache_path})")

    # Load MATR
    logger.info("Loading MATR model...")
    model, opt = load_matr_model(args.checkpoint)
    model = model.to(device)

    # Step 1: collect all predicted windows from every query
    all_windows = []   # [(start, end, label, conf), ...]
    results_summary = {}

    for label, query_paths in queries:
        logger.info(f"\n=== Query: '{label}' ({len(query_paths)} video(s)) ===")

        # Load pre-extracted query features
        all_q_feats = []
        missing = []
        for query_path in query_paths:
            vid_id = os.path.splitext(os.path.basename(query_path))[0]
            npz_path = os.path.join(args.query_feat_dir, f"{vid_id}.npz") if args.query_feat_dir else None
            if npz_path and os.path.exists(npz_path):
                all_q_feats.append(np.load(npz_path)["features"].astype(np.float32))
            else:
                missing.append(query_path)

        if missing:
            logger.warning(f"  {len(missing)} videos missing pre-extracted features, skipping them.")
        if not all_q_feats:
            logger.warning(f"  No features found for '{label}', skipping.")
            continue

        # Randomly sample up to num_queries from available features
        import random as _random
        sampled = _random.sample(all_q_feats, min(args.num_queries, len(all_q_feats)))
        logger.info(f"  Running {len(sampled)} query inference passes (sampled from {len(all_q_feats)} available)")

        windows_list, confidences_list = [], []
        for qi, q_feats in enumerate(sampled):
            ws, cs = run_inference(model, vid_feats, q_feats, device)
            windows_list.append(ws)
            confidences_list.append(cs)
            logger.info(f"    Query {qi+1}/{len(sampled)}: {len(ws)} raw windows")

        # Aggregate: NMS + confidence boost for windows confirmed by multiple queries
        windows, confidences = aggregate_multi_query(windows_list, confidences_list,
                                                     iou_threshold=args.nms_iou)

        saved = []
        for window, conf in zip(windows, confidences):
            if len(saved) >= args.top_k:
                break
            if conf < args.confidence_threshold:
                logger.info(f"  Skipping window (conf={conf:.3f} < threshold {args.confidence_threshold})")
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
        logger.info(f"  Kept {len(saved)} windows for '{label}' "
                    f"(threshold={args.confidence_threshold})")

    # Step 2: merge overlapping windows across all queries
    def merge_intervals(intervals):
        """intervals: list of (start, end). Returns merged list sorted by start."""
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:          # overlapping or touching
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged]

    raw_intervals = [(s, e) for s, e, _, _ in all_windows]
    merged = merge_intervals(raw_intervals)
    logger.info(f"\nMerged {len(all_windows)} windows → {len(merged)} unique intervals")

    # Step 3: cut merged clips and stitch
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
        logger.warning("No clips found. Try lowering --min_duration or increasing --top_k.")


if __name__ == '__main__':
    main()
