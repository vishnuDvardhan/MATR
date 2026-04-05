"""
Extract CLIP (ViT-B/32) visual features from raw video clips for MATR training.

For each video:
  - Sample frames every CLIP_LENGTH seconds (one frame per 2-sec clip segment)
  - Extract CLIP visual features for each frame
  - Save as {vid}.npz with key "features", shape (n_clips, 512)

Usage:
  python extract_clip_features.py --video_dir data/sportsmr/raw_query_videos --out_dir data/sportsmr/vid_clip_query
  python extract_clip_features.py --video_dir data/sportsmr/raw_target_videos --out_dir data/sportsmr/vid_clip
"""
import os
import argparse
import numpy as np
import torch
import clip
from PIL import Image
import cv2
from tqdm import tqdm


CLIP_LENGTH = 2  # seconds per segment


def extract_features_for_video(video_path, model, preprocess, device, clip_length=CLIP_LENGTH):
    """Extract per-segment CLIP features from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration <= 0 or fps <= 0:
        cap.release()
        return None

    n_clips = max(1, int(duration / clip_length))
    frames = []

    for clip_idx in range(n_clips):
        # Sample one frame from the middle of each clip segment
        center_time = (clip_idx + 0.5) * clip_length
        frame_idx = min(int(center_time * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # Use a blank frame as fallback
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(preprocess(pil_img))

    cap.release()

    # Batch inference
    batch = torch.stack(frames).to(device)
    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

    return features.cpu().float().numpy()  # (n_clips, 512)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading CLIP model {args.clip_model} on {args.device}...")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()

    video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos in {args.video_dir}")

    skipped = 0
    for fname in tqdm(video_files):
        vid_id = fname.replace('.mp4', '')
        out_path = os.path.join(args.out_dir, f'{vid_id}.npz')

        if os.path.exists(out_path):
            continue

        video_path = os.path.join(args.video_dir, fname)
        features = extract_features_for_video(video_path, model, preprocess, args.device)

        if features is None:
            skipped += 1
            print(f"  Warning: could not process {fname}")
            continue

        np.savez_compressed(out_path, features=features)

    print(f"\nDone. Skipped {skipped} videos. Features saved to {args.out_dir}")


if __name__ == '__main__':
    main()
