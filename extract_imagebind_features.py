"""
Extract ImageBind (vision + audio) fused features from raw video clips.

For each 2-second segment:
  - Extracts center frame (vision)
  - Extracts audio waveform (audio)
  - Passes both to ImageBind, averages embeddings → 1024-dim
  - Falls back to vision-only if video has no audio track

Usage:
  python extract_imagebind_features.py --video_dir data/sportsmr/raw_target_videos --out_dir data/sportsmr/vid_imagebind
  python extract_imagebind_features.py --video_dir data/sportsmr/raw_query_videos --out_dir data/sportsmr/vid_imagebind_query
"""
import os
import argparse
import subprocess
import tempfile
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

CLIP_LENGTH = 2       # seconds per segment
AUDIO_SAMPLE_RATE = 16000  # ImageBind expects 16kHz
FFMPEG = '/home/user/miniconda3/envs/ml_fresh_start/bin/ffmpeg'
FFPROBE = '/home/user/miniconda3/envs/ml_fresh_start/bin/ffprobe'
PYTHON = '/home/user/miniconda3/pkgs/python-3.10.18-h1a3bd86_0/bin/python3.10'


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
    """Convert raw waveform to mel spectrogram matching ImageBind's expected format."""
    import torchaudio
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr,
        use_energy=False, window_type='hanning',
        num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10)  # (T, num_mel_bins)
    # Pad or trim to target_length
    T = fbank.shape[0]
    if T < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - T))
    else:
        fbank = fbank[:target_length]
    fbank = (fbank - mean) / std
    return fbank.unsqueeze(0)  # (1, target_length, num_mel_bins)


def extract_audio_segments(video_path, n_clips, clip_length=CLIP_LENGTH, sr=AUDIO_SAMPLE_RATE):
    """Extract per-segment audio mel spectrograms. Returns list of (1, 128, 204) tensors or None."""
    # Check if video has audio
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
        waveform, file_sr = torchaudio.load(tmp_path)  # (1, total_samples)
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
            mel = waveform_to_melspec(seg, sr=sr)  # (1, T, num_mel_bins)
            mel = mel.permute(0, 2, 1)             # (1, num_mel_bins, T) = (1, 128, 204)
            segments.append(mel)

    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return segments  # list of (1, 128, 204)


def extract_features_for_video(video_path, model, transform, device, clip_length=CLIP_LENGTH,
                               batch_size=64, show_progress=False):
    from imagebind.models.imagebind_model import ModalityType

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
    fname = os.path.basename(video_path)

    # Extract frames with progress
    frames = []
    frame_iter = tqdm(range(n_clips), desc=f"  frames [{fname}]", leave=False) if show_progress else range(n_clips)
    for clip_idx in frame_iter:
        center_time = (clip_idx + 0.5) * clip_length
        frame_idx = min(int(center_time * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(Image.fromarray(frame_rgb)))
    cap.release()

    # Extract audio segments
    if show_progress:
        tqdm.write(f"  [{fname}] Extracting audio...")
    audio_segments = extract_audio_segments(video_path, n_clips, clip_length)

    # Vision embeddings in batches with progress
    vis_feats = []
    n_batches = (n_clips + batch_size - 1) // batch_size
    batch_iter = tqdm(range(0, n_clips, batch_size), total=n_batches,
                      desc=f"  vision [{fname}]", leave=False) if show_progress else range(0, n_clips, batch_size)
    for i in batch_iter:
        batch = torch.stack(frames[i:i+batch_size]).to(device)
        with torch.no_grad():
            emb = model({ModalityType.VISION: batch})[ModalityType.VISION]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vis_feats.append(emb.cpu().float())
    vis_emb = torch.cat(vis_feats, dim=0)

    if audio_segments is not None:
        aud_feats = []
        aud_iter = tqdm(range(0, n_clips, batch_size), total=n_batches,
                        desc=f"  audio  [{fname}]", leave=False) if show_progress else range(0, n_clips, batch_size)
        for i in aud_iter:
            batch = torch.stack(audio_segments[i:i+batch_size]).to(device)
            with torch.no_grad():
                emb = model({ModalityType.AUDIO: batch})[ModalityType.AUDIO]
                emb = emb / emb.norm(dim=-1, keepdim=True)
            aud_feats.append(emb.cpu().float())
        aud_emb = torch.cat(aud_feats, dim=0)
        fused = (vis_emb + aud_emb) / 2.0
        fused = fused / fused.norm(dim=-1, keepdim=True)
    else:
        if show_progress:
            tqdm.write(f"  [{fname}] No audio track, using vision only.")
        fused = vis_emb

    return fused.cpu().float().numpy()  # (n_clips, 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--restart', action='store_true',
                        help='Re-extract all videos even if .npz already exists')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from imagebind.models import imagebind_model
    print(f"Loading ImageBind model on {args.device}...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(args.device)

    transform = get_vision_transform()

    video_files = sorted([f for f in os.listdir(args.video_dir)
                          if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])
    print(f"Found {len(video_files)} videos in {args.video_dir}")

    single_video = len(video_files) == 1
    skipped = 0
    for fname in tqdm(video_files, desc="Videos"):
        vid_id = os.path.splitext(fname)[0]
        out_path = os.path.join(args.out_dir, f"{vid_id}.npz")
        if os.path.exists(out_path) and not args.restart:
            continue

        video_path = os.path.join(args.video_dir, fname)
        features = extract_features_for_video(video_path, model, transform, args.device,
                                              show_progress=single_video)
        if features is None:
            skipped += 1
            continue

        np.savez_compressed(out_path, features=features)
        tqdm.write(f"Saved: {out_path}  shape={features.shape}")

    print(f"Done. Skipped {skipped} videos.")


if __name__ == '__main__':
    main()
