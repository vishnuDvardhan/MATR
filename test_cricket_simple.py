#!/usr/bin/env python
"""
Simple standalone cricket highlights test script.
Bypasses complex argument parsing issues.
"""

import os
import sys

# Set up paths
sys.path.insert(0, '/home/user/Desktop/MATR')

print("="*70)
print("🏏 Cricket Highlights - Simple Test")
print("="*70)
print()

# Configuration
VIDEO_FEATURES = "test_output/features/cricket_sample_features.npz"
MODEL_CHECKPOINT = "ckpt/model_best.ckpt"
OUTPUT_JSON = "test_output/highlights_simple.json"
VIDEO_FILE = "test_videos/cricket_sample.mp4"

# Check files exist
if not os.path.exists(VIDEO_FEATURES):
    print(f"❌ Error: Video features not found: {VIDEO_FEATURES}")
    print("   Run feature extraction first!")
    sys.exit(1)

if not os.path.exists(MODEL_CHECKPOINT):
    print(f"❌ Error: Model checkpoint not found: {MODEL_CHECKPOINT}")
    sys.exit(1)

print(f"✓ Video features: {VIDEO_FEATURES}")
print(f"✓ Model checkpoint: {MODEL_CHECKPOINT}")
print()

# Simple detection using pre-computed query features
print("[Step 1/2] Running simple highlight detection...")
print()

import torch
import numpy as np
import json

# Load video features
print("Loading video features...")
vid_data = np.load(VIDEO_FEATURES)
vid_features = vid_data['features']
print(f"✓ Loaded {len(vid_features)} video segments")

# Load query features
print("Loading query features...")
query_path = "cricket_highlights/queries/cricket_queries_simple.npz"
query_data = np.load(query_path, allow_pickle=True)
query_features = query_data['last_hidden_state']
query_names = query_data['query_names']
print(f"✓ Loaded {len(query_names)} queries: {list(query_names)}")

# Simple similarity-based detection (without full MATR model for now)
print()
print("Computing similarities...")

from utils.basic_utils import l2_normalize_np_array

# Normalize features
vid_norm = l2_normalize_np_array(vid_features)
query_norm = l2_normalize_np_array(query_features)

# Compute similarities
similarities = {}
clip_duration = 2.0  # seconds per clip

for idx, query_name in enumerate(query_names):
    query_vec = query_norm[idx]
    sims = np.dot(vid_norm, query_vec)

    # Find peaks (potential highlights)
    threshold = 0.2  # Adjust this threshold
    peaks = np.where(sims > threshold)[0]

    highlights_for_query = []
    for peak in peaks:
        start_time = peak * clip_duration
        end_time = (peak + 1) * clip_duration
        confidence = float(sims[peak])

        highlights_for_query.append({
            'start_time': start_time,
            'end_time': end_time,
            'confidence': confidence,
            'query': query_name
        })

    # Sort by confidence
    highlights_for_query = sorted(highlights_for_query, key=lambda x: x['confidence'], reverse=True)

    # Take top 10
    highlights_for_query = highlights_for_query[:10]

    similarities[query_name] = highlights_for_query
    print(f"  {query_name}: Found {len(highlights_for_query)} potential moments")

# Combine all highlights
all_highlights = []
for query_name, highlights in similarities.items():
    all_highlights.extend(highlights)

# Sort by confidence
all_highlights = sorted(all_highlights, key=lambda x: x['confidence'], reverse=True)

# Remove overlapping highlights (keep higher confidence)
def remove_overlaps(highlights, time_threshold=5.0):
    """Remove overlapping highlights."""
    if not highlights:
        return []

    result = [highlights[0]]
    for current in highlights[1:]:
        overlap = False
        for existing in result:
            if abs(current['start_time'] - existing['start_time']) < time_threshold:
                overlap = True
                break
        if not overlap:
            result.append(current)
    return result

final_highlights = remove_overlaps(all_highlights, time_threshold=5.0)

print()
print(f"✓ Found {len(final_highlights)} highlights (after removing overlaps)")
print()

# Save results
output_data = {
    'video': VIDEO_FILE,
    'method': 'simple_similarity',
    'total_highlights': len(final_highlights),
    'highlights': final_highlights,
    'by_query': similarities
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Results saved to: {OUTPUT_JSON}")
print()

# Print top highlights
print("Top 10 Highlights:")
print("-" * 70)
for i, h in enumerate(final_highlights[:10], 1):
    print(f"{i:2d}. [{h['query'].upper():10s}] "
          f"{h['start_time']:6.1f}s - {h['end_time']:6.1f}s  "
          f"(confidence: {h['confidence']:.3f})")

print()
print("=" * 70)
print("✅ Simple detection complete!")
print("=" * 70)
print()
print("Note: This is a simplified detection using similarity matching.")
print("For full MATR model inference, we need to resolve the config issues.")
print()
print("You can now compile highlights with:")
print(f"  python cricket_highlights/inference/highlight_compiler.py \\")
print(f"    --video {VIDEO_FILE} \\")
print(f"    --highlights_json {OUTPUT_JSON} \\")
print(f"    --output cricket_highlights_simple.mp4 \\")
print(f"    --method opencv")
print()
