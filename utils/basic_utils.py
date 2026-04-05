"""
Basic utility functions for MATR.
Minimal implementation for cricket highlights functionality.
"""

import os
import shutil
import numpy as np
import torch


def mkdirp(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def remkdirp(path):
    """Remove directory if it exists and recreate it."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def l2_normalize_np_array(np_array, eps=1e-5):
    """L2 normalize a numpy array."""
    norm = np.linalg.norm(np_array, axis=-1, keepdims=True)
    return np_array / (norm + eps)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    """Save data to JSON file."""
    import json
    with open(filename, 'w') as f:
        if save_pretty:
            json.dump(data, f, indent=4, sort_keys=sort_keys)
        else:
            json.dump(data, f)


def load_json(filename):
    """Load data from JSON file."""
    import json
    with open(filename, 'r') as f:
        return json.load(f)


def save_jsonl(data, filename):
    """Save data to JSONL file (one JSON object per line)."""
    import json
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_jsonl(filename):
    """Load data from JSONL file."""
    import json
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def flat_list_of_lists(list_of_lists):
    """Flatten a list of lists."""
    return [item for sublist in list_of_lists for item in sublist]


def get_lr(optimizer):
    """Get learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_paths=None,
                 exclude_extensions=None, exclude_dirs=None, exclude_dirs_substring=None):
    """Create a zip file from a directory."""
    import zipfile
    from pathlib import Path

    exclude_paths = exclude_paths or []
    exclude_extensions = exclude_extensions or []
    exclude_dirs = exclude_dirs or []
    exclude_dirs_substring = exclude_dirs_substring or ""

    with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs
                       and (not exclude_dirs_substring or exclude_dirs_substring not in d)]

            for file in files:
                file_path = os.path.join(root, file)

                # Skip excluded paths and extensions
                if any(excl in file_path for excl in exclude_paths):
                    continue
                if any(file.endswith(ext) for ext in exclude_extensions):
                    continue

                # Add file to zip
                arcname = os.path.join(
                    enclosing_dir,
                    os.path.relpath(file_path, src_dir)
                )
                zipf.write(file_path, arcname)

    return save_path


def dict_to_markdown(d, max_str_len=120):
    """Convert a dictionary to markdown format."""
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"**{key}**:")
            for k, v in value.items():
                v_str = str(v)
                if max_str_len is not None and len(v_str) > max_str_len:
                    v_str = v_str[:max_str_len] + "..."
                lines.append(f"  - {k}: {v_str}")
        else:
            v_str = str(value)
            if max_str_len is not None and len(v_str) > max_str_len:
                v_str = v_str[:max_str_len] + "..."
            lines.append(f"- **{key}**: {v_str}")
    return "\n".join(lines)


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = float('-inf')
        self.min = float('inf')

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        self.max = max(self.max, val)
        self.min = min(self.min, val)
