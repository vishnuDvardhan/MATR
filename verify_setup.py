#!/usr/bin/env python3
"""
Verification script for Cricket Highlights Generator setup.
Run this to check if all dependencies and files are in place.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Warning: Python 3.8+ recommended")
        return False
    return True

def check_dependency(name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError:
        print(f"✗ {name}: NOT INSTALLED")
        return False

def check_file(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"✓ {description}: found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ {description}: NOT FOUND")
        return False

def main():
    print("="*70)
    print("Cricket Highlights Generator - Setup Verification")
    print("="*70)
    print()

    all_good = True

    # Check Python version
    print("[1/5] Python Version")
    print("-"*70)
    all_good &= check_python_version()
    print()

    # Check core dependencies
    print("[2/5] Core Dependencies")
    print("-"*70)
    core_deps = [
        ('PyTorch', 'torch'),
        ('torchvision', 'torchvision'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('tqdm', 'tqdm'),
    ]

    for name, import_name in core_deps:
        all_good &= check_dependency(name, import_name)

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("○ CUDA: not available (will use CPU)")
    except:
        pass
    print()

    # Check additional dependencies
    print("[3/5] Additional Dependencies")
    print("-"*70)
    additional_deps = [
        ('OpenCV', 'cv2'),
        ('yt-dlp', 'yt_dlp'),
    ]

    for name, import_name in additional_deps:
        result = check_dependency(name, import_name)
        if not result and name == 'yt-dlp':
            print("  ℹ️  Optional: only needed for YouTube downloads")
    print()

    # Check project files
    print("[4/5] Project Files")
    print("-"*70)

    files_to_check = [
        ('ckpt/model_best.ckpt', 'Model checkpoint'),
        ('cricket_highlights/queries/cricket_queries.json', 'Cricket queries'),
        ('cricket_highlights/data_preparation/video_processor.py', 'Video processor'),
        ('cricket_highlights/inference/cricket_inference.py', 'Inference module'),
        ('cricket_highlights/inference/highlight_compiler.py', 'Compiler module'),
    ]

    for path, desc in files_to_check:
        result = check_file(path, desc)
        if not result and 'checkpoint' in path:
            print("  📥 Download from: https://drive.google.com/file/d/1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt/view")
            all_good = False
    print()

    # Check query features
    print("[5/5] Generated Files")
    print("-"*70)
    query_features = 'cricket_highlights/queries/cricket_queries_simple.npz'
    if check_file(query_features, 'Query features'):
        print("  ✓ Ready for inference")
    else:
        print("  ⚠️  Need to generate query features")
        print("  Run: cd cricket_highlights/queries && python3 generate_query_features.py")
        all_good = False
    print()

    # Check external tools
    print("[Bonus] External Tools")
    print("-"*70)

    # Check FFmpeg
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True,
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.decode().split('\n')[0]
            print(f"✓ FFmpeg: {version_line}")
        else:
            print("○ FFmpeg: not found (will use OpenCV method)")
    except:
        print("○ FFmpeg: not found (will use OpenCV method)")
    print()

    # Final summary
    print("="*70)
    if all_good:
        print("✅ Setup is COMPLETE! You're ready to generate cricket highlights.")
        print()
        print("Next steps:")
        print("  1. Place a cricket video in the project directory")
        print("  2. Run: cd cricket_highlights/examples")
        print("  3. Run: python3 quick_start.py --video ../../your_video.mp4 \\")
        print("          --checkpoint ../../ckpt/model_best.ckpt")
    else:
        print("⚠️  Setup is INCOMPLETE. Please install missing dependencies.")
        print()
        print("Quick fix:")
        print("  pip install torch torchvision opencv-python yt-dlp")
        print()
        print("For detailed setup instructions, see SETUP_AND_TEST.md")
    print("="*70)
    print()

if __name__ == '__main__':
    main()
