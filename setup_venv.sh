#!/bin/bash
# Complete setup using Python virtual environment (no conda needed)

set -e  # Exit on error

echo "======================================================================="
echo "Setting up Virtual Environment for Cricket Highlights Generator"
echo "======================================================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python: $PYTHON_VERSION"
echo ""

# Ask about GPU
echo "Do you have an NVIDIA GPU with CUDA support?"
read -p "Enter y for GPU (CUDA) or n for CPU only [n]: " -r GPU_CHOICE
GPU_CHOICE=${GPU_CHOICE:-n}
echo ""

# Create virtual environment
echo "Step 1: Creating virtual environment 'matr_env'..."
echo "-----------------------------------------------------------------------"
if [ -d "matr_env" ]; then
    echo "⚠️  Virtual environment 'matr_env' already exists."
    read -p "Do you want to remove and recreate it? (y/n) [n]: " -r RECREATE
    RECREATE=${RECREATE:-n}
    if [[ $RECREATE =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf matr_env
        python3 -m venv matr_env
    else
        echo "Using existing environment..."
    fi
else
    python3 -m venv matr_env
fi

# Activate environment
source matr_env/bin/activate

echo ""
echo "Step 2: Installing PyTorch..."
echo "-----------------------------------------------------------------------"

# Upgrade pip
pip install --upgrade pip

if [[ $GPU_CHOICE =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Step 3: Installing other dependencies..."
echo "-----------------------------------------------------------------------"

# Install from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install cricket-specific dependencies
pip install opencv-python yt-dlp tqdm

echo ""
echo "Step 4: Verifying installation..."
echo "-----------------------------------------------------------------------"
python verify_setup.py

echo ""
echo "======================================================================="
echo "✅ Virtual Environment Setup Complete!"
echo "======================================================================="
echo ""
echo "Environment location: $(pwd)/matr_env"
echo ""
echo "To activate the environment in the future, run:"
echo "  source matr_env/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Download model checkpoint (see below)"
echo "  2. Generate query features: bash run_complete_test.sh"
echo ""
echo "-----------------------------------------------------------------------"
echo "📥 Download Model Checkpoint:"
echo "-----------------------------------------------------------------------"
echo "Manual: https://drive.google.com/file/d/1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt/view"
echo "Save to: ./ckpt/model_best.ckpt"
echo ""
echo "OR use gdown (install if needed):"
echo "  pip install gdown"
echo "  mkdir -p ckpt"
echo "  gdown 1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt -O ckpt/model_best.ckpt"
echo ""
echo "======================================================================="
