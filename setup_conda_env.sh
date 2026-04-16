#!/bin/bash
# Complete conda environment setup for Cricket Highlights Generator

set -e  # Exit on error

echo "======================================================================="
echo "Setting up Conda Environment for Cricket Highlights Generator"
echo "======================================================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed!"
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo "  1. Download from: https://docs.conda.io/en/latest/miniconda.html"
    echo "  2. Install it"
    echo "  3. Run this script again"
    echo ""
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Ask about GPU
echo "Do you have an NVIDIA GPU with CUDA support?"
read -p "Enter y for GPU (CUDA) or n for CPU only [n]: " -r GPU_CHOICE
GPU_CHOICE=${GPU_CHOICE:-n}
echo ""

# Create conda environment
echo "Step 1: Creating conda environment 'MATR'..."
echo "-----------------------------------------------------------------------"
if conda env list | grep -q "^MATR "; then
    echo "⚠️  Environment 'MATR' already exists."
    read -p "Do you want to remove and recreate it? (y/n) [n]: " -r RECREATE
    RECREATE=${RECREATE:-n}
    if [[ $RECREATE =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n MATR -y
        echo "Creating new environment..."
        conda create -n MATR python=3.10.4 -y
    else
        echo "Using existing environment..."
    fi
else
    conda create -n MATR python=3.10.4 -y
fi

echo ""
echo "Step 2: Activating environment and installing dependencies..."
echo "-----------------------------------------------------------------------"

# Activate environment and install packages
if [[ $GPU_CHOICE =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    conda run -n MATR conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "Installing PyTorch (CPU only)..."
    conda run -n MATR conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

echo ""
echo "Installing other dependencies..."
conda run -n MATR pip install -r requirements.txt
conda run -n MATR pip install opencv-python yt-dlp

echo ""
echo "Step 3: Verifying installation..."
echo "-----------------------------------------------------------------------"
conda run -n MATR python verify_setup.py

echo ""
echo "======================================================================="
echo "✅ Conda Environment Setup Complete!"
echo "======================================================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate MATR"
echo ""
echo "Next steps:"
echo "  1. Download model checkpoint (see below)"
echo "  2. Generate query features: bash generate_queries_conda.sh"
echo "  3. Test the system: bash test_with_sample.sh"
echo ""
echo "-----------------------------------------------------------------------"
echo "📥 Download Model Checkpoint:"
echo "-----------------------------------------------------------------------"
echo "Manual: https://drive.google.com/file/d/1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt/view"
echo "Save to: ./ckpt/model_best.ckpt"
echo ""
echo "OR use gdown:"
echo "  conda activate MATR"
echo "  pip install gdown"
echo "  mkdir -p ckpt"
echo "  gdown 1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt -O ckpt/model_best.ckpt"
echo ""
echo "======================================================================="
