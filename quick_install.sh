#!/bin/bash
# Quick installation script for Cricket Highlights Generator
# This will install all necessary dependencies

echo "======================================================================="
echo "Cricket Highlights Generator - Quick Install"
echo "======================================================================="
echo ""

# Detect if we should use pip or pip3
if command -v pip3 &> /dev/null; then
    PIP="pip3"
elif command -v pip &> /dev/null; then
    PIP="pip"
else
    echo "❌ Error: pip not found"
    exit 1
fi

echo "Using: $PIP"
echo ""

# Ask user about GPU
echo "Do you have an NVIDIA GPU with CUDA support?"
read -p "Enter y for GPU (CUDA) or n for CPU only: " -n 1 -r
echo ""
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    echo "-----------------------------------------------------------------------"
    $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    echo "-----------------------------------------------------------------------"
    $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Installing other dependencies..."
echo "-----------------------------------------------------------------------"

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    $PIP install -r requirements.txt
fi

# Install cricket-specific dependencies
$PIP install opencv-python yt-dlp tqdm

echo ""
echo "Installation complete!"
echo ""

# Verify installation
echo "Verifying installation..."
echo "-----------------------------------------------------------------------"
python3 verify_setup.py

echo ""
echo "======================================================================="
echo "Next Steps:"
echo "======================================================================="
echo ""
echo "1. Download the MATR model checkpoint:"
echo "   📥 https://drive.google.com/file/d/1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt/view"
echo "   Save to: ./ckpt/model_best.ckpt"
echo ""
echo "2. Generate cricket query features:"
echo "   cd cricket_highlights/queries"
echo "   python3 generate_query_features.py"
echo ""
echo "3. Test with a cricket video:"
echo "   bash test_with_sample.sh"
echo ""
echo "======================================================================="
