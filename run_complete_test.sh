#!/bin/bash
# Complete end-to-end setup and test script
# This will set up everything and run a test

set -e  # Exit on error

echo "======================================================================="
echo "🏏 Cricket Highlights Generator - Complete Setup & Test"
echo "======================================================================="
echo ""

# Check if we're already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Already in a virtual environment: $VIRTUAL_ENV"
    echo ""
else
    # Check if matr_env exists
    if [ -d "matr_env" ]; then
        echo "✓ Found existing virtual environment"
        echo "  Activating matr_env..."
        source matr_env/bin/activate
    else
        echo "❌ No virtual environment found!"
        echo ""
        echo "Please run one of these first:"
        echo "  bash setup_venv.sh        (Virtual Environment - Faster)"
        echo "  bash setup_conda_env.sh   (Conda - If you have conda)"
        echo ""
        exit 1
    fi
fi

echo ""
echo "Step 1: Checking setup..."
echo "-----------------------------------------------------------------------"
python verify_setup.py | grep -E "(✓|✗|Model checkpoint|Query features)"

echo ""
echo ""

# Check for model checkpoint
if [ ! -f "ckpt/model_best.ckpt" ]; then
    echo "Step 2: Downloading model checkpoint..."
    echo "-----------------------------------------------------------------------"

    # Check if gdown is installed
    if ! python -c "import gdown" 2>/dev/null; then
        echo "Installing gdown..."
        pip install gdown
    fi

    mkdir -p ckpt
    echo "Downloading model checkpoint (~400MB, this may take a few minutes)..."
    gdown 1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt -O ckpt/model_best.ckpt

    if [ -f "ckpt/model_best.ckpt" ]; then
        echo "✓ Model checkpoint downloaded successfully!"
    else
        echo "❌ Failed to download model checkpoint"
        echo "   Please download manually from:"
        echo "   https://drive.google.com/file/d/1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt/view"
        exit 1
    fi
else
    echo "Step 2: Model checkpoint"
    echo "-----------------------------------------------------------------------"
    echo "✓ Model checkpoint already exists"
fi

echo ""
echo ""

# Generate query features
if [ ! -f "cricket_highlights/queries/cricket_queries_simple.npz" ]; then
    echo "Step 3: Generating cricket query features..."
    echo "-----------------------------------------------------------------------"
    cd cricket_highlights/queries
    python generate_query_features.py
    cd ../..
    echo "✓ Query features generated successfully!"
else
    echo "Step 3: Query features"
    echo "-----------------------------------------------------------------------"
    echo "✓ Query features already exist"
fi

echo ""
echo ""
echo "Step 4: Final verification..."
echo "-----------------------------------------------------------------------"
python verify_setup.py

echo ""
echo ""
echo "======================================================================="
echo "✅ Setup Complete! Ready to Test"
echo "======================================================================="
echo ""
echo "Now let's test with a cricket video..."
echo ""
echo "Options:"
echo "  1) I have a cricket video file"
echo "  2) Download a sample from YouTube"
echo "  3) Skip test for now"
echo ""
read -p "Choose option (1-3): " choice

case $choice in
    1)
        read -p "Enter path to your cricket video: " VIDEO_PATH
        if [ ! -f "$VIDEO_PATH" ]; then
            echo "❌ Video file not found: $VIDEO_PATH"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo "⚠️  You'll need a YouTube URL with cricket content."
        echo "   Example sources:"
        echo "   - ICC official highlights"
        echo "   - BCCI videos"
        echo "   - Cricket board official channels"
        echo ""
        read -p "Enter YouTube URL: " YT_URL

        if [ -z "$YT_URL" ]; then
            echo "No URL provided. Exiting."
            exit 0
        fi

        # Install yt-dlp if needed
        if ! command -v yt-dlp &> /dev/null; then
            echo "Installing yt-dlp..."
            pip install yt-dlp
        fi

        mkdir -p test_videos
        echo "Downloading video..."
        yt-dlp -f "best[height<=720]" -o "test_videos/cricket_sample.mp4" "$YT_URL"
        VIDEO_PATH="test_videos/cricket_sample.mp4"

        if [ ! -f "$VIDEO_PATH" ]; then
            echo "❌ Failed to download video"
            exit 1
        fi
        echo "✓ Video downloaded"
        ;;
    3)
        echo ""
        echo "Setup complete! To test later, run:"
        echo "  source matr_env/bin/activate"
        echo "  cd cricket_highlights/examples"
        echo "  python quick_start.py --video ../../your_video.mp4 \\"
        echo "                        --checkpoint ../../ckpt/model_best.ckpt"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "======================================================================="
echo "🚀 Running Cricket Highlights Pipeline"
echo "======================================================================="
echo ""
echo "Video: $VIDEO_PATH"
echo "Output: test_output/"
echo ""

mkdir -p test_output

cd cricket_highlights/examples

# Check if VIDEO_PATH is absolute or relative
if [[ "$VIDEO_PATH" = /* ]]; then
    # Absolute path - use as is
    VIDEO_ARG="$VIDEO_PATH"
else
    # Relative path - add ../../
    VIDEO_ARG="../../$VIDEO_PATH"
fi

python quick_start.py \
    --video "$VIDEO_ARG" \
    --checkpoint ../../ckpt/model_best.ckpt \
    --output_dir ../../test_output

cd ../..

echo ""
echo "======================================================================="
echo "✅ Test Complete!"
echo "======================================================================="
echo ""
echo "📁 Output files:"
echo "  - Features:  test_output/features/"
echo "  - Results:   test_output/highlights.json"
echo "  - Video:     test_output/*_highlights.mp4"
echo "  - Summary:   test_output/summary.txt"
echo ""
echo "View results:"
echo "  cat test_output/highlights.json | python -m json.tool | head -50"
echo "  cat test_output/summary.txt"
echo ""

# Find and display the output video
OUTPUT_VIDEO=$(ls test_output/*_highlights.mp4 2>/dev/null | head -n 1)
if [ -f "$OUTPUT_VIDEO" ]; then
    echo "🎬 Highlight video: $OUTPUT_VIDEO"
    echo ""
    echo "To play:"
    echo "  vlc '$OUTPUT_VIDEO'"
    echo "  # or"
    echo "  mpv '$OUTPUT_VIDEO'"
fi

echo ""
echo "======================================================================="
echo ""
echo "To run again with a different video:"
echo "  source matr_env/bin/activate"
echo "  cd cricket_highlights/examples"
echo "  python quick_start.py --video ../../your_video.mp4 \\"
echo "                        --checkpoint ../../ckpt/model_best.ckpt"
echo ""
echo "For more options, see:"
echo "  cricket_highlights/USAGE_GUIDE.md"
echo ""
echo "======================================================================="
