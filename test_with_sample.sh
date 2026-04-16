#!/bin/bash
# Automated test script with a sample cricket video
# This script will download a short sample and test the complete pipeline

set -e  # Exit on error

echo "======================================================================="
echo "Cricket Highlights Generator - Automated Test"
echo "======================================================================="
echo ""

# Check if we're in the right directory
if [ ! -d "cricket_highlights" ]; then
    echo "❌ Error: cricket_highlights directory not found"
    echo "   Please run this script from the MATR project root"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

echo "Step 1: Verifying setup..."
echo "-----------------------------------------------------------------------"
python3 verify_setup.py
echo ""

# Ask user to continue
read -p "Continue with the test? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test cancelled."
    exit 0
fi

# Check for model checkpoint
if [ ! -f "ckpt/model_best.ckpt" ]; then
    echo ""
    echo "❌ Model checkpoint not found!"
    echo "   Please download it from:"
    echo "   https://drive.google.com/file/d/1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt/view"
    echo "   Save to: ./ckpt/model_best.ckpt"
    echo ""
    exit 1
fi

# Check for query features
if [ ! -f "cricket_highlights/queries/cricket_queries_simple.npz" ]; then
    echo ""
    echo "Step 2: Generating query features..."
    echo "-----------------------------------------------------------------------"
    cd cricket_highlights/queries
    python3 generate_query_features.py
    cd ../..
    echo "✓ Query features generated"
    echo ""
fi

# Create test directory
mkdir -p test_output
mkdir -p test_videos

# Check if user has a test video
echo ""
echo "Step 3: Getting test video..."
echo "-----------------------------------------------------------------------"
echo ""
echo "Do you have a cricket video to test with?"
echo "  1) Yes, I'll provide the path"
echo "  2) No, download a sample from YouTube"
echo "  3) Skip video test (only verify setup)"
echo ""
read -p "Choose option (1-3): " choice

TEST_VIDEO=""

case $choice in
    1)
        read -p "Enter path to your cricket video: " TEST_VIDEO
        if [ ! -f "$TEST_VIDEO" ]; then
            echo "❌ Video file not found: $TEST_VIDEO"
            exit 1
        fi
        ;;
    2)
        echo ""
        read -p "Enter YouTube URL (or press Enter for a default): " YT_URL

        if [ -z "$YT_URL" ]; then
            echo "⚠️  No URL provided. Please provide a YouTube URL with cricket content."
            echo "   Example: https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
            exit 0
        fi

        # Check for yt-dlp
        if ! command -v yt-dlp &> /dev/null; then
            echo "Installing yt-dlp..."
            pip3 install yt-dlp
        fi

        echo "Downloading video (this may take a few minutes)..."
        yt-dlp -f "best[height<=720]" -o "test_videos/sample_cricket.mp4" "$YT_URL"
        TEST_VIDEO="test_videos/sample_cricket.mp4"

        if [ ! -f "$TEST_VIDEO" ]; then
            echo "❌ Failed to download video"
            exit 1
        fi
        echo "✓ Video downloaded: $TEST_VIDEO"
        ;;
    3)
        echo "✓ Setup verification complete"
        echo "  Run this script again when you have a test video"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Step 4: Running cricket highlights pipeline..."
echo "-----------------------------------------------------------------------"
echo "Video: $TEST_VIDEO"
echo "Output: test_output/"
echo ""

# Run the pipeline
cd cricket_highlights/examples

python3 quick_start.py \
    --video "../../$TEST_VIDEO" \
    --checkpoint ../../ckpt/model_best.ckpt \
    --output_dir ../../test_output

cd ../..

echo ""
echo "======================================================================="
echo "✅ Test Complete!"
echo "======================================================================="
echo ""
echo "Output files:"
echo "  📁 Features: test_output/features/"
echo "  📄 Results:  test_output/highlights.json"
echo "  🎬 Video:    test_output/*_highlights.mp4"
echo "  📝 Summary:  test_output/summary.txt"
echo ""
echo "Check the results:"
echo "  cat test_output/highlights.json | head -50"
echo "  cat test_output/summary.txt"
echo ""

# Try to get video info
if [ -f test_output/*_highlights.mp4 ]; then
    HIGHLIGHT_VIDEO=$(ls test_output/*_highlights.mp4 | head -n 1)
    echo "Play the highlight reel:"
    echo "  vlc '$HIGHLIGHT_VIDEO'"
    echo "  # OR"
    echo "  mpv '$HIGHLIGHT_VIDEO'"
    echo ""
fi

echo "======================================================================="
