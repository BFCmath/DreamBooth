#!/bin/bash
# ControlNet Inference Script for Kaggle P100
# Run this script to generate pose-controlled images

set -e  # Exit on error

echo "========================================="
echo "ControlNet Inference with OpenPose"
echo "========================================="

# Install dependencies (skip torch to avoid conflicts with Kaggle's preinstalled version)
echo "📦 Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub
pip install -q controlnet-aux opencv-python Pillow

# Try to install xformers for memory efficiency (optional)
pip install -q xformers 2>/dev/null || echo "⚠️  xformers not installed (optional)"

# ===========================================
# CRITICAL: Reduce CUDA memory fragmentation
# ===========================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories
mkdir -p input_images
mkdir -p output_images

# =========================================
# CONFIGURATION - Modify these variables
# =========================================

# Input image path (image to extract pose from, or pose image if USE_POSE_DIRECTLY=true)
INPUT_IMAGE="${INPUT_IMAGE:-./input_images/pose_reference.png}"

# Text prompt for generation
PROMPT="${PROMPT:-a professional photo of a person standing, high quality, detailed}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./output_images}"

# Model path (can be DreamBooth-finetuned model path)
MODEL_PATH="${MODEL_PATH:-runwayml/stable-diffusion-v1-5}"

# ControlNet model
CONTROLNET_MODEL="${CONTROLNET_MODEL:-lllyasviel/control_v11p_sd15_openpose}"

# Generation parameters
NUM_IMAGES="${NUM_IMAGES:-4}"
NUM_STEPS="${NUM_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-1.0}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEED="${SEED:-42}"

# Set to "true" to use input image as pose directly (skip pose extraction)
# TIP: Using pose directly saves ~2GB VRAM (no OpenPose detector needed)
USE_POSE_DIRECTLY="${USE_POSE_DIRECTLY:-false}"

# LOW VRAM MODE: Enable aggressive memory optimizations for P100 (16GB)
# Uses sequential CPU offloading - slower but uses much less GPU memory
LOW_VRAM_MODE="${LOW_VRAM_MODE:-true}"

# Negative prompt
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-lowres, bad anatomy, worst quality, low quality, deformed, ugly}"

# =========================================

echo ""
echo "========================================="
echo "Configuration"
echo "========================================="
echo "Input image: $INPUT_IMAGE"
echo "Prompt: $PROMPT"
echo "Model: $MODEL_PATH"
echo "ControlNet: $CONTROLNET_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Num images: $NUM_IMAGES"
echo "Steps: $NUM_STEPS"
echo "Guidance: $GUIDANCE_SCALE"
echo "ControlNet scale: $CONTROLNET_SCALE"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Seed: $SEED"
echo "Use pose directly: $USE_POSE_DIRECTLY"
echo "Low VRAM mode: $LOW_VRAM_MODE"
echo "========================================="
echo ""
echo "Memory optimizations enabled:"
echo "  ✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  ✅ Sequential CPU offloading (low VRAM mode)"
echo "  ✅ FP16 precision"
echo "  ✅ VAE slicing"
if [ "$USE_POSE_DIRECTLY" = "true" ]; then
    echo "  ✅ OpenPose detector skipped (saves ~2GB)"
fi
echo ""
echo "If still OOM, try:"
echo "  HEIGHT=384 WIDTH=384 bash run_inference.sh"
echo "  NUM_IMAGES=1 bash run_inference.sh"
echo "  USE_POSE_DIRECTLY=true bash run_inference.sh"
echo "========================================="
echo ""

# Check if input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "========================================="
    echo "⚠️  WARNING: Input image not found!"
    echo "========================================="
    echo "Please add your input image to: $INPUT_IMAGE"
    echo ""
    echo "Options:"
    echo "  1. Upload an image to input_images/ directory"
    echo "  2. Set INPUT_IMAGE environment variable:"
    echo "     export INPUT_IMAGE=/path/to/your/image.jpg"
    echo ""
    echo "The input image can be:"
    echo "  - A photo of a person (pose will be extracted)"
    echo "  - A pre-made pose/skeleton image (set USE_POSE_DIRECTLY=true)"
    echo ""
    
    # Create a sample download command
    echo "Example: Download a sample pose image"
    echo "  wget -O input_images/pose_reference.png \\"
    echo "    https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png"
    echo ""
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU: %s | Total: %.1f GB | Free: %.1f GB\n", $1, $2/1024, $3/1024}'
    echo ""
fi

# Build command
CMD="python infer_controlnet.py \
    --prompt=\"$PROMPT\" \
    --input_image=\"$INPUT_IMAGE\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --model_path=\"$MODEL_PATH\" \
    --controlnet_model=\"$CONTROLNET_MODEL\" \
    --num_images=$NUM_IMAGES \
    --num_inference_steps=$NUM_STEPS \
    --guidance_scale=$GUIDANCE_SCALE \
    --controlnet_scale=$CONTROLNET_SCALE \
    --height=$HEIGHT \
    --width=$WIDTH \
    --seed=$SEED \
    --negative_prompt=\"$NEGATIVE_PROMPT\""

# Add pose direct flag if enabled
if [ "$USE_POSE_DIRECTLY" = "true" ]; then
    CMD="$CMD --use_pose_directly"
fi

# Add low VRAM flag if enabled
if [ "$LOW_VRAM_MODE" = "true" ]; then
    CMD="$CMD --low_vram"
fi

# Run inference
echo "🚀 Starting ControlNet inference..."
echo ""
eval $CMD

echo ""
echo "========================================="
echo "✅ Inference Complete!"
echo "========================================="
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  - Images: $OUTPUT_DIR/*.png"
echo "  - Gallery: $OUTPUT_DIR/gallery.html"
echo ""
echo "========================================="
