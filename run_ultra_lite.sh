#!/bin/bash
# DreamBooth ULTRA LITE MODE - Maximum stability
# This version uses the most conservative settings possible

set -e  # Exit on error

echo "========================================="
echo "DreamBooth ULTRA LITE MODE"
echo "Maximum Stability Configuration"
echo "========================================="

# First, run debug script to check if models load
echo ""
echo "Running pre-flight checks..."
python debug_segfault.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Pre-flight checks failed!"
    echo "The segfault occurs during model loading."
    echo "This usually means:"
    echo "  1. Insufficient GPU memory"
    echo "  2. Corrupted model cache"
    echo "  3. CUDA/driver issues"
    echo ""
    echo "Try:"
    echo "  - Restart Kaggle kernel"
    echo "  - Clear cache: rm -rf ~/.cache/huggingface/"
    echo "  - Use higher GPU tier (P100)"
    exit 1
fi

echo ""
echo "✅ Pre-flight checks passed!"
echo ""

# Install only what's absolutely necessary
echo "Installing minimal dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub

# Create directories
mkdir -p instance_images
mkdir -p output
mkdir -p logs

# Check images
if [ ! -d "instance_images" ] || [ -z "$(ls -A instance_images)" ]; then
    echo "ERROR: No training images found!"
    echo "Run: python download_example_images.py dog"
    exit 1
fi

NUM_IMAGES=$(ls instance_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES training images"

# ULTRA LITE settings
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks person}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400}"  # Even fewer steps
RESOLUTION="${RESOLUTION:-384}"  # Lower resolution
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"  # Accumulate gradients

echo ""
echo "========================================="
echo "ULTRA LITE Configuration"
echo "========================================="
echo "Instance prompt: $INSTANCE_PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Max training steps: $MAX_TRAIN_STEPS"
echo "Resolution: $RESOLUTION (lower = less memory)"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Mixed precision: NO (using fp32 for stability)"
echo "Prior preservation: NO"
echo ""
echo "Ultra-conservative settings:"
echo "  ✅ Lower resolution (384 vs 512)"
echo "  ✅ Fewer steps (400 vs 1000)"
echo "  ✅ No mixed precision (fp32)"
echo "  ✅ Gradient accumulation (less memory)"
echo "  ✅ No prior preservation"
echo "  ✅ No xformers"
echo "  ✅ No 8-bit optimizer"
echo "========================================="
echo ""

# Show available memory
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits | awk '{printf "  Total: %.1f GB, Free: %.1f GB, Used: %.1f GB\n", $1/1024, $2/1024, $3/1024}'
    echo ""
fi

# Ultra minimal command
echo "Starting training with ultra-conservative settings..."
echo ""

python train_dreambooth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="./instance_images" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --output_dir="$OUTPUT_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=1 \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="no" \
    --checkpointing_steps=999999 \
    --seed=42 \
    2>&1 | tee training.log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Training completed successfully!"
    echo "========================================="
    echo "Model saved to: $OUTPUT_DIR"
else
    echo ""
    echo "========================================="
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "========================================="
    echo ""
    echo "Check training.log for details"
    echo ""
    echo "If segfault still occurs, try:"
    echo "  1. Restart Kaggle kernel completely"
    echo "  2. Use P100 GPU (more memory)"
    echo "  3. Clear cache: rm -rf ~/.cache/huggingface/"
    echo "  4. Reduce resolution: export RESOLUTION=256"
    exit $EXIT_CODE
fi
