#!/bin/bash
# ControlNet Fine-tuning Script for Kaggle P100
# Run this script to fine-tune ControlNet with your own data

set -e  # Exit on error

echo "========================================="
echo "ControlNet Fine-tuning"
echo "========================================="

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q bitsandbytes 2>/dev/null || echo "⚠️  bitsandbytes not installed"

# CRITICAL: Uninstall xformers if it causes issues
pip uninstall -y xformers 2>/dev/null || true

# =========================================
# CONFIGURATION
# =========================================

# Data directory structure:
#   data/
#   ├── images/           # Ground truth images
#   ├── conditioning/     # Conditioning images (pose/edges)
#   └── prompts.txt       # One prompt per line
DATA_DIR="${DATA_DIR:-./data}"

# Output directory for trained model
OUTPUT_DIR="${OUTPUT_DIR:-./output/controlnet-finetuned}"

# Base model
PRETRAINED_MODEL="${PRETRAINED_MODEL:-runwayml/stable-diffusion-v1-5}"

# Training parameters
RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
SEED="${SEED:-42}"

# =========================================

echo ""
echo "========================================="
echo "Configuration"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Base model: $PRETRAINED_MODEL"
echo "Resolution: $RESOLUTION"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Effective batch size: $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_TRAIN_STEPS"
echo "========================================="
echo ""

# Check data directory structure
if [ ! -d "$DATA_DIR/images" ] || [ ! -d "$DATA_DIR/conditioning" ] || [ ! -f "$DATA_DIR/prompts.txt" ]; then
    echo "========================================="
    echo "❌ ERROR: Data directory structure invalid!"
    echo "========================================="
    echo ""
    echo "Expected structure:"
    echo "  $DATA_DIR/"
    echo "  ├── images/           # Ground truth images (001.png, 002.png, ...)"
    echo "  ├── conditioning/     # Conditioning images (same names as images)"
    echo "  └── prompts.txt       # One prompt per line"
    echo ""
    echo "Example setup:"
    echo "  mkdir -p $DATA_DIR/images $DATA_DIR/conditioning"
    echo "  # Add your images..."
    echo "  # Add conditioning (pose/edges)..."
    echo "  # Create prompts.txt with one prompt per line"
    echo ""
    exit 1
fi

# Count files
NUM_IMAGES=$(ls $DATA_DIR/images/*.{png,jpg,jpeg} 2>/dev/null | wc -l)
NUM_COND=$(ls $DATA_DIR/conditioning/*.{png,jpg,jpeg} 2>/dev/null | wc -l)
NUM_PROMPTS=$(wc -l < $DATA_DIR/prompts.txt)

echo "📊 Dataset:"
echo "  Ground truth images: $NUM_IMAGES"
echo "  Conditioning images: $NUM_COND"
echo "  Prompts: $NUM_PROMPTS"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU: %s | Total: %.1f GB | Free: %.1f GB\n", $1, $2/1024, $3/1024}'
    echo ""
fi

# Run training
echo "🚀 Starting ControlNet fine-tuning..."
echo ""

python train_controlnet.py \
    --data_dir="$DATA_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --pretrained_model="$PRETRAINED_MODEL" \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --learning_rate=$LEARNING_RATE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_steps=$CHECKPOINTING_STEPS \
    --seed=$SEED

echo ""
echo "========================================="
echo "✅ Fine-tuning Complete!"
echo "========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To use your trained ControlNet for inference:"
echo "  MODEL_PATH=$OUTPUT_DIR python infer_controlnet.py --prompt \"your prompt\" --input_image pose.png"
echo ""
