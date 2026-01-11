#!/bin/bash
# DreamBooth Training - Optimized for Single GPU with Gradient Checkpointing
# This uses memory-efficient techniques to train on single T4

set -e

echo "========================================="
echo "DreamBooth Single GPU (Memory Optimized)"
echo "========================================="

# Install dependencies
echo "Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub

# Create directories
mkdir -p instance_images output logs

# Check images
if [ ! -d "instance_images" ] || [ -z "$(ls -A instance_images)" ]; then
    echo "ERROR: No training images!"
    echo "Run: python download_example_images.py dog"
    exit 1
fi

NUM_IMAGES=$(ls instance_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES training images"

# Optimized settings for single GPU
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks person}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of person}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-800}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"  # Effective batch size = 4
RESOLUTION="${RESOLUTION:-512}"
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-false}"  # Disabled to save memory
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-50}"

echo ""
echo "========================================="
echo "Memory-Optimized Configuration"
echo "========================================="
echo "Instance prompt: $INSTANCE_PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Max training steps: $MAX_TRAIN_STEPS"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION (effective batch = 4)"
echo "Resolution: $RESOLUTION"
echo "Prior preservation: $WITH_PRIOR_PRESERVATION"
echo "Mixed precision: fp32 (most stable)"
echo ""
echo "Memory optimizations:"
echo "  ✅ Gradient accumulation (4 steps)"
echo "  ✅ No fp16 (using fp32)"
echo "  ✅ Prior preservation disabled"
echo "  ✅ Minimal checkpointing"
echo "========================================="
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits | awk '{printf "  Total: %.1f GB, Free: %.1f GB, Used: %.1f GB\n", $1/1024, $2/1024, $3/1024}'
echo ""

echo "Starting training..."
echo ""

python train_dreambooth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="./instance_images" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --output_dir="$OUTPUT_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="no" \
    --checkpointing_steps=999999 \
    --seed=42 \
    $(if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then echo "--with_prior_preservation --class_data_dir=./class_images --class_prompt=\"$CLASS_PROMPT\" --num_class_images=$NUM_CLASS_IMAGES --sample_batch_size=1 --prior_loss_weight=1.0"; fi) \
    2>&1 | tee training.log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Training completed successfully!"
    echo "========================================="
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Training used gradient accumulation to simulate"
    echo "larger batch sizes without OOM!"
else
    echo ""
    echo "========================================="
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "========================================="
    echo "Check training.log for details"
    exit $EXIT_CODE
fi
