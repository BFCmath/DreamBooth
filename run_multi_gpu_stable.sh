#!/bin/bash
# DreamBooth Multi-GPU Training - STABLE VERSION
# This version avoids common multi-GPU pitfalls on Kaggle

set -e

echo "========================================="
echo "DreamBooth Multi-GPU Training (STABLE)"
echo "========================================="

# Check GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "⚠️  Only $NUM_GPUS GPU detected. Use run_lite.sh instead."
    exit 1
fi

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

# STABLE Multi-GPU settings
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks person}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of person}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-800}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
RESOLUTION="${RESOLUTION:-512}"
# Disable prior preservation for stability with multi-GPU
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-false}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-50}"

echo ""
echo "========================================="
echo "Stable Multi-GPU Configuration"
echo "========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Instance prompt: $INSTANCE_PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Max training steps: $MAX_TRAIN_STEPS"
echo "Batch size per GPU: $TRAIN_BATCH_SIZE"
echo "Resolution: $RESOLUTION"
echo "Prior preservation: $WITH_PRIOR_PRESERVATION (disabled for stability)"
echo "Mixed precision: fp32 (more stable for multi-GPU)"
echo ""
echo "Stability optimizations:"
echo "  ✅ No fp16 (using fp32)"
echo "  ✅ Prior preservation disabled by default"
echo "  ✅ Simple accelerate auto-detection"
echo "========================================="
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | while read line; do
    echo "  GPU $line"
done
echo ""

# Use simple accelerate launch (auto-detect GPUs)
echo "Starting training with accelerate auto-detection..."
echo ""

accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=no \
    train_dreambooth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="./instance_images" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --output_dir="$OUTPUT_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="no" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=2 \
    --seed=42 \
    $(if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then echo "--with_prior_preservation --class_data_dir=./class_images --class_prompt=\"$CLASS_PROMPT\" --num_class_images=$NUM_CLASS_IMAGES --sample_batch_size=2 --prior_loss_weight=1.0"; fi) \
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
    echo "❌ Training failed"
    echo "========================================="
    echo "Check training.log for details"
    exit $EXIT_CODE
fi
