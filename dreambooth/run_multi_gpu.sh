#!/bin/bash
# DreamBooth Training Script for MULTI-GPU (2x T4 on Kaggle)
# This script uses both GPUs with distributed training via accelerate

set -e  # Exit on error

echo "========================================="
echo "DreamBooth Multi-GPU Training (2x T4)"
echo "========================================="

# Check number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "⚠️  WARNING: Only $NUM_GPUS GPU(s) detected!"
    echo "This script is optimized for 2 GPUs."
    echo "Consider using run.sh or run_lite.sh instead."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub

# Optional packages
pip install -q xformers 2>/dev/null || echo "Warning: xformers not installed (optional)"

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

# Multi-GPU optimized settings
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a sks humanoid robot}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of person}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"  # Per GPU
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"  # Can be 1 since we have 2 GPUs
RESOLUTION="${RESOLUTION:-512}"
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-true}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"

echo ""
echo "========================================="
echo "Multi-GPU Configuration"
echo "========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Instance prompt: $INSTANCE_PROMPT"
echo "Class prompt: $CLASS_PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Max training steps: $MAX_TRAIN_STEPS"
echo "Batch size per GPU: $TRAIN_BATCH_SIZE"
echo "Effective batch size: $(($TRAIN_BATCH_SIZE * $NUM_GPUS * $GRADIENT_ACCUMULATION))"
echo "Resolution: $RESOLUTION"
echo "Prior preservation: $WITH_PRIOR_PRESERVATION"
if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then
    echo "Number of class images: $NUM_CLASS_IMAGES"
fi
echo "========================================="
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while read line; do
    echo "  GPU $line"
done
echo ""

# Setup accelerate config for multi-GPU
export ACCELERATE_CONFIG_FILE="./accelerate_config_multi_gpu.yaml"

# Build command
CMD="accelerate launch --config_file=accelerate_config_multi_gpu.yaml train_dreambooth.py \
    --pretrained_model_name_or_path=\"runwayml/stable-diffusion-v1-5\" \
    --instance_data_dir=\"./instance_images\" \
    --instance_prompt=\"$INSTANCE_PROMPT\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler=\"constant\" \
    --lr_warmup_steps=0 \
    --mixed_precision=\"fp16\" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=3 \
    --seed=42"

# Add prior preservation if enabled
if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then
    CMD="$CMD \
    --with_prior_preservation \
    --class_data_dir=\"./class_images\" \
    --class_prompt=\"$CLASS_PROMPT\" \
    --num_class_images=$NUM_CLASS_IMAGES \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --prior_loss_weight=$PRIOR_LOSS_WEIGHT"
fi

# Run training
echo "Starting Multi-GPU DreamBooth training..."
echo "This will distribute the training across $NUM_GPUS GPUs!"
echo ""
eval $CMD

echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Multi-GPU training completed!"
echo "Both GPUs were used to speed up training."
echo "========================================="
