#!/bin/bash
# DreamBooth Training - Optimized for Single GPU with Gradient Checkpointing
# This uses memory-efficient techniques to train on single T4

set -e

echo "========================================="
echo "DreamBooth Single GPU (Memory Optimized)"
echo "========================================="

# Install dependencies (include bitsandbytes for 8-bit Adam)
echo "Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub bitsandbytes

# Reduce fragmentation on Kaggle GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a sks humanoid robot}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of person}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-800}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"  # Effective batch size = 4
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-true}"
RESOLUTION="${RESOLUTION:-512}"
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-false}"  # Disabled to save memory
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-50}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

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
echo "Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "Resolution: $RESOLUTION"
echo "Prior preservation: $WITH_PRIOR_PRESERVATION"
echo "Mixed precision: $MIXED_PRECISION"
echo ""
echo "Memory optimizations:"
echo "  ✅ Gradient accumulation (4 steps)"
echo "  ✅ Gradient checkpointing"
echo "  ✅ 8-bit Adam: $USE_8BIT_ADAM"
echo "  ✅ Mixed precision: $MIXED_PRECISION"
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
    $(if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then echo "--gradient_checkpointing"; fi) \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    $(if [ "$USE_8BIT_ADAM" = "true" ]; then echo "--use_8bit_adam"; fi) \
    --mixed_precision="$MIXED_PRECISION" \
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
