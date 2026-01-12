#!/bin/bash
# DreamBooth Training - HuggingFace Style (Optimized for 16GB P100)
# Based on: https://huggingface.co/docs/diffusers/en/training/dreambooth

set -e

echo "=============================================="
echo "DreamBooth Training - HuggingFace Style"
echo "Optimized for 16GB VRAM (Kaggle P100)"
echo "=============================================="

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub bitsandbytes

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories
mkdir -p instance_images output class_images logs

# Check for training images
if [ ! -d "instance_images" ] || [ -z "$(ls -A instance_images 2>/dev/null)" ]; then
    echo "❌ ERROR: No training images found!"
    echo "Run: python download_example_images.py dog"
    exit 1
fi

NUM_IMAGES=$(ls instance_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
echo "✅ Found $NUM_IMAGES training images"

# Configuration (customize these!)
MODEL_NAME="${MODEL_NAME:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
INSTANCE_DIR="${INSTANCE_DIR:-./instance_images}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks dog}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of dog}"

# Training hyperparameters (optimized for P100 16GB)
RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"

# Prior preservation settings
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-false}"
CLASS_DIR="${CLASS_DIR:-./class_images}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-100}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"

# Validation (optional)
VALIDATION_PROMPT="${VALIDATION_PROMPT:-}"
VALIDATION_STEPS="${VALIDATION_STEPS:-100}"

echo ""
echo "=============================================="
echo "Configuration"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Instance prompt: $INSTANCE_PROMPT"
echo "Output: $OUTPUT_DIR"
echo "Resolution: $RESOLUTION"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_TRAIN_STEPS"
echo "Prior preservation: $WITH_PRIOR_PRESERVATION"
echo ""
echo "Memory Optimizations (16GB P100):"
echo "  ✅ Gradient checkpointing"
echo "  ✅ 8-bit Adam optimizer"
echo "  ✅ Mixed precision (fp16)"
echo "  ❌ Text encoder training (requires 24GB+)"
echo "=============================================="
echo ""

# Show GPU status
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Build training command
TRAIN_CMD="python train_dreambooth_hf.py \
    --pretrained_model_name_or_path=\"$MODEL_NAME\" \
    --instance_data_dir=\"$INSTANCE_DIR\" \
    --instance_prompt=\"$INSTANCE_PROMPT\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler=\"$LR_SCHEDULER\" \
    --lr_warmup_steps=0 \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --mixed_precision=\"fp16\" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --checkpointing_steps=9999 \
    --seed=42"

# Add prior preservation if enabled
if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD \
    --with_prior_preservation \
    --class_data_dir=\"$CLASS_DIR\" \
    --class_prompt=\"$CLASS_PROMPT\" \
    --num_class_images=$NUM_CLASS_IMAGES \
    --prior_loss_weight=$PRIOR_LOSS_WEIGHT"
fi

# Add validation if prompt provided
if [ -n "$VALIDATION_PROMPT" ]; then
    TRAIN_CMD="$TRAIN_CMD \
    --validation_prompt=\"$VALIDATION_PROMPT\" \
    --validation_steps=$VALIDATION_STEPS \
    --num_validation_images=4"
fi

echo "🚀 Starting training..."
echo ""

# Run training
eval $TRAIN_CMD 2>&1 | tee training.log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ Training Complete!"
    echo "=============================================="
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "To generate images:"
    echo "  python generate_images.py --model_path=\"$OUTPUT_DIR\" \\"
    echo "    --prompts \"a photo of sks dog in a bucket\""
else
    echo ""
    echo "=============================================="
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "=============================================="
    echo "Check training.log for details"
    exit $EXIT_CODE
fi
