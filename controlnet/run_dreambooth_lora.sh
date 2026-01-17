#!/bin/bash
# DreamBooth LoRA Training
# Trains UNet LoRA to learn identity, then combine with any ControlNet
#
# Workflow:
# 1. Train LoRA on UNet (learns "sks cat" identity)
# 2. Inference: SD1.5 + LoRA + ControlNet (pose) = identity in any pose

set -e

echo "=============================================="
echo "DreamBooth LoRA Training"
echo "=============================================="
echo "LoRA learns identity in UNet"
echo "Then combine with any ControlNet for pose control"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q diffusers transformers accelerate peft bitsandbytes

# Reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories
mkdir -p data/instance_images output logs

# =============================================
# Configuration
# =============================================
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-lora}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-runwayml/stable-diffusion-v1-5}"

# Identity
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks cat}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of cat}"
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-true}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"

# LoRA Config
LORA_RANK="${LORA_RANK:-4}"      # Lower = smaller, faster
LORA_ALPHA="${LORA_ALPHA:-4}"    # Usually same as rank

# Training
LEARNING_RATE="${LEARNING_RATE:-1e-4}"     # Higher for LoRA
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
RESOLUTION="${RESOLUTION:-512}"
REPEATS="${REPEATS:-100}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"

MIXED_PRECISION="${MIXED_PRECISION:-true}"

# =============================================
# Validation
# =============================================
if [ ! -d "$DATA_DIR/instance_images" ] || [ -z "$(ls -A $DATA_DIR/instance_images 2>/dev/null)" ]; then
    echo "❌ ERROR: No instance images in $DATA_DIR/instance_images"
    echo "Add 3-5 images of your subject"
    exit 1
fi

NUM_INSTANCE=$(ls $DATA_DIR/instance_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
echo "✅ Found $NUM_INSTANCE instance images"

# =============================================
# Config Summary
# =============================================
echo ""
echo "=============================================="
echo "Configuration"
echo "=============================================="
echo ""
echo "🎭 Identity:"
echo "   Instance prompt: $INSTANCE_PROMPT"
echo "   Class prompt: $CLASS_PROMPT"
echo "   Prior preservation: $WITH_PRIOR_PRESERVATION"
echo ""
echo "🔧 LoRA:"
echo "   Rank: $LORA_RANK"
echo "   Alpha: $LORA_ALPHA"
echo "   Learning rate: $LEARNING_RATE"
echo ""
echo "📊 Training:"
echo "   Max steps: $MAX_TRAIN_STEPS"
echo "   Batch size: $TRAIN_BATCH_SIZE"
echo "   Grad accumulation: $GRADIENT_ACCUMULATION"
echo ""

# GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "🔋 GPU:"
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | awk '{printf "   Total: %.1f GB, Free: %.1f GB\n", $1/1024, $2/1024}'
    echo ""
fi

echo "=============================================="
echo "🚀 Starting Training..."
echo "=============================================="
echo ""

# =============================================
# Run Training
# =============================================
python train_dreambooth_lora.py \
    --data_dir="$DATA_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --pretrained_model="$PRETRAINED_MODEL" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --class_prompt="$CLASS_PROMPT" \
    $(if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then echo "--with_prior_preservation"; fi) \
    --prior_loss_weight=$PRIOR_LOSS_WEIGHT \
    --num_class_images=$NUM_CLASS_IMAGES \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --lora_rank=$LORA_RANK \
    --lora_alpha=$LORA_ALPHA \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --learning_rate=$LEARNING_RATE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_steps=$CHECKPOINTING_STEPS \
    --repeats=$REPEATS \
    --seed=42 \
    $(if [ "$MIXED_PRECISION" = "false" ]; then echo "--no_mixed_precision"; fi) \
    2>&1 | tee training.log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ Training completed!"
    echo "=============================================="
    echo ""
    echo "LoRA saved to: $OUTPUT_DIR"
    echo ""
    echo "Next: Run inference with ControlNet:"
    echo "  python infer_lora_controlnet.py \\"
    echo "    --lora_path $OUTPUT_DIR \\"
    echo "    --controlnet lllyasviel/control_v11p_sd15_canny \\"
    echo "    --prompt '$INSTANCE_PROMPT with a specific pose' \\"
    echo "    --control_image pose.png"
else
    echo "❌ Training failed. Check training.log"
    exit $EXIT_CODE
fi
