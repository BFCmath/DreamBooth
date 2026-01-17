#!/bin/bash
# DreamBooth + ControlNet Training - Optimized for Single GPU
# Combines ControlNet pose/edge conditioning with DreamBooth identity learning
#
# Required Data Structure:
#   data/
#   ├── instance_images/     # 3-5 images of your subject (different poses)
#   │   ├── 001.png
#   │   ├── 002.png
#   │   └── ...
#   ├── conditioning/        # OpenPose/edge for each instance image
#   │   ├── 001.png
#   │   ├── 002.png
#   │   └── ...
#   └── (optional) prompts.txt  # Custom prompts per image

set -e

echo "=============================================="
echo "DreamBooth + ControlNet Training"
echo "=============================================="
echo "Identity-oriented ControlNet with rare token [V]"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub bitsandbytes

# Reduce fragmentation on Kaggle GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories
mkdir -p data/instance_images data/conditioning output logs

# =============================================
# Configuration
# =============================================
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/controlnet-dreambooth}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-runwayml/stable-diffusion-v1-5}"

# DreamBooth Identity Config
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks person}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of person}"
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-true}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-100}"

# Training Config (optimized for identity learning)
LEARNING_RATE="${LEARNING_RATE:-5e-6}"        # Lower LR for identity
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-800}"     # More steps for small dataset
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
RESOLUTION="${RESOLUTION:-512}"
REPEATS="${REPEATS:-100}"                      # Repeat instance images
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-200}"

# Memory Optimization
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-true}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"

# Optional: Load from existing ControlNet checkpoint
# export CONTROLNET_MODEL="lllyasviel/control_v11p_sd15_openpose"

# =============================================
# Validation
# =============================================
echo "Checking data directories..."

if [ ! -d "$DATA_DIR/instance_images" ] || [ -z "$(ls -A $DATA_DIR/instance_images 2>/dev/null)" ]; then
    echo "❌ ERROR: No instance images found in $DATA_DIR/instance_images"
    echo ""
    echo "Please add 3-5 images of your subject (person/object)"
    echo "Each image should show the subject in a DIFFERENT POSE"
    exit 1
fi

if [ ! -d "$DATA_DIR/conditioning" ] || [ -z "$(ls -A $DATA_DIR/conditioning 2>/dev/null)" ]; then
    echo "❌ ERROR: No conditioning images found in $DATA_DIR/conditioning"
    echo ""
    echo "Please add OpenPose/edge conditioning for each instance image"
    echo "Filenames must match instance images (001.png -> 001.png)"
    exit 1
fi

NUM_INSTANCE=$(ls $DATA_DIR/instance_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
NUM_COND=$(ls $DATA_DIR/conditioning/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)

echo "✅ Found $NUM_INSTANCE instance images"
echo "✅ Found $NUM_COND conditioning images"

if [ "$NUM_INSTANCE" -ne "$NUM_COND" ]; then
    echo "⚠️  Warning: Instance and conditioning count mismatch!"
fi

# =============================================
# Configuration Summary
# =============================================
echo ""
echo "=============================================="
echo "Configuration"
echo "=============================================="
echo ""
echo "📁 Paths:"
echo "   Data directory: $DATA_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo "   Base model: $PRETRAINED_MODEL"
echo ""
echo "🎭 DreamBooth Identity:"
echo "   Instance prompt: $INSTANCE_PROMPT"
echo "   Class prompt: $CLASS_PROMPT"
echo "   Prior preservation: $WITH_PRIOR_PRESERVATION"
echo "   Prior loss weight: $PRIOR_LOSS_WEIGHT"
echo "   Num class images: $NUM_CLASS_IMAGES"
echo ""
echo "🔧 Training:"
echo "   Learning rate: $LEARNING_RATE"
echo "   Max steps: $MAX_TRAIN_STEPS"
echo "   Batch size: $TRAIN_BATCH_SIZE"
echo "   Gradient accumulation: $GRADIENT_ACCUMULATION (effective batch = $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION)))"
echo "   Resolution: $RESOLUTION"
echo "   Instance repeats: $REPEATS"
echo ""
echo "💾 Memory Optimization:"
echo "   Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "   8-bit Adam: $USE_8BIT_ADAM"
echo "   Mixed precision: $MIXED_PRECISION"
echo ""

# Show GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "🔋 GPU Status:"
    nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits | awk '{printf "   Total: %.1f GB, Free: %.1f GB, Used: %.1f GB\n", $1/1024, $2/1024, $3/1024}'
    echo ""
fi

echo "=============================================="
echo "🚀 Starting Training..."
echo "=============================================="
echo ""

# =============================================
# Run Training
# =============================================
python dreambooth_controlnet.py \
    --data_dir="$DATA_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --pretrained_model="$PRETRAINED_MODEL" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --class_prompt="$CLASS_PROMPT" \
    $(if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then echo "--with_prior_preservation"; fi) \
    --prior_loss_weight=$PRIOR_LOSS_WEIGHT \
    --num_class_images=$NUM_CLASS_IMAGES \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --learning_rate=$LEARNING_RATE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_steps=$CHECKPOINTING_STEPS \
    --repeats=$REPEATS \
    --seed=42 \
    $(if [ "$GRADIENT_CHECKPOINTING" = "false" ]; then echo "--no_gradient_checkpointing"; fi) \
    $(if [ "$USE_8BIT_ADAM" = "false" ]; then echo "--no_8bit_adam"; fi) \
    $(if [ "$MIXED_PRECISION" = "false" ]; then echo "--no_mixed_precision"; fi) \
    2>&1 | tee training.log

EXIT_CODE=$?

# =============================================
# Completion
# =============================================
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ Training completed successfully!"
    echo "=============================================="
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "To use your trained model:"
    echo "  from diffusers import StableDiffusionControlNetPipeline, ControlNetModel"
    echo "  controlnet = ControlNetModel.from_pretrained('$OUTPUT_DIR')"
    echo "  pipe = StableDiffusionControlNetPipeline.from_pretrained("
    echo "      '$PRETRAINED_MODEL', controlnet=controlnet)"
    echo "  image = pipe('$INSTANCE_PROMPT', image=pose_image).images[0]"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "=============================================="
    echo "Check training.log for details"
    exit $EXIT_CODE
fi
