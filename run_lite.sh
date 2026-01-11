#!/bin/bash
# DreamBooth Training Script - LITE MODE for Kaggle
# This version uses minimal memory and is more stable on low-resource environments

set -e  # Exit on error

echo "========================================="
echo "DreamBooth LITE MODE (Memory Optimized)"
echo "========================================="

# Install dependencies (skip torch/torchvision to avoid conflicts)
echo "Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub

# Skip xformers and bitsandbytes to avoid segfaults
echo "Skipping optional packages for stability..."

# Create necessary directories
mkdir -p instance_images
mkdir -p output
mkdir -p logs

# Check if instance images directory exists and has images
if [ ! -d "instance_images" ] || [ -z "$(ls -A instance_images)" ]; then
    echo "========================================="
    echo "ERROR: No training images found!"
    echo "========================================="
    echo "Please add your training images to the instance_images/ directory."
    echo ""
    echo "Quick start:"
    echo "  python download_example_images.py dog"
    exit 1
fi

# Count instance images
NUM_IMAGES=$(ls instance_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES training images in instance_images/"

# LITE MODE defaults - optimized for stability and low memory
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks person}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of person}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-800}"  # Slightly fewer steps
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
RESOLUTION="${RESOLUTION:-512}"
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-false}"  # Disabled by default in lite mode
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-50}"  # Much fewer if enabled
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"  # Smaller batches

echo ""
echo "========================================="
echo "LITE MODE Configuration"
echo "========================================="
echo "Instance prompt: $INSTANCE_PROMPT"
echo "Class prompt: $CLASS_PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Max training steps: $MAX_TRAIN_STEPS"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Resolution: $RESOLUTION"
echo "Prior preservation: $WITH_PRIOR_PRESERVATION (disabled for memory savings)"
if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then
    echo "Number of class images: $NUM_CLASS_IMAGES"
fi
echo ""
echo "Memory optimizations enabled:"
echo "  - No xformers (avoids segfaults)"
echo "  - No 8-bit Adam (more stable)"
echo "  - Smaller class image set"
echo "  - Conservative settings"
echo "========================================="
echo ""

# Build command
CMD="python train_dreambooth.py \
    --pretrained_model_name_or_path=\"runwayml/stable-diffusion-v1-5\" \
    --instance_data_dir=\"./instance_images\" \
    --instance_prompt=\"$INSTANCE_PROMPT\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler=\"constant\" \
    --lr_warmup_steps=0 \
    --mixed_precision=\"fp16\" \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=2 \
    --seed=42"

# Add prior preservation arguments if enabled
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
echo "Starting DreamBooth training (LITE MODE)..."
echo ""
eval $CMD

echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To use your trained model:"
echo ""
echo "from diffusers import StableDiffusionPipeline"
echo "import torch"
echo ""
echo "pipe = StableDiffusionPipeline.from_pretrained("
echo "    \"$OUTPUT_DIR\","
echo "    torch_dtype=torch.float16"
echo ").to(\"cuda\")"
echo ""
echo "image = pipe(\"$INSTANCE_PROMPT\", num_inference_steps=50).images[0]"
echo "image.save(\"output.png\")"
echo "========================================="
