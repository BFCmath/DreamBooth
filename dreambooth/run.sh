#!/bin/bash
# DreamBooth Training Script for Kaggle
# This script sets up the environment and runs proper DreamBooth training with prior preservation

set -e  # Exit on error

echo "========================================="
echo "DreamBooth Training for Stable Diffusion"
echo "========================================="

# Install dependencies (skip torch/torchvision to avoid conflicts with Kaggle's preinstalled versions)
echo "Installing dependencies..."
pip install -q diffusers transformers accelerate huggingface_hub

# Try to install xformers (may fail on some platforms, that's okay)
pip install -q xformers 2>/dev/null || echo "Warning: xformers not installed (optional, for memory efficiency)"

# Try to install bitsandbytes (may fail on some platforms, that's okay)
pip install -q bitsandbytes 2>/dev/null || echo "Warning: bitsandbytes not installed (optional, for 8-bit Adam)"

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
    echo "You can do this by:"
    echo "  1. Download example images:"
    echo "     python download_example_images.py dog"
    echo ""
    echo "  2. Upload your own images:"
    echo "     - Use Kaggle's file upload feature"
    echo "     - Link a Kaggle dataset"
    echo "     - Use wget/curl to download"
    echo ""
    echo "Example with your own images:"
    echo "  mkdir -p instance_images"
    echo "  wget -O instance_images/img1.jpg <your-image-url>"
    exit 1
fi

# Count instance images
NUM_IMAGES=$(ls instance_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES training images in instance_images/"

# Set default values if not provided
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks person}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of person}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dreambooth-model}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
RESOLUTION="${RESOLUTION:-512}"
WITH_PRIOR_PRESERVATION="${WITH_PRIOR_PRESERVATION:-true}"
NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-100}"  # Reduced from 200 to save memory
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-2}"  # Batch size for generating class images

echo ""
echo "========================================="
echo "Training Configuration"
echo "========================================="
echo "Instance prompt: $INSTANCE_PROMPT"
echo "Class prompt: $CLASS_PROMPT"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Max training steps: $MAX_TRAIN_STEPS"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Resolution: $RESOLUTION"
echo "Prior preservation: $WITH_PRIOR_PRESERVATION"
if [ "$WITH_PRIOR_PRESERVATION" = "true" ]; then
    echo "Number of class images: $NUM_CLASS_IMAGES"
    echo "Sample batch size: $SAMPLE_BATCH_SIZE"
    echo "Prior loss weight: $PRIOR_LOSS_WEIGHT"
fi
echo "========================================="
echo ""

# Check available memory
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Check:"
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "  Total: %.1f GB, Free: %.1f GB\n", $1/1024, $2/1024}'
    echo ""
fi

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
echo "Starting DreamBooth training..."
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
