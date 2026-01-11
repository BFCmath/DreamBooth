#!/bin/bash
# Example: Train DreamBooth with example dog images
# This is a complete end-to-end example for testing

set -e

echo "========================================="
echo "DreamBooth Example Training (Dog)"
echo "========================================="
echo ""

# Step 1: Download example images
echo "Step 1: Downloading example dog images..."
python download_example_images.py dog

echo ""
echo "Step 2: Starting training..."
echo ""

# Step 2: Train with proper prompts
export INSTANCE_PROMPT="a photo of sks dog"
export CLASS_PROMPT="a photo of dog"
export MAX_TRAIN_STEPS=800  # Slightly fewer steps for faster testing
export LEARNING_RATE=2e-6

bash run.sh

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "To generate images, run:"
echo ""
echo "python -c \""
echo "from diffusers import StableDiffusionPipeline"
echo "import torch"
echo "pipe = StableDiffusionPipeline.from_pretrained('./output/dreambooth-model', torch_dtype=torch.float16).to('cuda')"
echo "image = pipe('a photo of sks dog in a garden', num_inference_steps=50).images[0]"
echo "image.save('result.png')"
echo "print('Saved to result.png')"
echo "\""
