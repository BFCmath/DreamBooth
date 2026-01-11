#!/usr/bin/env python3
"""
Quick Start Script for Kaggle
Run this in a Kaggle notebook cell for easy setup and training
"""

# Step 1: Clone the repository (if needed)
# !git clone <your-repo-url>
# %cd <repo-name>

# Step 2: Upload images to instance_images/ directory
# Use Kaggle's upload feature or link a dataset

# Step 3: Run this cell to train
import os
import subprocess

# Configuration - CHANGE THESE VALUES
INSTANCE_PROMPT = "a photo of sks person"  # Change 'person' to 'dog', 'cat', 'toy', etc.
CLASS_PROMPT = "a photo of person"         # Match the class from above
MAX_TRAIN_STEPS = 1000                     # 800-1200 is typical
LEARNING_RATE = 2e-6                       # 1e-6 to 5e-6
RESOLUTION = 512                           # 512 is standard, reduce to 384 if OOM

# Set environment variables
os.environ["INSTANCE_PROMPT"] = INSTANCE_PROMPT
os.environ["CLASS_PROMPT"] = CLASS_PROMPT
os.environ["MAX_TRAIN_STEPS"] = str(MAX_TRAIN_STEPS)
os.environ["LEARNING_RATE"] = str(LEARNING_RATE)
os.environ["RESOLUTION"] = str(RESOLUTION)

# Run training
print("Starting DreamBooth training...")
subprocess.run(["bash", "run.sh"], check=True)

print("\n" + "="*50)
print("Training complete!")
print("="*50)

# Step 4: Generate images with your trained model
print("\nGenerating test images...")

from diffusers import StableDiffusionPipeline
import torch

# Load trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "./output/dreambooth-model",
    torch_dtype=torch.float16
).to("cuda")

# Generate samples
test_prompts = [
    f"{INSTANCE_PROMPT}",
    f"{INSTANCE_PROMPT} in a beautiful garden",
    f"{INSTANCE_PROMPT} wearing sunglasses",
    f"oil painting of {INSTANCE_PROMPT}",
]

from IPython.display import display
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(test_prompts), figsize=(20, 5))

for i, prompt in enumerate(test_prompts):
    print(f"Generating: {prompt}")
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(f"test_output_{i}.png")
    axes[i].imshow(image)
    axes[i].axis('off')
    axes[i].set_title(prompt, fontsize=10)

plt.tight_layout()
plt.savefig("test_results.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nDone! Check test_output_*.png for results")
