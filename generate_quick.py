#!/usr/bin/env python3
"""
Quick image generation for Kaggle notebooks
Just run this after training to see results immediately
"""

from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display
import os

# Configuration
MODEL_PATH = "./output/dreambooth-model"
PROMPTS = [
    "a photo of sks person",
    "a photo of sks person in a suit",
    "a photo of sks person at the beach",
    "oil painting of sks person",
    "sks person as a superhero",
]

print("🎨 Quick DreamBooth Image Generation")
print("=" * 60)

# Load model
print(f"⏳ Loading model from {MODEL_PATH}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
print("✅ Model loaded!\n")

# Generate images
os.makedirs("./generated_images", exist_ok=True)

for i, prompt in enumerate(PROMPTS):
    print(f"📝 {i+1}/{len(PROMPTS)}: '{prompt}'")
    
    image = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]
    
    # Save
    filename = f"./generated_images/image_{i:02d}.png"
    image.save(filename)
    print(f"   💾 Saved: {filename}")
    
    # Display in notebook (if running in Kaggle/Jupyter)
    try:
        display(image)
    except:
        pass
    
    print()

print("=" * 60)
print("✅ Done! Check ./generated_images/ folder")
