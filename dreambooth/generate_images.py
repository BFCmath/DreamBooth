#!/usr/bin/env python3
"""
Generate images using trained DreamBooth model
Run after training completes to test your custom model
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from pathlib import Path


def generate_images(
    model_path="./output/dreambooth-model",
    prompts=None,
    output_dir="./generated_images",
    num_images_per_prompt=1,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    seed=None,
):
    """Generate images using trained DreamBooth model."""
    
    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "a sks humanoid robot",
            "a sks humanoid robot in a suit",
            "a sks humanoid robot at the beach",
            "oil painting of sks person",
            "sks person as a superhero",
        ]
    
    print("=" * 60)
    print("DreamBooth Image Generation")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Images per prompt: {num_images_per_prompt}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Resolution: {width}x{height}")
    if seed is not None:
        print(f"Seed: {seed}")
    print("=" * 60)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the trained model
    print(f"⏳ Loading trained model from {model_path}...")
    try:
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,  # Disable safety checker for faster inference
        )
        pipe = pipe.to(device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure the model path is correct and training completed successfully.")
        return
    
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    print()
    print("=" * 60)
    print("Generating Images")
    print("=" * 60)
    
    # Generate images for each prompt
    image_count = 0
    all_images = []
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n📝 Prompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")
        
        for img_idx in range(num_images_per_prompt):
            try:
                print(f"  ⏳ Generating image {img_idx + 1}/{num_images_per_prompt}...")
                
                # Generate image
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                ).images[0]
                
                # Save image
                filename = f"image_{image_count:04d}_prompt{prompt_idx}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                all_images.append((prompt, image, filepath))
                image_count += 1
                
                print(f"  ✅ Saved: {filepath}")
                
            except Exception as e:
                print(f"  ❌ Error generating image: {e}")
                continue
    
    print()
    print("=" * 60)
    print(f"✅ Generation Complete!")
    print("=" * 60)
    print(f"Generated {image_count} images")
    print(f"Saved to: {output_dir}")
    print()
    
    # Create a summary HTML file
    try:
        html_path = os.path.join(output_dir, "gallery.html")
        with open(html_path, "w") as f:
            f.write("<html><head><title>DreamBooth Generated Images</title>")
            f.write("<style>body{font-family:Arial;margin:20px;background:#f5f5f5;}")
            f.write(".image-container{margin:20px;padding:20px;background:white;border-radius:8px;}")
            f.write("img{max-width:512px;border:1px solid #ddd;border-radius:4px;}")
            f.write("h3{color:#333;}</style></head><body>")
            f.write("<h1>🎨 DreamBooth Generated Images</h1>")
            
            for prompt, _, filepath in all_images:
                filename = os.path.basename(filepath)
                f.write(f"<div class='image-container'>")
                f.write(f"<h3>{prompt}</h3>")
                f.write(f"<img src='{filename}' alt='{prompt}'>")
                f.write(f"</div>")
            
            f.write("</body></html>")
        
        print(f"📄 Gallery created: {html_path}")
        print("   Open this file in a browser to view all images!")
        print()
        
    except Exception as e:
        print(f"⚠️  Could not create gallery HTML: {e}")
    
    return all_images


def main():
    parser = argparse.ArgumentParser(description="Generate images using trained DreamBooth model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/dreambooth-model",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="List of prompts to generate images for",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps (more = better quality, slower)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale (higher = more prompt adherence)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    generate_images(
        model_path=args.model_path,
        prompts=args.prompts,
        output_dir=args.output_dir,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
