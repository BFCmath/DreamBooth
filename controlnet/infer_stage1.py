#!/usr/bin/env python3
"""
Stage 1 Inference: Test identity learning before/after training

This script uses standard Stable Diffusion (NO ControlNet) to test
the Stage 1 trained model which learns identity from text only.

Usage (Before training - baseline):
    python infer_stage1.py --prompt "a photo of sks person" --output_dir ./before_training

Usage (After training - with LoRA):
    python infer_stage1.py \
        --prompt "a photo of sks person" \
        --lora_path ./output/stage1-appearance \
        --output_dir ./after_training

This helps verify that Stage 1 correctly learned the "sks" identity.
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gc


def generate_images(
    prompt: str,
    output_dir: str = "./output_stage1_infer",
    model_path: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str = None,  # Path to Stage 1 output (contains unet_lora/ and text_encoder_lora/)
    num_images: int = 4,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: int = None,
    negative_prompt: str = "lowres, bad anatomy, worst quality, low quality, deformed, ugly",
):
    """
    Generate images using standard Stable Diffusion (no ControlNet).
    Optionally loads Stage 1 trained LoRA weights.
    """
    
    print("=" * 60)
    print("🎨 Stage 1 Inference (No ControlNet)")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    else:
        print("LoRA: None (baseline test)")
    print(f"Output: {output_dir}")
    print(f"Images: {num_images}")
    print(f"Steps: {num_inference_steps}")
    print(f"CFG: {guidance_scale}")
    print(f"Resolution: {width}x{height}")
    if seed is not None:
        print(f"Seed: {seed}")
    print("=" * 60)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"🖥️  Device: {device}")
    
    # Load pipeline
    print(f"⏳ Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
    )
    
    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print("✅ Pipeline loaded!")
    
    # Load LoRA weights if provided
    if lora_path:
        from peft import PeftModel
        
        unet_lora = os.path.join(lora_path, "unet_lora")
        text_encoder_lora = os.path.join(lora_path, "text_encoder_lora")
        
        if os.path.exists(unet_lora):
            print(f"⏳ Loading UNet LoRA from {unet_lora}")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora)
            print("✅ UNet LoRA loaded!")
        else:
            print(f"⚠️  UNet LoRA not found at {unet_lora}")
        
        if os.path.exists(text_encoder_lora):
            print(f"⏳ Loading Text Encoder LoRA from {text_encoder_lora}")
            pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_lora)
            print("✅ Text Encoder LoRA loaded!")
        else:
            print(f"⚠️  Text Encoder LoRA not found at {text_encoder_lora}")
    
    # Move to device
    pipe = pipe.to(device)
    
    # Memory optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers enabled")
    except:
        pipe.enable_attention_slicing()
        print("✅ Attention slicing enabled")
    pipe.enable_vae_slicing()
    
    print()
    
    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate images
    print("=" * 60)
    print("🎨 Generating Images")
    print("=" * 60)
    
    all_images = []
    for i in range(num_images):
        print(f"\n⏳ Generating image {i + 1}/{num_images}...")
        
        # Use different seeds for each image if base seed provided
        if seed is not None and i > 0:
            generator = torch.Generator(device=device).manual_seed(seed + i)
        
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
            
            # Save image
            filename = f"stage1_output_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            all_images.append(filepath)
            print(f"✅ Saved: {filepath}")
            
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error generating image {i + 1}: {e}")
            continue
    
    print()
    print("=" * 60)
    print(f"✅ Generation Complete!")
    print("=" * 60)
    print(f"Generated {len(all_images)} images")
    print(f"Saved to: {output_dir}")
    
    # Create simple comparison gallery
    try:
        html_path = os.path.join(output_dir, "gallery.html")
        with open(html_path, "w") as f:
            f.write("<html><head><title>Stage 1 Inference Results</title>")
            f.write("<style>body{font-family:Arial;margin:20px;background:#1a1a2e;color:#eee;}")
            f.write(".container{display:flex;flex-wrap:wrap;gap:20px;}")
            f.write(".image-card{background:#16213e;padding:20px;border-radius:12px;max-width:540px;}")
            f.write("img{max-width:512px;border-radius:8px;}")
            f.write("h3{color:#e94560;}h1{color:#0f3460;}</style></head><body>")
            f.write("<h1>🎨 Stage 1 Inference (No ControlNet)</h1>")
            f.write(f"<p>Prompt: <em>{prompt}</em></p>")
            if lora_path:
                f.write(f"<p>LoRA: <code>{lora_path}</code></p>")
            else:
                f.write("<p>LoRA: <em>None (baseline)</em></p>")
            f.write("<div class='container'>")
            
            for filepath in all_images:
                filename = os.path.basename(filepath)
                f.write("<div class='image-card'>")
                f.write(f"<img src='{filename}' alt='{prompt}'>")
                f.write("</div>")
            
            f.write("</div></body></html>")
        
        print(f"📄 Gallery created: {html_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create gallery: {e}")
    
    return all_images


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 Inference: Test identity learning (no ControlNet)"
    )
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation (e.g., 'a photo of sks person')")
    parser.add_argument("--output_dir", type=str, default="./output_stage1_infer",
                        help="Directory to save generated images")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base Stable Diffusion model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to Stage 1 output containing unet_lora/ and text_encoder_lora/")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG scale")
    parser.add_argument("--height", type=int, default=512,
                        help="Output height")
    parser.add_argument("--width", type=int, default=512,
                        help="Output width")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--negative_prompt", type=str,
                        default="lowres, bad anatomy, worst quality, low quality, deformed, ugly",
                        help="Negative prompt")
    
    args = parser.parse_args()
    
    generate_images(
        prompt=args.prompt,
        output_dir=args.output_dir,
        model_path=args.model_path,
        lora_path=args.lora_path,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )


if __name__ == "__main__":
    main()
