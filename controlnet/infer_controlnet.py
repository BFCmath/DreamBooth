#!/usr/bin/env python3
"""
ControlNet Inference Script with OpenPose Conditioning
Run on Kaggle P100 to generate pose-controlled images using Stable Diffusion 1.5

Supports:
- Automatic pose extraction from input image via controlnet-aux
- Pre-made pose/skeleton images as direct input
- Custom or DreamBooth-finetuned SD 1.5 models
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image


def extract_pose(image_path: str, detector):
    """Extract OpenPose skeleton from an image."""
    print(f"  ⏳ Extracting pose from: {image_path}")
    image = load_image(image_path)
    pose_image = detector(image)
    print("  ✅ Pose extracted successfully!")
    return pose_image


def generate_images(
    prompt: str,
    input_image: str,
    output_dir: str = "./output_images",
    model_path: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose",
    num_images: int = 1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    height: int = 512,
    width: int = 512,
    seed: int = None,
    use_pose_image_directly: bool = False,
    negative_prompt: str = "lowres, bad anatomy, worst quality, low quality, deformed, ugly",
    low_vram_mode: bool = False,
):
    """
    Generate images using ControlNet with OpenPose conditioning.
    
    Args:
        prompt: Text prompt for image generation
        input_image: Path to input image (for pose extraction) or pose image
        output_dir: Directory to save generated images
        model_path: Path to SD 1.5 model (can be DreamBooth-finetuned)
        controlnet_model: ControlNet model for pose conditioning
        num_images: Number of images to generate
        num_inference_steps: Denoising steps (higher = better quality)
        guidance_scale: How closely to follow the prompt
        controlnet_conditioning_scale: Weight of ControlNet conditioning
        height: Output image height
        width: Output image width
        seed: Random seed for reproducibility
        use_pose_image_directly: If True, skip pose extraction
        negative_prompt: What to avoid in generation
        low_vram_mode: Enable aggressive memory optimizations for P100 (16GB)
    """
    
    print("=" * 60)
    print("ControlNet Inference with OpenPose")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Input image: {input_image}")
    print(f"Model: {model_path}")
    print(f"ControlNet: {controlnet_model}")
    print(f"Output directory: {output_dir}")
    print(f"Number of images: {num_images}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"ControlNet scale: {controlnet_conditioning_scale}")
    print(f"Resolution: {width}x{height}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Low VRAM mode: {low_vram_mode}")
    print("=" * 60)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"🖥️  Using device: {device}")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"📊 GPU Memory: {gpu_mem:.1f} GB")
    print()
    
    # =========================================
    # STEP 1: Extract pose FIRST (if needed)
    # This runs the OpenPose detector separately to free GPU memory before loading SD
    # =========================================
    if use_pose_image_directly:
        print("📷 Using input as pose image directly")
        pose_image = load_image(input_image)
    else:
        print("🕺 Extracting pose from input image...")
        print("   (Running OpenPose detector separately to save VRAM)")
        from controlnet_aux import OpenposeDetector
        
        # Load OpenPose detector
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        if device == "cuda":
            openpose = openpose.to(device)
        
        pose_image = extract_pose(input_image, openpose)
        
        # Save extracted pose for reference
        pose_save_path = os.path.join(output_dir, "extracted_pose.png")
        pose_image.save(pose_save_path)
        print(f"💾 Saved extracted pose to: {pose_save_path}")
        
        # FREE GPU MEMORY: Delete OpenPose detector before loading SD pipeline
        print("🧹 Clearing OpenPose from GPU memory...")
        del openpose
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        print("✅ GPU memory cleared!")
    
    # Resize pose image to target size
    pose_image = pose_image.resize((width, height))
    print()
    
    # =========================================
    # STEP 2: Load ControlNet and SD Pipeline
    # =========================================
    print(f"⏳ Loading ControlNet: {controlnet_model}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=dtype,
    )
    print("✅ ControlNet loaded!")
    
    # Load pipeline
    print(f"⏳ Loading Stable Diffusion pipeline: {model_path}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    
    # Use faster scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # =========================================
    # STEP 3: Apply memory optimizations
    # =========================================
    print("🔧 Applying memory optimizations...")
    
    if low_vram_mode and device == "cuda":
        # AGGRESSIVE MEMORY SAVING for P100 (16GB)
        # This combination should reduce VRAM from ~14GB to ~8GB
        print("   ⚡ LOW VRAM MODE ENABLED")
        
        # 1. Enable model CPU offload (moves entire models, faster than sequential)
        pipe.enable_model_cpu_offload()
        print("   ✅ Model CPU offload enabled")
        
        # 2. Enable attention slicing (reduces memory during attention computation)
        pipe.enable_attention_slicing("max")
        print("   ✅ Attention slicing enabled (max)")
        
        # 3. Enable VAE slicing (reduces memory during VAE decode)
        pipe.enable_vae_slicing()
        print("   ✅ VAE slicing enabled")
        
        # 4. Enable VAE tiling (for processing large images in tiles)
        pipe.enable_vae_tiling()
        print("   ✅ VAE tiling enabled")
        
        # 5. Try xformers on top of other optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✅ xformers memory efficient attention enabled")
        except Exception:
            pass  # Already using attention slicing as fallback
            
    elif device == "cuda":
        # Standard optimizations
        pipe = pipe.to(device)
        
        # Try xformers first, fall back to attention slicing
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✅ xformers memory efficient attention enabled")
        except Exception as e:
            print(f"   ⚠️  xformers not available: {e}")
            print("   ✅ Using attention slicing instead")
            pipe.enable_attention_slicing("auto")
        
        # Enable VAE slicing for lower memory during decode
        pipe.enable_vae_slicing()
        print("   ✅ VAE slicing enabled")
    else:
        pipe = pipe.to(device)
    
    print("✅ Pipeline ready!")
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
        
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pose_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
            
            # Save image
            filename = f"controlnet_output_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            all_images.append((prompt, image, filepath))
            print(f"✅ Saved: {filepath}")
            
        except Exception as e:
            print(f"❌ Error generating image {i + 1}: {e}")
            continue
    
    print()
    print("=" * 60)
    print(f"✅ Generation Complete!")
    print("=" * 60)
    print(f"Generated {len(all_images)} images")
    print(f"Saved to: {output_dir}")
    print()
    
    # Create gallery HTML
    try:
        html_path = os.path.join(output_dir, "gallery.html")
        with open(html_path, "w") as f:
            f.write("<html><head><title>ControlNet Generated Images</title>")
            f.write("<style>body{font-family:Arial;margin:20px;background:#1a1a2e;color:#eee;}")
            f.write(".container{display:flex;flex-wrap:wrap;gap:20px;}")
            f.write(".image-card{background:#16213e;padding:20px;border-radius:12px;max-width:540px;}")
            f.write("img{max-width:512px;border-radius:8px;}")
            f.write("h3{color:#e94560;}h1{color:#0f3460;}</style></head><body>")
            f.write("<h1>🎨 ControlNet + OpenPose Generated Images</h1>")
            f.write(f"<p>Prompt: <em>{prompt}</em></p>")
            f.write("<div class='container'>")
            
            # Add pose image
            f.write("<div class='image-card'>")
            f.write("<h3>Input Pose</h3>")
            if not use_pose_image_directly:
                f.write("<img src='extracted_pose.png' alt='Extracted Pose'>")
            else:
                f.write(f"<img src='{os.path.basename(input_image)}' alt='Input Pose'>")
            f.write("</div>")
            
            for prompt_text, _, filepath in all_images:
                filename = os.path.basename(filepath)
                f.write("<div class='image-card'>")
                f.write(f"<h3>Generated</h3>")
                f.write(f"<img src='{filename}' alt='{prompt_text}'>")
                f.write("</div>")
            
            f.write("</div></body></html>")
        
        print(f"📄 Gallery created: {html_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create gallery HTML: {e}")
    
    return all_images


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using ControlNet with OpenPose conditioning"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image (for pose extraction) or pose image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to SD model (can be DreamBooth-finetuned model)",
    )
    parser.add_argument(
        "--controlnet_model",
        type=str,
        default="lllyasviel/control_v11p_sd15_openpose",
        help="ControlNet model for pose conditioning",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale (0.0-2.0)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_pose_directly",
        action="store_true",
        help="Use input image as pose image directly (skip pose extraction)",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="lowres, bad anatomy, worst quality, low quality, deformed, ugly",
        help="Negative prompt to avoid unwanted features",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Enable low VRAM mode for P100 (16GB) - uses CPU offloading",
    )
    
    args = parser.parse_args()
    print("START")
    generate_images(
        prompt=args.prompt,
        input_image=args.input_image,
        output_dir=args.output_dir,
        model_path=args.model_path,
        controlnet_model=args.controlnet_model,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        use_pose_image_directly=args.use_pose_directly,
        negative_prompt=args.negative_prompt,
        low_vram_mode=args.low_vram,
    )


if __name__ == "__main__":
    main()
