#!/usr/bin/env python3
"""
ControlNet Inference Script with Multiple Conditioning Types
Run on Kaggle P100 to generate conditioned images using Stable Diffusion 1.5

Supports:
- Canny edge detection (for lllyasviel/control_v11p_sd15_canny)
- HED edge detection (for lllyasviel/control_v11p_sd15_hed)
- OpenPose skeleton detection (for lllyasviel/control_v11p_sd15_openpose)
- Pre-made conditioning images as direct input
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import gc


def extract_canny(image, low_threshold: int = 100, high_threshold: int = 200):
    """Extract Canny edges from an image."""
    import cv2
    import numpy as np
    
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def extract_hed(image, detector):
    """Extract HED soft edges from an image."""
    return detector(image)


def extract_pose(image, detector):
    """Extract OpenPose skeleton from an image."""
    return detector(image)


def generate_images(
    prompt: str,
    input_image: str,
    output_dir: str = "./output_images",
    model_path: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model: str = "lllyasviel/control_v11p_sd15_canny",
    base_controlnet_model: str = None,  # Base ControlNet model for LoRA loading
    detector_type: str = "auto",  # auto, canny, hed, openpose, none
    num_images: int = 1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    height: int = 512,
    width: int = 512,
    seed: int = None,
    negative_prompt: str = "lowres, bad anatomy, worst quality, low quality, deformed, ugly",
    low_vram_mode: bool = False,
    unet_lora_path: str = None,  # Path to LoRA weights for UNet
    text_encoder_lora_path: str = None,  # Path to LoRA weights for text encoder
    token_embedding_path: str = None,  # Path to token embedding (from --train_token_embedding)
):
    """
    Generate images using ControlNet with various conditioning types.
    
    Args:
        prompt: Text prompt for image generation
        input_image: Path to input image (for conditioning extraction) or conditioning image
        output_dir: Directory to save generated images
        model_path: Path to SD 1.5 model (can be DreamBooth-finetuned)
        controlnet_model: ControlNet model (canny, hed, openpose, etc.)
        detector_type: "auto" (detect from model), "canny", "hed", "openpose", or "none" (use input directly)
        num_images: Number of images to generate
        num_inference_steps: Denoising steps (higher = better quality)
        guidance_scale: How closely to follow the prompt
        controlnet_conditioning_scale: Weight of ControlNet conditioning
        height: Output image height
        width: Output image width
        seed: Random seed for reproducibility
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
    # STEP 1: Extract conditioning FIRST (if needed)
    # This runs the detector separately to free GPU memory before loading SD
    # =========================================
    
    # Auto-detect detector type from ControlNet model name
    if detector_type == "auto":
        if "canny" in controlnet_model.lower():
            detected_type = "canny"
        elif "hed" in controlnet_model.lower():
            detected_type = "hed"
        elif "openpose" in controlnet_model.lower() or "pose" in controlnet_model.lower():
            detected_type = "openpose"
        else:
            detected_type = "none"  # Unknown model, use input directly
            print(f"⚠️  Unknown ControlNet type, using input image directly")
        print(f"🔍 Auto-detected detector: {detected_type}")
    else:
        detected_type = detector_type
    
    if detected_type == "none":
        print("📷 Using input as conditioning image directly")
        conditioning_image = load_image(input_image)
    elif detected_type == "canny":
        print("🔲 Extracting Canny edges from input image...")
        input_img = load_image(input_image)
        conditioning_image = extract_canny(input_img)
        
        # Save extracted conditioning for reference
        cond_save_path = os.path.join(output_dir, "extracted_canny.png")
        conditioning_image.save(cond_save_path)
        print(f"💾 Saved extracted Canny to: {cond_save_path}")
    elif detected_type == "hed":
        print("🔲 Extracting HED edges from input image...")
        from controlnet_aux import HEDdetector
        
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        if device == "cuda":
            hed = hed.to(device)
        
        input_img = load_image(input_image)
        conditioning_image = extract_hed(input_img, hed)
        
        cond_save_path = os.path.join(output_dir, "extracted_hed.png")
        conditioning_image.save(cond_save_path)
        print(f"💾 Saved extracted HED to: {cond_save_path}")
        
        # Free memory
        del hed
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    elif detected_type == "openpose":
        print("🕺 Extracting OpenPose from input image...")
        from controlnet_aux import OpenposeDetector
        
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        if device == "cuda":
            openpose = openpose.to(device)
        
        input_img = load_image(input_image)
        conditioning_image = extract_pose(input_img, openpose)
        
        cond_save_path = os.path.join(output_dir, "extracted_pose.png")
        conditioning_image.save(cond_save_path)
        print(f"💾 Saved extracted pose to: {cond_save_path}")
        
        # Free memory
        del openpose
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    else:
        raise ValueError(f"Unknown detector type: {detected_type}")
    
    print("✅ Conditioning ready!")
    
    # Resize conditioning image to target size
    conditioning_image = conditioning_image.resize((width, height))
    print()
    
    # =========================================
    # STEP 2: Load ControlNet and SD Pipeline
    # =========================================
    print(f"⏳ Loading ControlNet: {controlnet_model}")
    
    # Check if controlnet_model is a full model or LoRA weights
    controlnet_path = Path(controlnet_model)
    config_exists = (controlnet_path / "config.json").exists() if controlnet_path.exists() else False
    
    if not config_exists and controlnet_path.exists():
        # This is likely a LoRA adapter directory - need to load base model first
        # Try to infer the base model from the adapter
        adapter_config_path = controlnet_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            print(f"   🔍 Detected LoRA adapter at {controlnet_model}")
            # Try to extract base model from adapter config
            import json
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            
            # Use explicit base model if provided, otherwise use default
            base_controlnet = base_controlnet_model or "lllyasviel/control_v11p_sd15_openpose"
            print(f"   ⏳ Loading base ControlNet: {base_controlnet}")
            
            from peft import PeftModel
            controlnet = ControlNetModel.from_pretrained(
                base_controlnet,
                torch_dtype=dtype,
            )
            print(f"   ⏳ Applying LoRA weights from: {controlnet_model}")
            controlnet_peft = PeftModel.from_pretrained(controlnet, str(controlnet_path))
            # Merge LoRA weights into base model for pipeline compatibility
            print("   🔄 Merging LoRA weights into base model...")
            controlnet = controlnet_peft.merge_and_unload()
            print("   ✅ ControlNet LoRA weights merged!")
        else:
            raise ValueError(
                f"ControlNet path {controlnet_model} exists but has no config.json or adapter_config.json. "
                "Cannot determine if this is a full model or LoRA adapter."
            )
    else:
        # Standard loading (either HuggingFace model or full saved model)
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
    
    # Load LoRA weights if provided
    if unet_lora_path:
        print(f"⏳ Loading UNet LoRA weights: {unet_lora_path}")
        try:
            from peft import PeftModel
            pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path)
            print("✅ UNet LoRA weights loaded!")
        except Exception as e:
            print(f"❌ Failed to load UNet LoRA weights: {e}")
            raise
    
    # Load text encoder LoRA if provided
    if text_encoder_lora_path:
        print(f"⏳ Loading text encoder LoRA weights: {text_encoder_lora_path}")
        try:
            from peft import PeftModel
            pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_lora_path)
            print("✅ Text encoder LoRA weights loaded!")
        except Exception as e:
            print(f"❌ Failed to load text encoder LoRA weights: {e}")
            raise
    
    # Load token embedding if provided (Custom Diffusion style)
    if token_embedding_path:
        print(f"⏳ Loading token embedding: {token_embedding_path}")
        try:
            token_data = torch.load(token_embedding_path, map_location="cpu")
            token_id = token_data["token_id"]
            embedding = token_data["embedding"]
            
            # Update the token embedding in the text encoder
            pipe.text_encoder.text_model.embeddings.token_embedding.weight.data[token_id] = embedding.to(
                pipe.text_encoder.text_model.embeddings.token_embedding.weight.device
            ).to(pipe.text_encoder.text_model.embeddings.token_embedding.weight.dtype)
            
            print(f"✅ Token embedding loaded for '{token_data['token']}' (ID: {token_id})")
        except Exception as e:
            print(f"❌ Failed to load token embedding: {e}")
            raise
    
    # =========================================
    # VRAM Debug Helper
    # =========================================
    def print_vram(label=""):
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   📊 VRAM [{label}]: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    # =========================================
    # STEP 3: Apply memory optimizations
    # =========================================
    print("🔧 Applying memory optimizations...")
    print_vram("Before optimizations")
    
    if low_vram_mode and device == "cuda":
        # BALANCED MODE: Use GPU but with memory optimizations
        print("   ⚡ LOW VRAM MODE: GPU with smart memory management")
        
        # Move to GPU first
        pipe = pipe.to(device)
        print_vram("After moving to GPU")
        
        # 1. Enable attention slicing (reduces memory during attention)
        pipe.enable_attention_slicing("auto")
        print("   ✅ Attention slicing enabled")
        
        # 2. Enable VAE slicing (reduces memory during VAE decode)
        pipe.enable_vae_slicing()
        print("   ✅ VAE slicing enabled")
        
        # 3. Try xformers (best memory efficiency if available)
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
            print("   ✅ xformers memory efficient attention enabled")
        except Exception as e:
            print(f"   ⚠️  xformers not available: {e}")
        
        print_vram("After optimizations")
            
    elif device == "cuda":
        # Standard optimizations
        pipe = pipe.to(device)
        print_vram("After moving to GPU")
        
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
        print_vram("After optimizations")
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
        print_vram("Before generation")
        
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=conditioning_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
            
            print_vram("After generation")
            
            # Save image
            filename = f"controlnet_output_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            all_images.append((prompt, image, filepath))
            print(f"✅ Saved: {filepath}")
            
            # Clear VRAM after each image
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                print_vram("After cache clear")
            
        except Exception as e:
            print(f"❌ Error generating image {i + 1}: {e}")
            # Clear cache on error too
            if device == "cuda":
                torch.cuda.empty_cache()
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
            
            # Add conditioning image
            f.write("<div class='image-card'>")
            f.write("<h3>Conditioning</h3>")
            if detected_type == "canny":
                f.write("<img src='extracted_canny.png' alt='Canny Edges'>")
            elif detected_type == "hed":
                f.write("<img src='extracted_hed.png' alt='HED Edges'>")
            elif detected_type == "openpose":
                f.write("<img src='extracted_pose.png' alt='OpenPose'>")
            else:
                f.write(f"<img src='{os.path.basename(input_image)}' alt='Input'>")
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
        help="ControlNet model for pose conditioning (can be full model or LoRA weights path)",
    )
    parser.add_argument(
        "--base_controlnet_model",
        type=str,
        default=None,
        help="Base ControlNet model to use when loading LoRA weights (e.g., lllyasviel/control_v11p_sd15_openpose)",
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
        "--detector",
        type=str,
        default="auto",
        choices=["auto", "canny", "hed", "openpose", "none"],
        help="Detector type: auto (detect from model), canny, hed, openpose, or none (use input directly)",
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
    parser.add_argument(
        "--unet_lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights for UNet (from DreamBooth LoRA training)",
    )
    parser.add_argument(
        "--text_encoder_lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights for text encoder (from DreamBooth LoRA training with --train_text_encoder)",
    )
    parser.add_argument(
        "--token_embedding_path",
        type=str,
        default=None,
        help="Path to token embedding file (from --train_token_embedding, e.g., token_embedding.pt)",
    )
    
    args = parser.parse_args()
    print("START")
    generate_images(
        prompt=args.prompt,
        input_image=args.input_image,
        output_dir=args.output_dir,
        model_path=args.model_path,
        controlnet_model=args.controlnet_model,
        base_controlnet_model=args.base_controlnet_model,
        detector_type=args.detector,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        low_vram_mode=args.low_vram,
        unet_lora_path=args.unet_lora_path,
        text_encoder_lora_path=args.text_encoder_lora_path,
        token_embedding_path=args.token_embedding_path,
    )


if __name__ == "__main__":
    main()
