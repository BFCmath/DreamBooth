#!/usr/bin/env python3
"""
Inference with DreamBooth LoRA + ControlNet

Combines:
- LoRA: Provides identity ("sks cat" = your specific cat)
- ControlNet: Provides pose/structure control

Usage:
    python infer_lora_controlnet.py \
        --lora_path ./output/dreambooth-lora \
        --controlnet lllyasviel/control_v11p_sd15_canny \
        --prompt "a photo of sks cat sitting on a couch" \
        --control_image canny_pose.png \
        --output output.png
"""

import argparse
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)


def extract_canny(image_path: str, low: int = 100, high: int = 200):
    """Extract Canny edges from an image."""
    import cv2
    import numpy as np
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def main():
    parser = argparse.ArgumentParser(description="LoRA + ControlNet Inference")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--controlnet", type=str, default="lllyasviel/control_v11p_sd15_canny",
                       help="ControlNet model ID")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, bad anatomy")
    parser.add_argument("--control_image", type=str, required=True,
                       help="Control image (will extract Canny if needed)")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto_canny", action="store_true",
                       help="Auto-extract Canny edges from control_image")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("LoRA + ControlNet Inference")
    print("=" * 60)
    print(f"LoRA: {args.lora_path}")
    print(f"ControlNet: {args.controlnet}")
    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    
    # Load ControlNet
    print("\n⏳ Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
    
    # Load pipeline with ControlNet
    print("⏳ Loading pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights
    print(f"⏳ Loading LoRA from {args.lora_path}...")
    pipe.load_lora_weights(args.lora_path)
    
    pipe.to(device)
    print("✅ Pipeline ready")
    
    # Load control image
    print(f"\n⏳ Loading control image: {args.control_image}")
    if args.auto_canny or "canny" in args.controlnet.lower():
        print("   Extracting Canny edges...")
        control_image = extract_canny(args.control_image)
    else:
        control_image = Image.open(args.control_image).convert("RGB")
    
    # Resize to standard SD resolution
    control_image = control_image.resize((512, 512))
    
    # Set seed
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Generate
    print(f"\n🎨 Generating image...")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=control_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        generator=generator,
    ).images[0]
    
    # Save
    result.save(args.output)
    print(f"\n✅ Saved: {args.output}")
    
    # Also save control image for reference
    control_output = args.output.replace(".png", "_control.png")
    control_image.save(control_output)
    print(f"✅ Control image: {control_output}")


if __name__ == "__main__":
    main()
