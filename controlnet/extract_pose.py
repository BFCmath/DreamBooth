#!/usr/bin/env python3
"""
Extract OpenPose conditioning from local images.

Usage:
    python extract_pose.py --input_dir ./dataset --output_dir ./data
    
This creates:
    data/
    ├── instance_images/   (copies of your images)
    └── conditioning/      (OpenPose extracted)
"""

import argparse
import os
from pathlib import Path
import shutil

import torch
from PIL import Image
from tqdm import tqdm


def extract_pose(image_path: str, detector, output_path: str):
    """Extract OpenPose from a single image."""
    image = Image.open(image_path).convert("RGB")
    pose_image = detector(image)
    pose_image.save(output_path)
    return pose_image


def main():
    parser = argparse.ArgumentParser(description="Extract OpenPose from local images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for dataset")
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person", help="Instance prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    instance_dir = output_dir / "instance_images"
    conditioning_dir = output_dir / "conditioning"
    instance_dir.mkdir(parents=True, exist_ok=True)
    conditioning_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        return
    
    print("=" * 60)
    print("OpenPose Extraction for ControlNet Training")
    print("=" * 60)
    print(f"Input: {input_dir} ({len(images)} images)")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    print()
    
    # Load DWPose detector (more accurate than OpenPose)
    print("⏳ Loading DWPose detector...")
    from controlnet_aux import DWposeDetector
    detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
    if args.device == "cuda":
        detector = detector.to(args.device)
    print("✅ DWPose detector loaded")
    print()
    
    # Process each image
    print("🕺 Extracting poses...")
    for i, img_path in enumerate(tqdm(images, desc="Processing")):
        # Copy original to instance_images (rename to sequential)
        new_name = f"{i+1:03d}{img_path.suffix}"
        instance_path = instance_dir / new_name
        shutil.copy(img_path, instance_path)
        
        # Extract pose
        cond_path = conditioning_dir / new_name.replace(img_path.suffix, ".png")
        extract_pose(str(img_path), detector, str(cond_path))
    
    # Create prompts file
    prompts_path = output_dir / "prompts.txt"
    with open(prompts_path, "w") as f:
        for _ in images:
            f.write(f"{args.instance_prompt}\n")
    
    print()
    print("=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print()
    print("Dataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── instance_images/   ({len(images)} images)")
    print(f"  ├── conditioning/      ({len(images)} OpenPose images)")
    print(f"  └── prompts.txt        ('{args.instance_prompt}')")
    print()
    print("Next steps:")
    print("  1. Test inference:")
    print(f"     python infer_controlnet.py --prompt \"{args.instance_prompt}\" \\")
    print(f"         --input_image {conditioning_dir}/001.png \\")
    print("         --controlnet_model lllyasviel/control_v11p_sd15_openpose \\")
    print("         --detector none")
    print()
    print("  2. Train ControlNet:")
    print(f"     INSTANCE_PROMPT=\"{args.instance_prompt}\" \\")
    print("     CLASS_PROMPT=\"a photo of person\" \\")
    print("     CONTROLNET_MODEL=\"lllyasviel/control_v11p_sd15_openpose\" \\")
    print("     bash run_dreambooth_controlnet.sh")


if __name__ == "__main__":
    main()
