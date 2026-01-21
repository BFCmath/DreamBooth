#!/usr/bin/env python3
"""
Prepare dataset for Stage 1: Appearance Control Pretraining

This script prepares images for Stage 1 training which does NOT require
pose/conditioning images (pure text-to-image identity learning).

Features:
- Resize images to target resolution (maintaining aspect ratio + center crop)
- Optional face detection to filter images with clear faces
- Optional center cropping for images with subject in center
- Creates prompts.txt with instance prompt

Usage:
    # Basic usage (just organize images)
    python prepare_dataset_stage1.py --input_dir ./raw_images --output_dir ./data_stage1

    # With resize to 512x512
    python prepare_dataset_stage1.py --input_dir ./raw_images --output_dir ./data_stage1 --resolution 512
    
    # With center cropping (keep 60% center)
    python prepare_dataset_stage1.py --input_dir ./raw_images --output_dir ./data_stage1 --center_crop 0.6
    
This creates:
    data_stage1/
    ├── instance_images/   (processed images)
    └── prompts.txt        (instance prompts)
"""

import argparse
import os
from pathlib import Path
import shutil

from PIL import Image
from tqdm import tqdm


def resize_and_crop(image: Image.Image, resolution: int) -> Image.Image:
    """
    Resize image to resolution maintaining aspect ratio, then center crop.
    
    This ensures the output is exactly resolution x resolution while
    preserving as much of the original as possible.
    """
    width, height = image.size
    
    # Resize so the smaller dimension equals resolution
    if width < height:
        new_width = resolution
        new_height = int(height * resolution / width)
    else:
        new_height = resolution
        new_width = int(width * resolution / height)
    
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Center crop to resolution x resolution
    left = (new_width - resolution) // 2
    top = (new_height - resolution) // 2
    right = left + resolution
    bottom = top + resolution
    
    return image.crop((left, top, right, bottom))


def center_crop_horizontal(image: Image.Image, keep_ratio: float = 0.6) -> Image.Image:
    """Crop image to keep center portion horizontally (for subjects in center)."""
    width, height = image.size
    
    cut_ratio = (1 - keep_ratio) / 2
    left = int(width * cut_ratio)
    right = int(width * (1 - cut_ratio))
    
    return image.crop((left, 0, right, height))


def has_face(image: Image.Image, detector) -> bool:
    """Check if image contains a detectable face."""
    try:
        import numpy as np
        img_array = np.array(image)
        faces = detector(img_array)
        return len(faces) > 0
    except Exception:
        return True  # If detection fails, assume face is present


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Stage 1: Appearance Control Pretraining"
    )
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory with input images")
    parser.add_argument("--output_dir", type=str, default="./data_stage1", 
                        help="Output directory for dataset")
    parser.add_argument("--instance_prompt", type=str, default="a sks humanoid robot", 
                        help="Instance prompt for training")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Resize images to this resolution (e.g., 512)")
    parser.add_argument("--center_crop", type=float, default=None,
                        help="Crop to keep center portion (e.g., 0.6 keeps 60%)")
    parser.add_argument("--filter_faces", action="store_true",
                        help="Only keep images with detectable faces (requires face_recognition)")
    parser.add_argument("--copy_only", action="store_true",
                        help="Just copy files without any processing")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to include")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    instance_dir = output_dir / "instance_images"
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        return
    
    # Limit images if specified
    if args.max_images and len(images) > args.max_images:
        images = images[:args.max_images]
    
    print("=" * 60)
    print("🎨 Stage 1 Dataset Preparation")
    print("=" * 60)
    print(f"Input: {input_dir} ({len(images)} images)")
    print(f"Output: {output_dir}")
    print()
    print("Processing options:")
    if args.copy_only:
        print("  - Copy only (no processing)")
    else:
        if args.resolution:
            print(f"  - Resize to: {args.resolution}x{args.resolution}")
        if args.center_crop:
            print(f"  - Center crop: keep {args.center_crop*100:.0f}%")
        if args.filter_faces:
            print("  - Face filtering: enabled")
    print("=" * 60)
    print()
    
    # Load face detector if needed
    face_detector = None
    if args.filter_faces:
        try:
            import face_recognition
            print("⏳ Face detection enabled...")
            face_detector = face_recognition.face_locations
        except ImportError:
            print("⚠️ face_recognition not installed, skipping face filter")
            print("   Install with: pip install face_recognition")
            args.filter_faces = False
    
    # Process images
    processed = 0
    skipped = 0
    
    print("📸 Processing images...")
    for img_path in tqdm(images, desc="Processing"):
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Optional face filtering
            if args.filter_faces and face_detector:
                if not has_face(image, face_detector):
                    skipped += 1
                    continue
            
            # Apply processing (unless copy_only)
            if not args.copy_only:
                # Center crop first if specified
                if args.center_crop:
                    image = center_crop_horizontal(image, args.center_crop)
                
                # Resize if specified
                if args.resolution:
                    image = resize_and_crop(image, args.resolution)
            
            # Save with sequential naming
            processed += 1
            new_name = f"{processed:03d}.png"
            output_path = instance_dir / new_name
            
            if args.copy_only:
                shutil.copy(img_path, output_path)
            else:
                image.save(output_path, "PNG")
                
        except Exception as e:
            print(f"\n⚠️ Error processing {img_path.name}: {e}")
            skipped += 1
    
    # Create prompts file
    prompts_path = output_dir / "prompts.txt"
    with open(prompts_path, "w") as f:
        for _ in range(processed):
            f.write(f"{args.instance_prompt}\n")
    
    print()
    print("=" * 60)
    print("✅ Stage 1 Dataset Preparation Complete!")
    print("=" * 60)
    print()
    print(f"📊 Results:")
    print(f"   - Processed: {processed} images")
    if skipped > 0:
        print(f"   - Skipped: {skipped} images")
    print()
    print(f"📁 Dataset structure:")
    print(f"   {output_dir}/")
    print(f"   ├── instance_images/   ({processed} images)")
    print(f"   └── prompts.txt        ('{args.instance_prompt}')")
    print()
    print("🚀 Next step - Run Stage 1 training:")
    print(f"   python dreambooth_controlnet_stage1.py \\")
    print(f"       --data_dir {output_dir} \\")
    print(f"       --output_dir ./output/stage1 \\")
    print(f"       --instance_prompt \"{args.instance_prompt}\" \\")
    print(f"       --with_prior_preservation")
    print()
    print("💡 Recommended: 5-7 images for good identity learning")
    if processed < 5:
        print(f"   ⚠️ You have only {processed} images - consider adding more!")
    elif processed > 10:
        print(f"   ⚠️ You have {processed} images - consider selecting best 5-7")


if __name__ == "__main__":
    main()
