#!/usr/bin/env python3
"""
Prepare DreamBooth Dataset for ControlNet Training

This script:
1. Downloads images from HuggingFace DreamBooth dataset
2. Extracts conditioning (Canny edge / OpenPose / HED) for each image
3. Saves to the required folder structure for dreambooth_controlnet.py

Usage:
    # For cat/dog/object datasets (uses Canny edge detection)
    python prepare_dreambooth_dataset.py --subset cat2 --detector canny

    # For person datasets (uses OpenPose)
    python prepare_dreambooth_dataset.py --subset person --detector openpose

Supported detectors:
    - canny: Canny edge detection (best for objects/animals)
    - hed: Holistically-Nested Edge Detection (soft edges)
    - openpose: Human pose detection (for person datasets only)
    - lineart: Line art extraction
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def download_dreambooth_dataset(subset: str, output_dir: str, max_images: int = None):
    """
    Download images from HuggingFace DreamBooth dataset.
    
    Available subsets: backpack, backpack_dog, bear_plushie, berry_bowl, can,
    candle, cat, cat2, clock, colorful_sneaker, dog, dog2, dog3, dog5, dog6,
    dog7, dog8, duck_toy, fancy_boot, grey_sloth_plushie, monster_toy,
    pink_sunglasses, poop_emoji, rc_car, red_cartoon, robot_toy, shiny_sneaker,
    teapot, vase, wolf_plushie
    """
    from datasets import load_dataset
    
    print(f"📥 Downloading DreamBooth dataset: {subset}")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("google/dreambooth", subset, split="train")
    
    output_path = Path(output_dir) / "instance_images"
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_images = len(dataset)
    if max_images:
        num_images = min(num_images, max_images)
    
    print(f"📊 Found {len(dataset)} images, downloading {num_images}")
    
    saved_paths = []
    for i in tqdm(range(num_images), desc="Downloading images"):
        image = dataset[i]["image"]
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save image
        image_path = output_path / f"{i+1:03d}.png"
        image.save(image_path)
        saved_paths.append(image_path)
    
    print(f"✅ Downloaded {len(saved_paths)} images to {output_path}")
    return saved_paths


def extract_canny_edges(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Extract Canny edges from an image."""
    import cv2
    import numpy as np
    
    # Convert to numpy
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to RGB (3 channels)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(edges_rgb)


def extract_hed_edges(image: Image.Image, detector) -> Image.Image:
    """Extract HED (Holistically-Nested Edge Detection) from an image."""
    return detector(image)


def extract_openpose(image: Image.Image, detector) -> Image.Image:
    """Extract OpenPose skeleton from an image."""
    return detector(image)


def extract_lineart(image: Image.Image, detector) -> Image.Image:
    """Extract line art from an image."""
    return detector(image)


def extract_conditioning(
    image_paths: list,
    output_dir: str,
    detector_type: str = "canny",
    device: str = "cuda",
):
    """
    Extract conditioning images using the specified detector.
    
    Args:
        image_paths: List of paths to instance images
        output_dir: Directory to save conditioning images
        detector_type: One of "canny", "hed", "openpose", "lineart"
        device: Device to run detector on
    """
    output_path = Path(output_dir) / "conditioning"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Extracting conditioning using: {detector_type}")
    
    # Initialize detector
    detector = None
    
    if detector_type == "canny":
        # Canny doesn't need a model, just use OpenCV
        print("   Using OpenCV Canny edge detection")
    
    elif detector_type == "hed":
        from controlnet_aux import HEDdetector
        detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        detector.to(device)
        print("   Loaded HED detector")
    
    elif detector_type == "openpose":
        from controlnet_aux import OpenposeDetector
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        print("   Loaded OpenPose detector")
        print("   ⚠️  Note: OpenPose is for HUMANS only, not animals!")
    
    elif detector_type == "lineart":
        from controlnet_aux import LineartDetector
        detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        detector.to(device)
        print("   Loaded Lineart detector")
    
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Extracting conditioning"):
        image = Image.open(image_path).convert("RGB")
        
        # Extract conditioning
        if detector_type == "canny":
            cond_image = extract_canny_edges(image)
        elif detector_type == "hed":
            cond_image = extract_hed_edges(image, detector)
        elif detector_type == "openpose":
            cond_image = extract_openpose(image, detector)
        elif detector_type == "lineart":
            cond_image = extract_lineart(image, detector)
        
        # Save with same filename
        cond_path = output_path / image_path.name
        cond_image.save(cond_path)
    
    print(f"✅ Saved {len(image_paths)} conditioning images to {output_path}")


def create_prompts_file(output_dir: str, instance_prompt: str, num_images: int):
    """Create prompts.txt with the same prompt for all images."""
    prompts_path = Path(output_dir) / "prompts.txt"
    
    with open(prompts_path, "w") as f:
        for _ in range(num_images):
            f.write(f"{instance_prompt}\n")
    
    print(f"✅ Created prompts.txt with {num_images} prompts: '{instance_prompt}'")


def main():
    parser = argparse.ArgumentParser(description="Prepare DreamBooth dataset for ControlNet training")
    
    parser.add_argument(
        "--subset", type=str, default="cat2",
        help="DreamBooth dataset subset (e.g., cat2, dog, dog2, person)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--detector", type=str, default="canny",
        choices=["canny", "hed", "openpose", "lineart"],
        help="Conditioning detector type"
    )
    parser.add_argument(
        "--instance_prompt", type=str, default=None,
        help="Instance prompt (default: 'a photo of sks {subset}')"
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Maximum number of images to download"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for detector inference"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip download, only extract conditioning from existing images"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DreamBooth Dataset Preparation for ControlNet")
    print("=" * 60)
    print(f"Subset: {args.subset}")
    print(f"Output: {args.output_dir}")
    print(f"Detector: {args.detector}")
    print(f"Device: {args.device}")
    print("=" * 60)
    print()
    
    # Step 1: Download images
    if args.skip_download:
        instance_dir = Path(args.output_dir) / "instance_images"
        image_paths = sorted(list(instance_dir.glob("*.png")) + list(instance_dir.glob("*.jpg")))
        print(f"📂 Using existing {len(image_paths)} images from {instance_dir}")
    else:
        image_paths = download_dreambooth_dataset(
            subset=args.subset,
            output_dir=args.output_dir,
            max_images=args.max_images,
        )
    
    print()
    
    # Step 2: Extract conditioning
    extract_conditioning(
        image_paths=image_paths,
        output_dir=args.output_dir,
        detector_type=args.detector,
        device=args.device,
    )
    
    print()
    
    # Step 3: Create prompts file
    instance_prompt = args.instance_prompt or f"a photo of sks {args.subset.replace('2', '').replace('3', '')}"
    create_prompts_file(
        output_dir=args.output_dir,
        instance_prompt=instance_prompt,
        num_images=len(image_paths),
    )
    
    print()
    print("=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print()
    print("Dataset structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── instance_images/   ({len(image_paths)} images)")
    print(f"  ├── conditioning/      ({len(image_paths)} {args.detector} images)")
    print(f"  └── prompts.txt        ('{instance_prompt}')")
    print()
    print("Next step: Run training with:")
    print(f"  INSTANCE_PROMPT='{instance_prompt}' \\")
    print(f"  CLASS_PROMPT='a photo of {args.subset.replace('2', '').replace('3', '')}' \\")
    print(f"  bash run_dreambooth_controlnet.sh")


if __name__ == "__main__":
    main()
