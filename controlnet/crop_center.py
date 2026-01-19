#!/usr/bin/env python3
"""
Crop images to keep center 60% (cut 20% from left and right).

Usage:
    python crop_center.py --input_dir ./unitree_dataset --output_dir ./unitree_cropped
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def crop_center(image: Image.Image, keep_ratio: float = 0.5) -> Image.Image:
    """Crop image to keep center portion horizontally."""
    width, height = image.size
    
    # Calculate crop boundaries
    cut_ratio = (1 - keep_ratio) / 2  # 20% from each side
    left = int(width * cut_ratio)
    right = int(width * (1 - cut_ratio))
    
    # Crop: (left, top, right, bottom)
    return image.crop((left, 0, right, height))


def main():
    parser = argparse.ArgumentParser(description="Crop images to keep center 60%")
    parser.add_argument("--input_dir", type=str, default="./unitree_dataset", 
                        help="Input directory with images")
    parser.add_argument("--output_dir", type=str, default="./unitree_cropped",
                        help="Output directory for cropped images")
    parser.add_argument("--keep_ratio", type=float, default=0.5,
                        help="Ratio of width to keep (0.5 = keep center 50%%)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📁 Input: {input_dir} ({len(images)} images)")
    print(f"📁 Output: {output_dir}")
    print(f"✂️  Keeping center {args.keep_ratio*100:.0f}% (cutting {(1-args.keep_ratio)/2*100:.0f}% from each side)")
    print()
    
    for img_path in tqdm(images, desc="Cropping"):
        image = Image.open(img_path).convert("RGB")
        cropped = crop_center(image, args.keep_ratio)
        
        output_path = output_dir / img_path.name
        cropped.save(output_path)
    
    print(f"\n✅ Done! Cropped {len(images)} images to {output_dir}")


if __name__ == "__main__":
    main()
