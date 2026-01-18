#!/usr/bin/env python3
"""
Extract Canny edges from images for ControlNet conditioning.

Usage:
    python extract_canny.py --input image.png --output canny.png
    python extract_canny.py --input_dir ./images --output_dir ./canny
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np


def extract_canny(
    image_path: str,
    output_path: str = None,
    low_threshold: int = 100,
    high_threshold: int = 200,
):
    """Extract Canny edges from a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Convert to grayscale and apply Canny
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to RGB (3 channels for ControlNet)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(edges_rgb)
    
    # Save if output path provided
    if output_path:
        result.save(output_path)
        print(f"✅ Saved: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract Canny edges from images")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output image path")
    parser.add_argument("--input_dir", type=str, help="Input directory (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Output directory (batch mode)")
    parser.add_argument("--low", type=int, default=100, help="Canny low threshold")
    parser.add_argument("--high", type=int, default=200, help="Canny high threshold")
    
    args = parser.parse_args()
    
    if args.input_dir:
        # Batch mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or "./canny_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
        
        print(f"Processing {len(images)} images...")
        for img_path in images:
            out_path = output_dir / img_path.name
            extract_canny(str(img_path), str(out_path), args.low, args.high)
        
        print(f"\n✅ Processed {len(images)} images to {output_dir}")
    
    elif args.input:
        # Single image mode
        output = args.output or args.input.replace(".", "_canny.")
        extract_canny(args.input, output, args.low, args.high)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
