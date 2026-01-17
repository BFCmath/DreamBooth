#!/usr/bin/env python3
"""
Download AP10K Poses ControlNet Dataset from HuggingFace
https://huggingface.co/datasets/JFoz/AP10K-poses-controlnet-dataset

This is a ready-to-use dataset for ControlNet training with:
- 7,023 image pairs
- Ground truth images + OpenPose conditioning images
- Text prompts
"""

import argparse
import os
from pathlib import Path


def download_ap10k_dataset(
    output_dir: str = "./data",
    max_samples: int = None,
    split: str = "train",
):
    """
    Download AP10K poses dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        max_samples: Maximum samples to download (None = all ~7000)
        split: Dataset split to use
    """
    
    # Install datasets if needed
    try:
        from datasets import load_dataset
    except ImportError:
        print("⏳ Installing datasets library...")
        os.system("pip install -q datasets")
        from datasets import load_dataset
    
    try:
        from PIL import Image
    except ImportError:
        os.system("pip install -q Pillow")
        from PIL import Image
    
    print("=" * 60)
    print("AP10K Poses ControlNet Dataset Downloader")
    print("=" * 60)
    print(f"Dataset: JFoz/AP10K-poses-controlnet-dataset")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {max_samples if max_samples else 'All (~7000)'}")
    print("=" * 60)
    print()
    
    # Create directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    conditioning_dir = output_path / "conditioning"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    conditioning_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from HuggingFace
    print("⏳ Loading dataset from HuggingFace...")
    print("   (This may take a few minutes for first download)")
    
    dataset = load_dataset(
        "JFoz/AP10K-poses-controlnet-dataset",
        split=split,
    )
    
    total_samples = len(dataset)
    print(f"✅ Loaded dataset: {total_samples} samples")
    
    # Limit samples if specified
    if max_samples and max_samples < total_samples:
        dataset = dataset.select(range(max_samples))
        print(f"   Using first {max_samples} samples")
    
    # Check column names
    print(f"📊 Dataset columns: {dataset.column_names}")
    
    # Determine column names (different datasets use different names)
    image_col = None
    cond_col = None
    text_col = None
    
    for col in dataset.column_names:
        col_lower = col.lower()
        if "image" in col_lower and "pose" not in col_lower and "condition" not in col_lower:
            image_col = col
        elif "pose" in col_lower or "condition" in col_lower or "control" in col_lower:
            cond_col = col
        elif "text" in col_lower or "prompt" in col_lower or "caption" in col_lower:
            text_col = col
    
    # Fallback: use first two image columns
    if not image_col or not cond_col:
        img_cols = [c for c in dataset.column_names if dataset.features[c].dtype == "image" or "image" in c.lower()]
        if len(img_cols) >= 2:
            image_col = img_cols[0]
            cond_col = img_cols[1]
        elif len(img_cols) == 1:
            # Single image column - might be conditioning only
            cond_col = img_cols[0]
    
    print(f"   Image column: {image_col}")
    print(f"   Conditioning column: {cond_col}")
    print(f"   Text column: {text_col}")
    print()
    
    # Process and save images
    print("⏳ Processing and saving images...")
    prompts = []
    saved_count = 0
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            print(f"   Processing {i}/{len(dataset)}...")
        
        try:
            # Get images
            if image_col and image_col in sample:
                img = sample[image_col]
                if isinstance(img, Image.Image):
                    img.save(images_dir / f"{i:05d}.png")
                elif hasattr(img, "convert"):
                    img.convert("RGB").save(images_dir / f"{i:05d}.png")
            else:
                # Create placeholder if no ground truth image
                placeholder = Image.new("RGB", (512, 512), (128, 128, 128))
                placeholder.save(images_dir / f"{i:05d}.png")
            
            if cond_col and cond_col in sample:
                cond = sample[cond_col]
                if isinstance(cond, Image.Image):
                    cond.save(conditioning_dir / f"{i:05d}.png")
                elif hasattr(cond, "convert"):
                    cond.convert("RGB").save(conditioning_dir / f"{i:05d}.png")
            
            # Get text prompt
            if text_col and text_col in sample:
                prompt = sample[text_col]
            else:
                prompt = "a photo of an animal in a natural pose, high quality"
            
            prompts.append(prompt)
            saved_count += 1
            
        except Exception as e:
            print(f"   ⚠️  Error processing sample {i}: {e}")
            continue
    
    # Save prompts
    prompts_file = output_path / "prompts.txt"
    with open(prompts_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(str(p) + "\n")
    
    print()
    print("=" * 60)
    print("✅ Dataset Download Complete!")
    print("=" * 60)
    print()
    print(f"📁 Dataset location: {output_path}")
    print(f"   - Images: {images_dir} ({saved_count} files)")
    print(f"   - Conditioning: {conditioning_dir} ({saved_count} files)")
    print(f"   - Prompts: {prompts_file}")
    print()
    print("🚀 Next steps:")
    print("   bash run_train.sh")
    print()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download AP10K ControlNet dataset")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to download")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    
    args = parser.parse_args()
    
    download_ap10k_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split=args.split,
    )


if __name__ == "__main__":
    main()
