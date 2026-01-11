#!/usr/bin/env python3
"""
Download example images from Hugging Face Hub
Useful for testing DreamBooth without your own images
"""

import argparse
from pathlib import Path


EXAMPLE_DATASETS = {
    "dog": {
        "repo": "diffusers/dog-example",
        "description": "5 images of a cute dog",
        "instance_prompt": "a photo of sks dog",
        "class_prompt": "a photo of dog",
    },
    "cat": {
        "repo": "diffusers/cat-example", 
        "description": "Images of a cat",
        "instance_prompt": "a photo of sks cat",
        "class_prompt": "a photo of cat",
    },
    "man": {
        "repo": "diffusers/man-example",
        "description": "Images of a person",
        "instance_prompt": "a photo of sks man",
        "class_prompt": "a photo of man",
    },
}


def download_example_dataset(dataset_name, output_dir="./instance_images"):
    """Download an example dataset from Hugging Face Hub."""
    
    if dataset_name not in EXAMPLE_DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"\nAvailable datasets:")
        for name, info in EXAMPLE_DATASETS.items():
            print(f"  - {name}: {info['description']}")
        return False
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("Install it with: pip install huggingface_hub")
        return False
    
    dataset_info = EXAMPLE_DATASETS[dataset_name]
    
    print(f"📥 Downloading {dataset_name} example images...")
    print(f"   Repository: {dataset_info['repo']}")
    print(f"   Description: {dataset_info['description']}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the dataset
        snapshot_download(
            dataset_info['repo'],
            local_dir=output_dir,
            repo_type="dataset",
            ignore_patterns=".gitattributes",
        )
        
        # Count downloaded images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in output_path.iterdir() if f.suffix.lower() in image_extensions]
        
        print(f"✅ Downloaded {len(images)} images to {output_dir}")
        print(f"\n📝 Recommended prompts for training:")
        print(f"   Instance prompt: {dataset_info['instance_prompt']}")
        print(f"   Class prompt: {dataset_info['class_prompt']}")
        print(f"\n💡 To train with these images:")
        print(f"   export INSTANCE_PROMPT=\"{dataset_info['instance_prompt']}\"")
        print(f"   export CLASS_PROMPT=\"{dataset_info['class_prompt']}\"")
        print(f"   bash run.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False


def list_datasets():
    """List all available example datasets."""
    print("Available example datasets:\n")
    for name, info in EXAMPLE_DATASETS.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Repository: {info['repo']}")
        print(f"    Instance prompt: {info['instance_prompt']}")
        print(f"    Class prompt: {info['class_prompt']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download example images for DreamBooth training"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset to download (dog, cat, man). Use 'list' to see all options.",
    )
    parser.add_argument(
        "-o", "--output",
        default="./instance_images",
        help="Output directory (default: ./instance_images)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available datasets",
    )
    
    args = parser.parse_args()
    
    if args.list or args.dataset == "list":
        list_datasets()
        return 0
    
    if not args.dataset:
        print("Usage: python download_example_images.py <dataset>")
        print("\nAvailable datasets: dog, cat, man")
        print("For more info: python download_example_images.py --list")
        return 1
    
    success = download_example_dataset(args.dataset, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
