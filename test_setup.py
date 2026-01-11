#!/usr/bin/env python3
"""
Test script to verify DreamBooth setup on Kaggle
Run this before training to catch issues early
"""

import sys
from pathlib import Path

def test_dependencies():
    """Test that all required dependencies are installed."""
    print("Testing dependencies...")
    
    required = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
    ]
    
    optional = [
        ("xformers", "xformers (for memory efficiency)"),
        ("bitsandbytes", "bitsandbytes (for 8-bit Adam)"),
    ]
    
    all_good = True
    
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING (required)")
            all_good = False
    
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - not installed (optional)")
    
    return all_good


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA is available")
            print(f"  📊 Device: {torch.cuda.get_device_name(0)}")
            print(f"  💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print(f"  ❌ CUDA is NOT available")
            print(f"  ⚠️  Training will be very slow on CPU!")
            return False
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False


def test_model_access():
    """Test that we can access the base model."""
    print("\nTesting model access...")
    
    try:
        from diffusers import StableDiffusionPipeline
        print(f"  ⏳ Attempting to load runwayml/stable-diffusion-v1-5...")
        print(f"     (This may take a minute on first run)")
        
        # Just check if we can load the config
        from huggingface_hub import model_info
        info = model_info("runwayml/stable-diffusion-v1-5")
        print(f"  ✅ Model is accessible")
        return True
    except Exception as e:
        print(f"  ❌ Cannot access model: {e}")
        print(f"  ⚠️  You may need to accept the license at:")
        print(f"     https://huggingface.co/runwayml/stable-diffusion-v1-5")
        return False


def test_instance_images():
    """Test that instance images exist."""
    print("\nTesting instance images...")
    
    instance_dir = Path("instance_images")
    
    if not instance_dir.exists():
        print(f"  ❌ instance_images/ directory does not exist")
        print(f"  ℹ️  Create it with: mkdir instance_images")
        return False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    images = [f for f in instance_dir.iterdir() if f.suffix in image_extensions]
    
    if len(images) == 0:
        print(f"  ❌ No images found in instance_images/")
        print(f"  ℹ️  Add 3-10 high-quality images of your subject")
        return False
    
    print(f"  ✅ Found {len(images)} training images")
    
    # Check image properties
    try:
        from PIL import Image
        sizes = []
        for img_path in images[:5]:  # Check first 5
            img = Image.open(img_path)
            sizes.append(img.size)
            if img.mode != 'RGB':
                print(f"  ⚠️  {img_path.name} is {img.mode}, will be converted to RGB")
        
        print(f"  📏 Image sizes: {sizes[:3]}{'...' if len(sizes) > 3 else ''}")
        
        # Warn if images are too small
        min_size = min(min(w, h) for w, h in sizes)
        if min_size < 512:
            print(f"  ⚠️  Some images are smaller than 512px (smallest: {min_size}px)")
            print(f"     Consider using higher resolution images for best results")
    
    except Exception as e:
        print(f"  ⚠️  Error checking images: {e}")
    
    return True


def test_xformers():
    """Test xformers functionality."""
    print("\nTesting xformers...")
    
    try:
        import xformers
        print(f"  ✅ xformers is installed (version {xformers.__version__})")
        
        # Try to actually use it
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet"
        )
        try:
            unet.enable_xformers_memory_efficient_attention()
            print(f"  ✅ xformers memory efficient attention works")
            return True
        except Exception as e:
            print(f"  ⚠️  xformers installed but cannot enable: {e}")
            return False
    except ImportError:
        print(f"  ⚠️  xformers not installed (optional but recommended)")
        return False
    except Exception as e:
        print(f"  ⚠️  Error testing xformers: {e}")
        return False


def test_disk_space():
    """Test available disk space."""
    print("\nTesting disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"  💾 Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print(f"  ⚠️  Low disk space! DreamBooth needs ~10-15GB")
            print(f"     (model downloads + class images + checkpoints)")
            return False
        else:
            print(f"  ✅ Sufficient disk space")
            return True
    except Exception as e:
        print(f"  ⚠️  Error checking disk space: {e}")
        return False


def main():
    print("="*60)
    print("DreamBooth Setup Test")
    print("="*60)
    
    results = {
        "Dependencies": test_dependencies(),
        "CUDA": test_cuda(),
        "Model Access": test_model_access(),
        "Instance Images": test_instance_images(),
        "Disk Space": test_disk_space(),
    }
    
    # Optional tests
    test_xformers()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All critical tests passed! Ready to train.")
        print("\nTo start training, run:")
        print("  bash run.sh")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Add training images to instance_images/")
        print("  - Enable GPU in Kaggle notebook settings")
        print("  - Accept SD license: https://huggingface.co/runwayml/stable-diffusion-v1-5")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
