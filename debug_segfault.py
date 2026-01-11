#!/usr/bin/env python3
"""
Debug script to find where the segfault occurs
Run this to isolate the problem
"""

import sys
print("="*60)
print("DreamBooth Segfault Debug Script")
print("="*60)

# Test 1: Basic imports
print("\n[1/10] Testing basic imports...")
try:
    import torch
    print(f"  ✅ torch {torch.__version__}")
    import torchvision
    print(f"  ✅ torchvision {torchvision.__version__}")
    print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 2: Diffusers imports
print("\n[2/10] Testing diffusers imports...")
try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
    print(f"  ✅ diffusers imported")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 3: Accelerate
print("\n[3/10] Testing accelerate...")
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    print(f"  ✅ accelerate imported")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 4: GPU memory
print("\n[4/10] Checking GPU memory...")
try:
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        free = total - allocated
        print(f"  ✅ Total: {total:.1f} GB")
        print(f"  ✅ Free: {free:.1f} GB")
        print(f"  ✅ Allocated: {allocated:.1f} GB")
        if free < 5:
            print(f"  ⚠️  WARNING: Low free memory!")
except Exception as e:
    print(f"  ❌ ERROR: {e}")

# Test 5: Load tokenizer (lightweight)
print("\n[5/10] Loading tokenizer...")
try:
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
    )
    print(f"  ✅ Tokenizer loaded")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 6: Load text encoder
print("\n[6/10] Loading text encoder...")
try:
    from transformers import CLIPTextModel
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder",
    )
    print(f"  ✅ Text encoder loaded")
    print(f"  ✅ Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    print(f"  ⚠️  This might be where segfault occurs!")
    sys.exit(1)

# Test 7: Load VAE
print("\n[7/10] Loading VAE...")
try:
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
    )
    print(f"  ✅ VAE loaded")
    print(f"  ✅ Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    print(f"  ⚠️  This might be where segfault occurs!")
    sys.exit(1)

# Test 8: Load UNet (this is the big one)
print("\n[8/10] Loading UNet (largest component)...")
try:
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
    )
    print(f"  ✅ UNet loaded")
    print(f"  ✅ Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    print(f"  ⚠️  This might be where segfault occurs!")
    sys.exit(1)

# Test 9: Move to GPU
print("\n[9/10] Moving models to GPU...")
try:
    if torch.cuda.is_available():
        text_encoder = text_encoder.to("cuda")
        vae = vae.to("cuda")
        unet = unet.to("cuda")
        print(f"  ✅ Models on GPU")
        print(f"  ✅ Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    print(f"  ⚠️  This might be where segfault occurs!")
    sys.exit(1)

# Test 10: Initialize Accelerator
print("\n[10/10] Initializing Accelerator...")
try:
    accelerator = Accelerator(
        mixed_precision="no",  # Test without mixed precision
    )
    print(f"  ✅ Accelerator initialized")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    print(f"  ⚠️  This might be where segfault occurs!")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed! No segfault in model loading.")
print("="*60)
print("\nThe segfault might be occurring during:")
print("  - Mixed precision (fp16) operations")
print("  - Training loop initialization")
print("  - Data loading")
print("  - Optimizer creation")
print("\nTry running with --mixed_precision=\"no\" to test.")
print("="*60)
