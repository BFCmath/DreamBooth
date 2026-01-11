# Fix for Segmentation Fault on Kaggle

## Error: `Segmentation fault (core dumped)`

This error typically occurs due to memory issues or incompatible compiled extensions on Kaggle.

## Quick Fixes (Try in Order)

### Fix 1: Use LITE MODE (Recommended)

Use the memory-optimized version that avoids problematic dependencies:

```bash
bash run_lite.sh
```

This version:
- ✅ Skips xformers (can cause segfaults)
- ✅ Skips bitsandbytes (can cause segfaults)  
- ✅ Uses fewer class images (50 instead of 200)
- ✅ Disables prior preservation by default (saves memory)
- ✅ More stable on Kaggle

### Fix 2: Reduce Memory Usage

If you want to keep prior preservation but avoid segfaults:

```bash
export NUM_CLASS_IMAGES=50        # Fewer class images
export SAMPLE_BATCH_SIZE=1        # Smaller batches
export MAX_TRAIN_STEPS=600        # Fewer steps
bash run.sh
```

### Fix 3: Disable Prior Preservation

Train without prior preservation (uses much less memory):

```bash
export WITH_PRIOR_PRESERVATION=false
bash run.sh
```

This skips class image generation entirely, which is often where the segfault occurs.

### Fix 4: Check GPU Memory

Make sure you have enough GPU memory:

```python
!nvidia-smi
```

If "Memory-Usage" shows you're at the limit:
1. Use a higher GPU tier (T4 x2 or P100)
2. Or use lite mode

## Common Causes

### 1. **xformers/bitsandbytes Segfault**

These compiled packages can segfault if incompatible with Kaggle's environment.

**Solution**: Use `run_lite.sh` which skips these packages.

### 2. **Out of Memory During Class Image Generation**

Generating 200 class images loads the model multiple times.

**Solution**: Reduce to 50 images or disable prior preservation.

```bash
export NUM_CLASS_IMAGES=50
bash run.sh
```

### 3. **Model Download Issues**

First time downloading SD v1.5 (4GB) can cause issues.

**Solution**: Pre-download the model:

```python
from diffusers import StableDiffusionPipeline

# Download once
print("Downloading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
print("Model cached!")
del pipe

# Now train
!bash run.sh
```

### 4. **Insufficient Disk Space**

Kaggle notebooks have limited disk space.

**Solution**: Check space and clean up:

```bash
df -h
rm -rf ~/.cache/huggingface/hub/*-snapshots/  # Clean old downloads
```

## Recommended: Start with LITE MODE

The safest way to train on Kaggle:

```python
# Complete lite mode example
!git clone https://github.com/BFCmath/DreamBooth.git
%cd DreamBooth

# Download example images
!python download_example_images.py dog

# Use lite mode (most stable)
!bash run_lite.sh
```

## Comparison: Standard vs Lite Mode

| Feature | Standard (`run.sh`) | Lite (`run_lite.sh`) |
|---------|-------------------|-------------------|
| xformers | Tries to install | Skipped |
| bitsandbytes | Tries to install | Skipped |
| Prior preservation | Enabled (200 images) | Disabled by default |
| Class images (if enabled) | 200 | 50 |
| Max steps | 1000 | 800 |
| Stability | May segfault | More stable |
| Quality | Best | Good |
| Memory usage | Higher | Lower |

## If Lite Mode Still Fails

Try training without prior preservation:

```bash
export WITH_PRIOR_PRESERVATION=false
export MAX_TRAIN_STEPS=800
bash run_lite.sh
```

Or reduce resolution:

```bash
export RESOLUTION=384  # Instead of 512
bash run_lite.sh
```

## Advanced: Debug the Segfault

To see exactly where it crashes:

```bash
# Run with debug mode
python -u train_dreambooth.py \
    --instance_data_dir="./instance_images" \
    --instance_prompt="a photo of sks person" \
    --output_dir="./output/test" \
    --max_train_steps=100 \
    --mixed_precision="fp16" 2>&1 | tee training.log
```

Check `training.log` to see where it crashed.

## Summary

✅ **Best solution**: Use `bash run_lite.sh`  
✅ **Alternative**: Reduce class images to 50  
✅ **Last resort**: Disable prior preservation  

❌ **Don't**: Try to install different torch/xformers versions  
❌ **Don't**: Increase batch size or resolution  

## Still Having Issues?

1. **Restart Kaggle session** completely
2. **Use a fresh notebook** 
3. **Enable higher GPU tier** (P100 or T4 x2)
4. **Try different Kaggle environment** (sometimes helps)

Most segfaults on Kaggle are due to:
- xformers being incompatible → Fixed by lite mode
- Too many class images → Fixed by reducing to 50
- Memory pressure → Fixed by disabling prior preservation

Use lite mode for the most stable training experience! 🚀
