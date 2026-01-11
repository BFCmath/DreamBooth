# Multi-GPU Training Guide for Kaggle

If you have access to **2x T4** GPUs on Kaggle (30GB total memory), you can train much faster and with better quality!

## Quick Start

```bash
# Enable 2x T4 in Kaggle
# Settings → Accelerator → GPU T4 x 2

# Download images
python download_example_images.py dog

# Train with both GPUs
bash run_multi_gpu.sh
```

That's it! The script automatically:
- ✅ Detects both GPUs
- ✅ Distributes training across them
- ✅ Uses proper distributed training settings
- ✅ Enables prior preservation (100 class images)
- ✅ Uses fp16 mixed precision safely

## Benefits of Multi-GPU

| Feature | Single T4 | 2x T4 |
|---------|-----------|-------|
| Total Memory | 15 GB | **30 GB** |
| Training Speed | 1x | **~1.8x faster** |
| Batch Size | 1 | Can use 1-2 per GPU |
| Prior Preservation | Risky | **Stable** |
| Resolution | 384-512px | **512px easy** |
| Class Images | 50-100 | **100-200 easy** |

## How It Works

The script uses **Accelerate's DistributedDataParallel (DDP)**:

1. **Model is replicated** on both GPUs
2. **Batches are split** between GPUs
3. **Gradients are synchronized** after each step
4. **Effective batch size** = batch_size × num_gpus × gradient_accumulation

Example:
- Batch size per GPU: 1
- Number of GPUs: 2
- Gradient accumulation: 2
- **Effective batch size: 4**

## Configuration

The script uses `accelerate_config_multi_gpu.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2          # One process per GPU
mixed_precision: fp16     # Safe with distributed training
gpu_ids: all              # Use all available GPUs
```

## Customization

```bash
# Use higher batch size (more memory per GPU)
export TRAIN_BATCH_SIZE=2  # 2 per GPU = 4 total

# More class images for better prior preservation
export NUM_CLASS_IMAGES=200

# Higher resolution
export RESOLUTION=512  # Or even 768 if you have memory

# Run
bash run_multi_gpu.sh
```

## Comparison: Single vs Multi-GPU

### Single GPU (run_lite.sh)
```bash
Resolution: 384px
Batch size: 1
Steps: 800
Prior preservation: false
Memory: ~14GB used
Time: ~40 minutes
```

### Multi-GPU (run_multi_gpu.sh)
```bash
Resolution: 512px
Batch size: 1 per GPU (2 total)
Steps: 1000
Prior preservation: true (100 images)
Memory: ~14GB per GPU (28GB total)
Time: ~25 minutes
```

## Troubleshooting

### OOM with Multi-GPU

If you still get Out of Memory:

```bash
# Reduce resolution
export RESOLUTION=448
bash run_multi_gpu.sh

# Or reduce class images
export NUM_CLASS_IMAGES=50
bash run_multi_gpu.sh

# Or disable prior preservation
export WITH_PRIOR_PRESERVATION=false
bash run_multi_gpu.sh
```

### Only 1 GPU Detected

Check Kaggle settings:
- Settings → Accelerator → **GPU T4 x 2** (not just "GPU T4")

Verify:
```bash
nvidia-smi --list-gpus
# Should show 2 GPUs
```

### Slower Than Expected

Distributed training has overhead. Benefits show with:
- ✅ Larger models (like SD)
- ✅ More training steps
- ✅ Larger batch sizes

For very short training (<100 steps), single GPU might be similar speed.

### Different GPU Types

If you have P100 x 2 or other GPU combinations:
```bash
bash run_multi_gpu.sh  # Auto-detects and uses all GPUs
```

## Advanced: Manual Accelerate Configuration

If you want full control:

```bash
# Configure accelerate interactively
accelerate config

# Then train
accelerate launch train_dreambooth.py \
    --instance_data_dir="./instance_images" \
    --instance_prompt="a photo of sks dog" \
    --max_train_steps=1000 \
    --with_prior_preservation \
    --class_data_dir="./class_images" \
    --class_prompt="a photo of dog" \
    --mixed_precision="fp16"
```

## Best Settings for 2x T4

Recommended configuration for quality + speed:

```bash
export INSTANCE_PROMPT="a photo of sks person"
export CLASS_PROMPT="a photo of person"
export RESOLUTION=512
export TRAIN_BATCH_SIZE=1        # Per GPU
export GRADIENT_ACCUMULATION=2   # Effective batch size = 4
export MAX_TRAIN_STEPS=1000
export NUM_CLASS_IMAGES=100
export WITH_PRIOR_PRESERVATION=true

bash run_multi_gpu.sh
```

This gives:
- ✅ High quality (proper DreamBooth with prior preservation)
- ✅ Fast training (~25 minutes)
- ✅ Stable (doesn't OOM)
- ✅ Good resolution (512px)

## When to Use Multi-GPU vs Single GPU

### Use Multi-GPU (run_multi_gpu.sh) when:
- ✅ You have 2x T4 or better
- ✅ Want prior preservation with 100+ class images
- ✅ Want full 512px resolution
- ✅ Want faster training
- ✅ Training for 500+ steps

### Use Single GPU (run_lite.sh) when:
- ✅ You only have 1 GPU
- ✅ Want simplest setup
- ✅ Quick testing/experimentation
- ✅ Small datasets (<5 images)

## Summary

With **2x T4 GPUs**, you get the best of both worlds:
- **Fast training** (~25 min instead of 40 min)
- **High quality** (can use prior preservation)
- **Stable** (no OOM with 30GB total memory)
- **Full resolution** (512px easily)

Just run: `bash run_multi_gpu.sh` 🚀

## Example: Complete Multi-GPU Workflow

```bash
# 1. Enable 2x T4 in Kaggle settings

# 2. Clone and setup
!git clone https://github.com/BFCmath/DreamBooth.git
%cd DreamBooth

# 3. Get images
!python download_example_images.py dog

# 4. Train with both GPUs
!bash run_multi_gpu.sh

# 5. Generate images
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "./output/dreambooth-model",
    torch_dtype=torch.float16
).to("cuda")

image = pipe("a photo of sks dog in space", num_inference_steps=50).images[0]
display(image)
```

Done! Your model will be trained using both GPUs efficiently. 🎉
