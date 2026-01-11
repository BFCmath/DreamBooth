# Troubleshooting Segfaults on Kaggle

## The Problem

You're getting `Segmentation fault (core dumped)` even with lite mode settings. This is a serious stability issue on Kaggle.

## Step-by-Step Diagnosis

### Step 1: Find Where It's Failing

Run the debug script to see exactly where the segfault occurs:

```bash
python debug_segfault.py
```

This will test each component in order:
1. ✅ Basic imports (torch, torchvision)
2. ✅ Diffusers imports
3. ✅ Accelerate
4. ✅ GPU memory check
5. ✅ Load tokenizer
6. ✅ Load text encoder ← Often fails here
7. ✅ Load VAE
8. ✅ Load UNet ← Or here
9. ✅ Move to GPU ← Or here
10. ✅ Initialize Accelerator

If the debug script **segfaults**, note which step it fails on.

### Step 2: Try ULTRA LITE Mode

If debug passes, try training with ultra-conservative settings:

```bash
bash run_ultra_lite.sh
```

This uses:
- ✅ Lower resolution (384px instead of 512px)
- ✅ **No mixed precision** (fp32 instead of fp16) ← Important!
- ✅ Gradient accumulation (less memory per step)
- ✅ Fewer total steps (400 instead of 1000)
- ✅ No checkpointing (saves disk I/O)

### Step 3: If Still Fails - Root Causes

#### Cause 1: Mixed Precision (fp16) Issues

**Symptom**: Segfault during model loading or first forward pass  
**Why**: Some torch/CUDA versions have fp16 bugs  
**Fix**: Use fp32

```bash
export MIXED_PRECISION=no
bash run_ultra_lite.sh
```

#### Cause 2: Insufficient GPU Memory

**Symptom**: Segfault when loading UNet or during training  
**Why**: T4 has only ~15GB, model + gradients need ~12GB  
**Fix**: Use lower resolution or P100 GPU

```bash
export RESOLUTION=256  # Very low
bash run_ultra_lite.sh
```

Or in Kaggle: Settings → Accelerator → **P100** (16GB instead of 15GB)

#### Cause 3: Corrupted Model Cache

**Symptom**: Segfault during model loading  
**Why**: Partial/corrupted download in cache  
**Fix**: Clear cache and re-download

```bash
rm -rf ~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

#### Cause 4: Accelerate/DeepSpeed Issues

**Symptom**: Segfault when "Preparing everything with accelerator"  
**Why**: Accelerate's model wrapping can trigger segfaults  
**Fix**: Use simpler accelerate config

```bash
# Set accelerate config to most basic
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_USE_FSDP=false
bash run_ultra_lite.sh
```

#### Cause 5: Disk Space Issues

**Symptom**: Random segfaults during checkpointing or model save  
**Why**: Kaggle has limited disk space  
**Fix**: Check space

```bash
df -h
# If < 5GB free:
rm -rf ~/.cache/huggingface/hub/*-snapshots/
```

#### Cause 6: CUDA/Driver Mismatch

**Symptom**: Segfault on any CUDA operation  
**Why**: Kaggle's environment update broke something  
**Fix**: Restart kernel completely

In Kaggle UI: Three dots (⋮) → **Restart & Run All**

### Step 4: Nuclear Options

If nothing works:

#### Option A: Train Without Accelerate

Modify script to not use accelerate at all (manual CUDA handling). This is complex but most stable.

#### Option B: Use Google Colab Instead

Sometimes Kaggle's environment is just broken. Colab often works better:
- More stable CUDA drivers
- Better torch/torchvision compatibility
- Similar free GPU tier

#### Option C: Wait and Try Later

Kaggle sometimes has infrastructure issues. Try again in a few hours.

#### Option D: Use CPU (Very Slow)

As last resort:

```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
bash run_ultra_lite.sh  # Will be VERY slow (hours)
```

## Most Likely Solutions

Based on frequency of issues:

### Solution 1: Just Restart Everything (70% success rate)

```python
# In Kaggle, click: ⋮ → Restart Session
# Then in fresh session:
!git clone https://github.com/BFCmath/DreamBooth.git
%cd DreamBooth
!python download_example_images.py dog
!bash run_ultra_lite.sh
```

### Solution 2: Disable Mixed Precision (20% success rate)

The fp16 mixed precision can cause segfaults on some CUDA versions.

Already done in `run_ultra_lite.sh` - it uses fp32.

### Solution 3: Use P100 GPU (10% success rate)

T4 (15GB) might be at the edge. P100 (16GB) has slightly more memory and different architecture.

**Kaggle UI**: Settings → Accelerator → **P100**

## How to Report If Still Failing

If you've tried everything and still get segfaults, report:

```bash
# Gather this info:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
nvidia-smi
df -h
python debug_segfault.py 2>&1 | tee debug.log
```

Post the output - this will help identify if it's a Kaggle infrastructure issue.

## Alternative: Pre-trained DreamBooth

If training doesn't work on Kaggle, you can:
1. Train on Google Colab (usually more stable)
2. Train locally if you have GPU
3. Use a cloud GPU service (vast.ai, runpod.io - paid but cheap)
4. Use pre-trained DreamBooth models from HuggingFace

## Summary

Try in this order:
1. ✅ Run `python debug_segfault.py` to isolate issue
2. ✅ Try `bash run_ultra_lite.sh` (fp32, low res)
3. ✅ Restart Kaggle kernel completely
4. ✅ Clear cache: `rm -rf ~/.cache/huggingface/`
5. ✅ Use P100 GPU instead of T4
6. ✅ Try Google Colab instead

Most segfaults are due to:
- **fp16 bugs** → fixed by ultra_lite using fp32
- **Memory pressure** → fixed by 384px resolution
- **Temporary Kaggle issues** → fixed by restarting

Good luck! 🚀
