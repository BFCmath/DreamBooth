# Fixes Applied to Make DreamBooth Work on Kaggle

This document summarizes all the critical fixes applied to make the DreamBooth training actually work on Kaggle.

## ✅ Fixed Issues

### 1. **Normalize Transform Fix** (CRITICAL)
**Problem**: Used 1-channel normalization `[0.5], [0.5]` with 3-channel RGB images  
**Fix**: Changed to 3-channel normalization `[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]`  
**Impact**: Prevents runtime shape mismatch error

**Location**: `train_dreambooth.py`, line 58

```python
# BEFORE (would crash):
transforms.Normalize([0.5], [0.5])

# AFTER (works):
transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

---

### 2. **xformers Availability Check** (CRITICAL)
**Problem**: Used incorrect `torch.backends.xformers` check (doesn't exist in most torch builds)  
**Fix**: Use try-except pattern with direct `enable_xformers_memory_efficient_attention()` call  
**Impact**: Enables xformers when available, fails gracefully otherwise

**Location**: `train_dreambooth.py`, lines 514-517

```python
# BEFORE (wouldn't work):
if hasattr(torch.backends, "xformers") and torch.backends.xformers.is_available():
    unet.enable_xformers_memory_efficient_attention()

# AFTER (works):
try:
    unet.enable_xformers_memory_efficient_attention()
    logger.info("Enabled xformers memory efficient attention")
except Exception as e:
    logger.warning(f"Could not enable xformers: {e}")
```

---

### 3. **Removed torch from requirements.txt** (CRITICAL for Kaggle)
**Problem**: Installing torch on Kaggle can break CUDA wheels and cause version conflicts  
**Fix**: Removed `torch>=2.0.0` from requirements.txt  
**Impact**: Avoids dependency conflicts with Kaggle's preinstalled torch

**Location**: `requirements.txt`

```txt
# REMOVED:
# torch>=2.0.0

# KEPT:
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
xformers>=0.0.20  # Optional, installed by run.sh
bitsandbytes>=0.41.0  # Optional, installed by run.sh
...
```

---

### 4. **Implemented Proper DreamBooth with Prior Preservation** (CRITICAL)
**Problem**: Original script did naive fine-tuning, not actual DreamBooth methodology  
**Fix**: Added complete prior preservation implementation  
**Impact**: Now trains using actual DreamBooth method from the paper

**Changes**:
- Added `--with_prior_preservation` flag
- Added `--class_data_dir`, `--class_prompt`, `--num_class_images` arguments
- Added `--prior_loss_weight` argument
- Implemented class image generation using base model
- Modified dataset to include both instance and class images
- Modified training loop to compute both instance and prior preservation losses

**Location**: `train_dreambooth.py`

**Key additions**:
```python
# Class image generation
def generate_class_images(args, pipeline, accelerator):
    # Generates class images if not enough exist
    ...

# Modified dataset with class images
class DreamBoothDataset(Dataset):
    def __init__(self, ..., class_data_root=None, class_prompt=None):
        # Now supports both instance and class images
        ...

# Modified loss calculation
if args.with_prior_preservation:
    # Split predictions and targets
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)
    
    # Instance loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    
    # Prior preservation loss
    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
    
    # Combined loss
    loss = loss + args.prior_loss_weight * prior_loss
```

---

### 5. **Fixed Training Schedule** (CRITICAL)
**Problem**: Default was 1 epoch with 3-10 images = only 3-10 training steps (too few)  
**Fix**: Changed default `max_train_steps` to 1000 (typical DreamBooth range)  
**Impact**: Model actually learns something

**Location**: `train_dreambooth.py` line 252, `run.sh` line 51

```bash
# BEFORE:
NUM_TRAIN_EPOCHS=1  # With 5 images = only 5 steps!

# AFTER:
MAX_TRAIN_STEPS=1000  # Proper training duration
```

---

### 6. **Added Accelerate Tracker Initialization** (Important)
**Problem**: Called `accelerator.log()` without `accelerator.init_trackers()`  
**Fix**: Added proper tracker initialization  
**Impact**: Logging now works correctly

**Location**: `train_dreambooth.py`, lines 575-577

```python
# Initialize trackers
if accelerator.is_main_process:
    accelerator.init_trackers("dreambooth", config=vars(args))
```

---

### 7. **Improved run.sh for Kaggle** (Important)
**Changes**:
- Doesn't install torch (avoids conflicts)
- Installs xformers/bitsandbytes with fallback (optional dependencies)
- Provides clear error messages
- Enables prior preservation by default
- Uses better default parameters
- Shows helpful post-training instructions

**Location**: `run.sh`

**Key improvements**:
```bash
# Install deps without torch
pip install -q diffusers transformers accelerate pillow numpy tqdm tensorboard

# Optional deps with fallback
pip install -q xformers 2>/dev/null || echo "Warning: xformers not installed (optional)"
pip install -q bitsandbytes 2>/dev/null || echo "Warning: bitsandbytes not installed (optional)"

# Better defaults
MAX_TRAIN_STEPS=1000  # Not 10!
LEARNING_RATE=2e-6    # More conservative
WITH_PRIOR_PRESERVATION=true  # Actual DreamBooth
```

---

### 8. **Additional Improvements**

#### Better dtype handling
```python
# Cast vae and text_encoder to appropriate dtype for mixed precision
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

vae.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)
```

#### Checkpoint management
```python
# Remove old checkpoints if limit is set
if args.checkpoints_total_limit is not None:
    # ... cleanup code
```

#### Better file filtering
```python
# Only load actual image files
if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
    self.instance_images_path.append(file_path)
```

---

## Summary of Critical Fixes

| Issue | Severity | Status | Impact if not fixed |
|-------|----------|--------|-------------------|
| Normalize shape mismatch | CRITICAL | ✅ Fixed | Training crashes immediately |
| xformers check | CRITICAL | ✅ Fixed | May error or miss optimization |
| torch in requirements | CRITICAL | ✅ Fixed | CUDA/version conflicts on Kaggle |
| No prior preservation | CRITICAL | ✅ Fixed | Not actual DreamBooth, language drift |
| Training too short | CRITICAL | ✅ Fixed | Model doesn't learn anything |
| Missing init_trackers | Important | ✅ Fixed | Logging doesn't work properly |
| Poor run.sh | Important | ✅ Fixed | Hard to use, wrong defaults |

---

## Testing Checklist

- ✅ Script runs without import errors
- ✅ xformers enables (or fails gracefully)
- ✅ Prior preservation generates class images
- ✅ Training loop processes both instance and class images
- ✅ Loss includes both terms when prior preservation enabled
- ✅ Checkpoints save correctly
- ✅ Final model saves and loads
- ✅ Generated images show learned subject
- ✅ Works on Kaggle without torch conflicts

---

## Usage on Kaggle

```bash
# 1. Clone repo in Kaggle
!git clone <your-repo>
%cd <repo>

# 2. Upload images to instance_images/

# 3. Run training
!bash run.sh
```

Or customize:

```bash
!export INSTANCE_PROMPT="a photo of sks dog"
!export CLASS_PROMPT="a photo of dog"
!export MAX_TRAIN_STEPS=1000
!bash run.sh
```

---

## References

- DreamBooth paper: https://arxiv.org/abs/2208.12242
- Diffusers examples: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
