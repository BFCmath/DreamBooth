# Fix for Kaggle Version Conflicts

## Error: `RuntimeError: operator torchvision::nms does not exist`

This error occurs when you've installed a version of torchvision that's incompatible with Kaggle's preinstalled torch version.

## Quick Fix

**Restart your Kaggle notebook** - this is the easiest solution:

1. In Kaggle, click the three dots (⋮) in the top right
2. Select "Restart Session"
3. Run your code again (it will use Kaggle's preinstalled compatible versions)

## Alternative: Uninstall and Use Kaggle's Versions

If restarting doesn't work, run this in a cell:

```python
# Uninstall any manually installed versions
!pip uninstall -y torch torchvision torchaudio

# Clear pip cache
!pip cache purge

# Restart the kernel (in Kaggle UI) and try again
```

After running this, **manually restart the kernel** using Kaggle's UI.

## Prevention

Our updated `run.sh` and `requirements.txt` now **exclude** torch and torchvision to prevent this issue.

**What NOT to do:**
```bash
# DON'T install these on Kaggle:
pip install torch
pip install torchvision
pip install -r requirements.txt  # If it includes torch/torchvision
```

**What TO do:**
```bash
# Use Kaggle's preinstalled versions:
bash run.sh  # This now skips torch/torchvision

# Or install only what's needed:
pip install diffusers transformers accelerate huggingface_hub
```

## Why This Happens

- Kaggle comes with **preinstalled** torch and torchvision that match their CUDA version
- Installing new versions creates a **version mismatch**:
  - New torchvision expects features from newer torch
  - But Kaggle's torch is a specific version
  - Result: `torchvision::nms does not exist` error

## Verify Your Setup

After fixing, verify you're using Kaggle's versions:

```python
import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

You should see something like:
```
PyTorch version: 2.1.2+cu121
TorchVision version: 0.16.2+cu121
CUDA available: True
CUDA version: 12.1
```

The `+cu121` part shows these are CUDA-compiled versions compatible with Kaggle's GPU.

## Still Having Issues?

1. **Clear everything and start fresh:**
   ```python
   # In a new Kaggle notebook
   !git clone https://github.com/BFCmath/DreamBooth.git
   %cd DreamBooth
   !python download_example_images.py dog
   !bash run.sh
   ```

2. **Check GPU is enabled:**
   - Settings → Accelerator → GPU T4 (or P100)

3. **Don't install from old requirements.txt:**
   - If you cloned before the fix, pull latest changes
   - The updated version excludes torch/torchvision

## Summary

✅ **Do**: Use Kaggle's preinstalled torch/torchvision  
✅ **Do**: Run `bash run.sh` (updated version)  
✅ **Do**: Restart kernel if you hit errors  

❌ **Don't**: Install torch manually on Kaggle  
❌ **Don't**: Install torchvision manually on Kaggle  
❌ **Don't**: Use old requirements.txt that includes these  
