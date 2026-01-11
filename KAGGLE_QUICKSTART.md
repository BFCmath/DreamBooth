# Kaggle Quick Start Guide

The absolute fastest way to get DreamBooth training running on Kaggle.

## Step-by-Step

### 1. Create Kaggle Notebook

- Go to [Kaggle](https://www.kaggle.com/)
- Create new notebook
- Enable GPU: **Settings → Accelerator → GPU T4** (or P100)

### 2. Clone Repository

In a code cell:

```python
!git clone https://github.com/BFCmath/DreamBooth.git
%cd DreamBooth
```

### 3. Choose Your Images

#### Option A: Test with Example Images (Fastest)

```python
# Download example dog images
!python download_example_images.py dog

# See all available examples
!python download_example_images.py --list
```

Available examples: `dog`, `cat`, `man`

#### Option B: Use Your Own Images

Upload 3-10 images to `instance_images/`:

```python
# Upload using Kaggle UI, or:
!mkdir -p instance_images
!wget -O instance_images/img1.jpg "https://your-url.com/image1.jpg"
!wget -O instance_images/img2.jpg "https://your-url.com/image2.jpg"
# ... add more images
```

### 4. Run Training

#### ✅ RECOMMENDED: Single GPU Optimized (Most Reliable)

This uses fp16 + gradient checkpointing + 8-bit Adam to fit a single T4 without OOM:

```bash
%%bash
export INSTANCE_PROMPT="a photo of sks dog"
bash run_single_gpu_optimized.sh
```

**Why this works**: fp16 cuts memory, gradient checkpointing reduces peak VRAM, 8-bit Adam saves optimizer memory, gradient accumulation keeps effective batch size 4, ~40min, 512px resolution

#### Option B: Multi-GPU Stable (If you have 2x T4)

```bash
%%bash
export INSTANCE_PROMPT="a photo of sks dog"
bash run_multi_gpu_stable.sh
```

**Note**: Uses fp32 for stability, ~25min with 2 GPUs

#### Option C: Ultra-Lite (If memory issues)

```bash
%%bash
export INSTANCE_PROMPT="a photo of sks dog"
export RESOLUTION=384
bash run_ultra_lite.sh
```

### 5. Wait for Training

Training takes approximately:
- **2x T4 GPU** (run_multi_gpu.sh): ~20-25 minutes ⚡ (FASTEST)
- **Single T4** (run_lite.sh): ~30-40 minutes
- **Single P100**: ~20-30 minutes
- **Single T4** (run_ultra_lite.sh): ~25-35 minutes (lower res)

You'll see progress with:
- Class image generation (if first time)
- Training progress bar with loss
- Checkpoint saves every 500 steps

### 6. Generate Images

#### ✅ EASIEST: Use the Quick Generation Script

After training completes, just run:

```python
!python generate_quick.py
```

This will automatically:
- Load your trained model
- Generate 5 different test images
- Display them in the notebook
- Save them to `./generated_images/`

#### Option B: Generate Single Image (Manual)

```python
from diffusers import StableDiffusionPipeline
import torch

# Load your trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "./output/dreambooth-model",
    torch_dtype=torch.float16,
    safety_checker=None  # Faster inference
).to("cuda")

# Generate an image
prompt = "a photo of sks person in a beautiful garden"  # Change 'person' to your subject
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# Display and save
display(image)
image.save("my_generated_image.png")
```

#### Option C: Advanced Generation (Custom Prompts)

```bash
# Generate with custom prompts and settings
!python generate_images.py \
    --prompts "a photo of sks person at the beach" \
              "sks person wearing sunglasses" \
              "oil painting of sks person" \
    --num_images_per_prompt 2 \
    --num_inference_steps 75 \
    --guidance_scale 8.0 \
    --seed 42
```

See `python generate_images.py --help` for all options.

### 7. Tips for Better Results

**Prompt Tips:**
- Always include your unique identifier (`sks person`, `sks dog`, etc.)
- Try different styles: "photo", "oil painting", "digital art", "sketch"
- Add context: "at the beach", "in space", "wearing a hat"
- Adjust `guidance_scale`: 7.5 (default), higher = more prompt adherence
- Increase `num_inference_steps` for better quality (50-100)

**Example Prompts:**

```python
prompts = [
    "a professional photo of sks person in a business suit",
    "sks person as a astronaut in space",
    "oil painting of sks person in renaissance style",
    "sks person wearing sunglasses at sunset",
    "digital art of sks person as a superhero",
]

for prompt in prompts:
    image = pipe(prompt, num_inference_steps=75, guidance_scale=8.0).images[0]
    display(image)
```

## Complete Single-Cell Example

Copy this entire cell into Kaggle:

```python
# Setup
!git clone https://github.com/BFCmath/DreamBooth.git
%cd DreamBooth

# Get example images
!python download_example_images.py dog

# Train (this takes ~30 minutes)
import os
os.environ["INSTANCE_PROMPT"] = "a photo of sks dog"
os.environ["CLASS_PROMPT"] = "a photo of dog"
!bash run.sh

# Generate
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "./output/dreambooth-model",
    torch_dtype=torch.float16
).to("cuda")

image = pipe("a photo of sks dog in space", num_inference_steps=50).images[0]
display(image)
image.save("result.png")
```

## Troubleshooting

### "No training images found"

```python
# Check what's in the directory
!ls -la instance_images/

# Download examples if empty
!python download_example_images.py dog
```

### Out of Memory

```bash
%%bash
export RESOLUTION=384
export TRAIN_BATCH_SIZE=1
bash run.sh
```

### CUDA Errors / torchvision::nms Error
- Make sure GPU is enabled in Kaggle settings
- **Restart notebook** and try again
- **Don't run** `pip install torch` or `pip install torchvision`
- Kaggle has these preinstalled - installing them causes version conflicts!

### Segmentation Fault

This is a serious issue. Try in order:

```bash
# 1. Debug where it's failing
!python debug_segfault.py

# 2. Try ultra-lite mode (fp32, no mixed precision)
!bash run_ultra_lite.sh

# 3. If still fails, restart kernel completely
# Kaggle UI: ⋮ → Restart Session
# Then run again

# 4. Use P100 instead of T4
# Kaggle UI: Settings → Accelerator → P100
```

See [TROUBLESHOOT_KAGGLE.md](TROUBLESHOOT_KAGGLE.md) for detailed diagnosis.

### Slow Training
- Make sure GPU is enabled (not CPU)
- Use T4 x2 if available for faster training
- Reduce `MAX_TRAIN_STEPS` to 600 for quicker (but lower quality) results

## Advanced: Custom Configuration

```bash
%%bash
export INSTANCE_PROMPT="a photo of sks person"
export CLASS_PROMPT="a photo of person"
export MAX_TRAIN_STEPS=800              # Fewer steps = faster
export LEARNING_RATE=2e-6               # Lower = more stable
export RESOLUTION=512                    # Lower = uses less memory
export NUM_CLASS_IMAGES=100             # Fewer = faster class generation
bash run.sh
```

## Next Steps

- Try different prompts and styles
- Download your model: `!zip -r model.zip output/dreambooth-model`
- Train with your own images
- Experiment with different subjects (pets, people, objects, art styles)

## Need Help?

Check the main [README.md](README.md) for:
- Detailed parameter explanations
- Training tips and best practices
- Common issues and solutions
- How to use your trained model elsewhere

Good luck! 🚀
