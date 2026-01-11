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

#### If you used example dog images:

```bash
%%bash
export INSTANCE_PROMPT="a photo of sks dog"
export CLASS_PROMPT="a photo of dog"
bash run.sh
```

#### If you used your own images:

```bash
%%bash
export INSTANCE_PROMPT="a photo of sks person"  # Change based on your subject
export CLASS_PROMPT="a photo of person"         # Change to match
bash run.sh
```

### 5. Wait for Training

Training takes approximately:
- **T4 GPU**: ~30-40 minutes
- **P100 GPU**: ~20-30 minutes
- **T4 x2 GPU**: ~15-20 minutes

You'll see progress with:
- Class image generation (if first time)
- Training progress bar with loss
- Checkpoint saves every 500 steps

### 6. Generate Images

After training completes:

```python
from diffusers import StableDiffusionPipeline
import torch

# Load your trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "./output/dreambooth-model",
    torch_dtype=torch.float16
).to("cuda")

# Generate an image
prompt = "a photo of sks dog in a beautiful garden"  # Change based on your subject
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# Display
display(image)

# Save
image.save("my_generated_image.png")
```

### 7. Try Different Prompts

```python
prompts = [
    "a photo of sks dog on the beach",
    "a photo of sks dog wearing sunglasses",
    "oil painting of sks dog",
    "sks dog as a superhero",
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=50).images[0]
    display(image)
    image.save(f"output_{i}.png")
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

### CUDA Errors
- Make sure GPU is enabled in Kaggle settings
- Restart notebook and try again
- Don't run `pip install torch` (uses Kaggle's preinstalled version)

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
