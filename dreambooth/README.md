# DreamBooth Training Module

This directory contains the essential scripts and configurations for fine-tuning Stable Diffusion models using DreamBooth, optimized for execution in high-compute environments like **Kaggle**.

## 📁 Directory Structure

*   `train_dreambooth.py`: The core training script implementing proper DreamBooth with Prior Preservation Loss.
*   `run.sh`: Main entry script for Kaggle. It installs dependencies, sets up the environment, and triggers the trainer.
*   `run_multi_gpu.sh` / `run_single_gpu_optimized.sh`: Specialized execution scripts for different hardware configurations.
*   `download_example_images.py`: Utility to fetch sample training data.
*   `generate_images.py` / `generate_quick.py`: Inference scripts to test and utilize your trained models.
*   `accelerate_config_multi_gpu.yaml`: Configuration for distributed training using Hugging Face Accelerate.

---

## 🚀 Getting Started on Kaggle

### 1. Prepare Training Images
Place your subject images in an `instance_images/` directory. You typically need 3-5 high-quality images.

```bash
mkdir -p instance_images
# Add your images here (via upload or download script)
python download_example_images.py robot
```

### 2. Configure and Run Training
The `run.sh` script is the primary way to start training. You can customize the training through environment variables.

```bash
# Set your unique identifiers
export INSTANCE_PROMPT="a photo of sks humanoid robot"
export CLASS_PROMPT="a photo of a humanoid robot"
export MAX_TRAIN_STEPS=1000

# Execute the training pipeline
bash run.sh
```

### 3. Key Configuration Options
| Variable                  | Description                                            | Default                |
| :------------------------ | :----------------------------------------------------- | :--------------------- |
| `INSTANCE_PROMPT`         | Prompt containing your unique identifier (e.g., `sks`) | `a sks humanoid robot` |
| `CLASS_PROMPT`            | Generic prompt for the class of object                 | `a photo of person`    |
| `LEARNING_RATE`           | Learning rate for the UNet                             | `2e-6`                 |
| `MAX_TRAIN_STEPS`         | Total optimization steps                               | `1000`                 |
| `WITH_PRIOR_PRESERVATION` | Enables regularization images to prevent overfitting   | `true`                 |

---

## 🎨 Inference (Using your model)

Once training is complete, the model is saved to `./output/dreambooth-model`. You can generate images using the provided script:

```bash
python generate_images.py \
    --model_path "./output/dreambooth-model" \
    --prompt "a photo of sks humanoid robot on a mountain" \
    --output_path "output.png"
```

## 🛠 Prerequisites
The following dependencies are automatically handled by `run.sh`, but can be manually installed via:
```bash
pip install diffusers transformers accelerate xformers bitsandbytes
```
> [!NOTE]
> On Kaggle, `torch` and `torchvision` are pre-installed; avoid re-installing them to prevent version conflicts.
